import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import FCNet, ConvNet, PolicyContinuous, PolicyDiscrete

class PPODiscreteAgent:
    def __init__(self, 
                 state_size,
                 action_size, 
                 seed,
                 hidden_layers,
                 lr_policy, 
                 use_reset, 
                 device
                ):

        #self.main_net = ConvNet(state_size, feature_dim, seed, use_reset, input_channel).to(device)
        self.main_net = FCNet(state_size, seed, hidden_layers=[64,64], use_reset=True, act_fnc=F.relu).to(device)
        self.policy = PolicyDiscrete(state_size, action_size, seed, self.main_net).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.device = device

    def update(self, log_probs_old, states, actions, returns, advantages, cliprange=0.1, beta=0.01):
    
        traj_info = self.policy.act(states, actions)
        
        ratio = torch.exp(traj_info['log_pi_a'] - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = 0.5*(returns - traj_info['v']).pow(2).mean()
        entropy = traj_info['ent'].mean()

        self.optimizer.zero_grad()
        (policy_loss + value_loss - beta*entropy).backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
        self.optimizer.step()

        return policy_loss.data.cpu().numpy(), value_loss.data.cpu().numpy(), entropy.data.cpu().numpy()
    
    def act(self, state, action=None):
        return self.policy.act(state,action)


class PPOContinuousAgent:
    def __init__(self, 
                 state_size,
                 action_size, 
                 seed,
                 hidden_layers,
                 lr_policy, 
                 use_reset,
                 use_common, 
                 device
                ):

        #self.main_net = ConvNet(state_size, feature_dim, seed, use_reset, input_channel).to(device)
        if use_common:
            main_net1 = FCNet(state_size, seed, hidden_layers=hidden_layers, use_reset=use_reset, act_fnc=nn.ReLU()).to(device)
            main_net2 = None
        else:
            main_net1 = FCNet(state_size, seed, hidden_layers=hidden_layers, use_reset=use_reset, act_fnc=nn.ReLU()).to(device)
            main_net2 = FCNet(state_size, seed, hidden_layers=hidden_layers, use_reset=use_reset, act_fnc=nn.ReLU()).to(device)
        self.policy = PolicyContinuous(state_size, action_size, seed, main_net1, main_net2).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        # self.optim_actor = optim.Adam([{'params': self.policy.main_net_actor.parameters(), 'weight_decay': 0},
        #                                 {'params': self.policy.fc_actor_mean.parameters(), 'weight_decay': 0},
        #                                 {'params': self.policy.fc_actor_sigma.parameters(), 'weight_decay': 0}],
        #                                 lr = lr_policy)
        # self.optim_value = optim.Adam([{'params': self.policy.main_net_critic.parameters(), 'weight_decay': 0},
        #                                 {'params': self.policy.fc_critic.parameters(), 'weight_decay': 0}],
        #                                 lr = 2*lr_policy)
        self.device = device

    def update(self, log_probs_old, states, actions, returns, advantages, cliprange=0.1, beta=0.01):
        traj_info = self.policy.act(states, actions)

        ratio = torch.exp(traj_info['log_pi_a'] - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # self.optim_actor.zero_grad()
        # policy_loss.backward()
        # self.optim_actor.step()

        # value_loss = 0.5*(returns - traj_info['v']).pow(2).mean()
        
        value_loss = F.mse_loss(returns, traj_info['v'])
        entropy = - beta*traj_info['ent'].mean()

        # self.optim_value.zero_grad()
        # value_loss.backward()
        # self.optim_value.step()

        loss = policy_loss + value_loss + entropy
        # loss = value_loss

        # print("logprob",log_probs_old[:2], traj_info['log_pi_a'][:2], '\n',\
        #         "action",actions[:2], "\n",\
        #         "value",returns[:2], traj_info['v'][:2], '\n',\
        #         "adv",advantages[:2])


        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()


        return policy_loss.data.cpu().numpy(), value_loss.data.cpu().numpy(), entropy.data.cpu().numpy()
    
    def act(self, state, action=None):
        return self.policy.act(state,action)

class vecAgents:
    def __init__(self, 
                 policy,
                 nenvs,
                 graph=None,
                 **kwargs):
        self.agents = [policy(**kwargs) for _ in range(nenvs)]
        self.graph = graph
        if graph is None:
            # Temporary fully connected networks
            self.graph = np.eye(nenvs,nenvs) / nenvs

    def update(self, log_probs_old, states, actions, returns, advantages, cliprange=0.1, beta=0.01):
        policy_loss, value_loss, entropy = [],[],[]

        for i,p in enumerate(self.agents):
            p_loss, v_loss, ent = p.update(log_probs_old[:,i:i+1], \
                                            states[:,i:i+1], \
                                            actions[:,i:i+1], \
                                            returns[:,i], \
                                            advantages[:,i:i+1], \
                                            cliprange, beta)
            policy_loss.append(p_loss)
            value_loss.append(v_loss)
            entropy.append(ent)
        
        # Diffusion
        for i,p in enumerate(self.agents):
            for param, *neighbors in zip(p.policy.parameters(), *map(lambda x: x.policy.parameters(), self.agents)):
                param.data.copy_(sum([self.graph[i][j]*neighbor.data for j,neighbor in enumerate(neighbors)]))
        
        return policy_loss, value_loss, entropy
    
    def act(self, state, action=None):
        actions, logprobs, entropys, vs = [],[],[],[]
        for i,p in enumerate(self.agents):
            a,log_pi_a,ent,v = p.act(state[i:i+1]).values()
            actions.append(a)
            logprobs.append(log_pi_a)
            entropys.append(ent)
            vs.append(v)

        return {'a': torch.concat(actions),
                'log_pi_a': torch.concat(logprobs),
                'ent': torch.concat(entropys),
                'v': torch.stack(vs)}