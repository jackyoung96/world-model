import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.multiprocessing as mp

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

        # print("logprob",log_probs_old[:2], traj_info['log_pi_a'][:2], '\n',\
        #         "action",actions[:2], "\n",\
        #         "value",returns[:2], traj_info['v'][:2], '\n',\
        #         "adv",advantages[:2])

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
        self.device = device

    def update(self, log_probs_old, states, actions, returns, advantages, cliprange=0.1, beta=0.01):
        # print(log_probs_old.shape, states.shape, actions.shape, returns.shape, advantages.shape)
        
        traj_info = self.policy.act(states, actions)

        ratio = torch.exp(traj_info['log_pi_a'] - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # value_loss = 0.5*(returns - traj_info['v']).pow(2).mean()
        value_loss = F.mse_loss(returns, traj_info['v'])
        entropy = - beta*traj_info['ent'].mean()


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

def worker(procnum, qout, agent, states):
    out = agent.act(states).values()
    qout.put(obj=(procnum, out))
    del out

class Worker(mp.Process):
    def __init__(self,agent, num_proc, state, res_queue):
        super(Worker, self).__init__()
        self.agent = agent
        self.state = state
        self.res_queue = res_queue
        self.num_proc = num_proc
        self.daemon = True

    def run(self):
        out = self.agent.act(self.state)
        self.res_queue.put((self.num_proc, out))

class vecAgents:
    def __init__(self, 
                 policy,
                 nenvs,
                 graph=None,
                 **kwargs):
        self.agents = [policy(**kwargs) for _ in range(nenvs)]
        for agent in self.agents:
            agent.policy.share_memory()
        self.graph = graph
        if graph is None:
            # Temporary fully connected networks
            self.graph = np.ones((nenvs,nenvs)) / (nenvs)

    def update(self, log_probs_old, states, actions, returns, advantages, cliprange=0.1, beta=0.01):
        policy_loss, value_loss, entropy = np.zeros((len(self.agents),)),\
                                            np.zeros((len(self.agents),)),\
                                            np.zeros((len(self.agents),))

        for i,p in enumerate(self.agents):
            p_loss, v_loss, ent = p.update(log_probs_old[:,i], 
                                            states[:,i], 
                                            actions[:,i], 
                                            returns[:,i], 
                                            advantages[:,i], 
                                            cliprange, beta)
            policy_loss[i] = p_loss
            value_loss[i] = v_loss
            entropy[i] = ent
        
        # # Diffusion
        # for i,p in enumerate(self.agents):
        #     for param, *neighbors in zip(p.policy.parameters(), *map(lambda x: x.policy.parameters(), self.agents)):
        #         param.data.copy_(sum([self.graph[i][j]*neighbor.data for j,neighbor in enumerate(neighbors)]))
        
        return policy_loss, value_loss, entropy
    
    def act(self, state, action=None):
        actions, logprobs, entropys, vs = [],[],[],[]

        qout = mp.Queue()
        state.share_memory_()
        workers = [Worker(agent.policy, i, state[i:i+1], qout) for i,agent in enumerate(self.agents)]
        [w.start() for w in workers]
        [w.join() for w in workers]
        
        unsorted_queue = [qout.get() for w in workers]
        return_queue = [t[1] for t in sorted(unsorted_queue)]
        del unsorted_queue

        for i in range(len(self.agents)):
            a,log_pi_a,ent,v = return_queue[i]
            actions.append(a)
            logprobs.append(log_pi_a)
            entropys.append(ent)
            vs.append(v)

        return {'a': torch.concat(actions),
                'log_pi_a': torch.concat(logprobs),
                'ent': torch.concat(entropys),
                'v': torch.stack(vs)}