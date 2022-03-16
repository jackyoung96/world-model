from email.policy import default
from secrets import choice
import numpy as np
from collections import deque
import pickle
import torch

from ppo import PPODiscreteAgent, PPOContinuousAgent
from utils import collect_trajectories, random_sample
from customEnv import parallelEnv, domainRandeEnv
import argparse
import gym

import os
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def train(episode,args):
    env_name, randomize = args.env, args.randomize

    if not os.path.isdir('save'):
        os.mkdir('save') 
    
    writer = None
    if args.tb_log:
        if not os.path.isdir('tb_log'):
            os.mkdir('tb_log')
        dtime = datetime.now()
        writer = SummaryWriter(os.path.join('tb_log', env_name, dtime.strftime("%y%b%d%H%M%S")))
    if randomize:
        dyn_range = {
            # cartpole
            'masscart': [0.7,1.3], # original 1.0
            'masspole': [0.07, 0.13], # original 0.1
            'length': [0.35, 0.65], # original 0.5
            'force_mag': [7.0, 13.0], # original 10.0

            # pendulum
            'max_torque': [1.4, 2.6], # original 2.0
            'm': [0.7, 1.3], # original 1.0
            'l': [0.7, 1.3], # original 1.0
        }
        default_ratio = 0.1
    else:
        dyn_range = {}
        default_ratio = 0.0


    gamma = .99
    gae_lambda = 0.95
    use_gae = True
    beta = .01
    cliprange = 0.1
    best_score = -np.inf
    goal_score = 495.0

    nenvs = 8
    rollout_length = 200
    minibatches = 10*8
    # Calculate the batch_size
    nbatch = nenvs * rollout_length
    # nbatch = 128
    optimization_epochs = 4
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    
    envs = domainRandeEnv(env_name, nenvs, seed=1234, dyn_range=dyn_range,default_ratio=default_ratio)

    if isinstance(envs.action_space, gym.spaces.Discrete):
        agent = PPODiscreteAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.n, 
                        seed=0,
                        hidden_layers=[64,64],
                        lr_policy=1e-4, 
                        use_reset=True,
                        device=device)
    elif isinstance(envs.action_space, gym.spaces.Box):
        agent = PPOContinuousAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.shape[0], 
                        seed=0,
                        hidden_layers=[128,128],
                        lr_policy=1e-4, 
                        use_reset=False,
                        use_common=False,
                        device=device)
    else:
        raise NotImplementedError
    # print("------------------")
    summary(agent.policy, envs.observation_space.shape)
    # print("------------------")

    # keep track of progress
    mean_rewards = []
    scores_window = deque(maxlen=100)

    for i_episode in range(episode+1):
        log_probs_old, states, actions, rewards, values, dones, vals_last = collect_trajectories(envs, agent.policy, rollout_length)

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        if not use_gae:
            for t in reversed(range(rollout_length)):
                if t == rollout_length - 1:
                    returns[t] = rewards[t] + gamma * (1-dones[t]) * vals_last
                else:
                    returns[t] = rewards[t] + gamma * (1-dones[t]) * returns[t+1]
                advantages[t] = returns[t] - values[t]
        else:
            for t in reversed(range(rollout_length)):
                if t == rollout_length - 1:
                    returns[t] = rewards[t] + gamma * (1-dones[t]) * vals_last
                    td_error = returns[t] - values[t]
                else:
                    returns[t] = rewards[t] + gamma * (1-dones[t]) * returns[t+1]
                    td_error = rewards[t] + gamma * (1-dones[t]) * values[t+1] - values[t]
                advantages[t] = advantages[t] * gae_lambda * gamma * (1-dones[t]) + td_error
        
        # convert to pytorch tensors and move to gpu if available
        returns = torch.from_numpy(returns).float().to(device).view(-1,)
        advantages = torch.from_numpy(advantages).float().to(device).view(-1,)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        loss_storage = {'p':[], 'v':[], 'ent':[]}
        for _ in range(optimization_epochs):
            sampler = random_sample(nbatch, minibatches)
            for inds in sampler:
                mb_log_probs_old = log_probs_old[inds]
                mb_states = states[inds]
                mb_actions = actions[inds]
                mb_returns = returns[inds]
                mb_advantages = advantages[inds]
                loss_p, loss_v, loss_ent = agent.update(mb_log_probs_old, mb_states, mb_actions, mb_returns, mb_advantages, cliprange=cliprange, beta=beta)
                loss_storage['p'].append(loss_p)
                loss_storage['v'].append(loss_v)
                loss_storage['ent'].append(loss_ent)
                
        total_rewards = np.sum(rewards, axis=0)
        scores_window.append(np.mean(total_rewards)) # last 100 scores
        mean_rewards.append(np.mean(total_rewards))  # get the average reward of the parallel environments
        cliprange*=.999                              # the clipping parameter reduces as time goes on
        beta*=.999                                   # the regulation term reduces

        if writer is not None and i_episode%10 == 0:
            writer.add_scalar('loss/loss_p', np.mean(loss_storage['p']), i_episode)
            writer.add_scalar('loss/loss_v', np.mean(loss_storage['v']), i_episode)
            writer.add_scalar('loss/loss_ent', np.mean(loss_storage['ent']), i_episode)
            # writer.add_scalars('loss', {'loss_p': np.mean(loss_storage['p']),
            #                             'loss_v': np.mean(loss_storage['v']),
            #                             'loss_ent': np.mean(loss_storage['ent'])}, i_episode)
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print(total_rewards)
            torch.save(agent.policy.state_dict(), "save/policy_%s.pth"%env_name)

        if np.mean(scores_window)>=goal_score and np.mean(scores_window)>=best_score:    
            torch.save(agent.policy.state_dict(), "save/policy_%s_best.pth"%env_name)
            best_score = np.mean(scores_window)

    return mean_rewards, loss_storage

if __name__=='__main__':
    # train(1000, 'CartPole-v1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, choices=['CartPole-v0','CartPole-v1','Pendulum-v0'])
    parser.add_argument('--randomize',action='store_true', help="Domain randomize")
    parser.add_argument('--tb_log', action='store_true', help="Tensorboard logging")
    args = parser.parse_args()
    train(20000, args)