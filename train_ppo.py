import numpy as np
from collections import deque
import torch

from ppo import PPODiscreteAgent, PPOContinuousAgent
from utils import collect_trajectories, random_sample
from envs.customEnv import parallelEnv, domainRandeEnv

from envs.customEnvDrone import customAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary

import argparse
import gym

import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train(args):
    env_name = args.env
    tag = 'simple'
    if args.randomize:
        tag = "randomize"
    elif args.multitask:
        tag = "multitask"

    savepath = "save/ppo/%s"%tag

    if not os.path.isdir(savepath):
        os.mkdir(savepath) 
    
    writer = None
    if args.tb_log:
        if not os.path.isdir('tb_log'):
            os.mkdir('tb_log')
        dtime = datetime.now()
        tbpath = "ppo/%s"%tag
        writer = SummaryWriter(os.path.join('tb_log', env_name, tbpath, dtime.strftime("%y%b%d%H%M%S")))
    if args.randomize or args.multitask:
        dyn_range = {
            # cartpole
            # 'masscart': [0.7,1.3], # original 1.0
            # 'masspole': [0.07, 0.13], # original 0.1
            # 'length': [0.35, 0.65], # original 0.5
            # 'force_mag': [7.0, 13.0], # original 10.0
            'masscart': 0.5, # original 1.0
            'masspole': 0.5, # original 0.1
            'length': 0.5, # original 0.5
            'force_mag': 0.5, # original 10.0

            # pendulum
            # 'max_torque': [1.4, 2.6], # original 2.0
            # 'm': [0.7, 1.3], # original 1.0
            # 'l': [0.7, 1.3], # original 1.0
            'max_torque': 0.5, # original 2.0
            'm': 0.5, # original 1.0
            'l': 0.5, # original 1.0

            # drones
            'mass_range': 0.3,
            'cm_range': 0.3,
            'kf_range': 0.1,
            'km_range': 0.1,
        }
        default_ratio = 0.3
    else:
        dyn_range = {}
        default_ratio = 0.0


    gamma = 0.99
    gae_lambda = 0.95
    use_gae = False
    beta = 0.01
    cliprange = 0.2

    nenvs = 25
    
    minibatches = 128
    # Calculate the batch_size
    
    # nbatch = 128
    optimization_epochs = 10
    best_score = -np.inf
    
    device=torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    envs = domainRandeEnv(env_name=env_name, tag=tag, n=nenvs, randomize=args.randomize, seed=100000, dyn_range=dyn_range,default_ratio=default_ratio)


    if 'CartPole' in env_name:
        agent = PPODiscreteAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.n, 
                        seed=0,
                        hidden_layers=[64,64],
                        lr_policy=1e-4, 
                        use_reset=True,
                        device=device)
        rollout_length = 500
        nbatch = nenvs * rollout_length
        epoch = 2000
    elif 'Pendulum' in env_name:
        agent = PPOContinuousAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.shape[0], 
                        seed=0,
                        hidden_layers=[64,256],
                        lr_policy=2e-5, 
                        use_reset=False,
                        use_common=True,
                        device=device)
        rollout_length = 200
        nbatch = nenvs * rollout_length
        epoch = 5000
    elif 'aviary' in env_name:
        agent = PPOContinuousAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.shape[0], 
                        seed=0,
                        hidden_layers=[64,256],
                        lr_policy=2e-5, 
                        use_reset=False,
                        use_common=True,
                        device=device)
        rollout_length = 400
        nbatch = nenvs * rollout_length
        epoch = 20000
    else:
        raise NotImplementedError
    # print("------------------")
    # summary(agent.policy, envs.observation_space.shape, device=device)
    # print("------------------")

    # keep track of progress
    mean_rewards = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1,epoch+1):
        log_probs_old, states, actions, rewards, values, dones, vals_last = collect_trajectories(envs, agent, rollout_length)

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        if not use_gae:
            for t in reversed(range(rollout_length)):
                if t == rollout_length - 1:
                    returns[t] = rewards[t] + gamma * (1-dones[t]) * vals_last
                else:
                    returns[t] = rewards[t] + gamma * (1-dones[t]) * values[t+1]
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
        if not 'aviary' in env_name:
            returns = torch.from_numpy(returns).float().to(device).view(-1,)
            advantages = torch.from_numpy(advantages).float().to(device).view(-1,)
        else:
            returns = torch.from_numpy(returns).float().to(device).view(-1,)
            advantages = torch.from_numpy(advantages).float().to(device).view(-1,1)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
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
        # cliprange*=.999                              # the clipping parameter reduces as time goes on
        beta*=.9995                                  # the regulation term reduces

        if writer is not None and i_episode%10 == 0:
            writer.add_scalar('loss/loss_p', np.mean(loss_storage['p']), i_episode)
            writer.add_scalar('loss/loss_v', np.mean(loss_storage['v']), i_episode)
            writer.add_scalar('loss/loss_ent', np.mean(loss_storage['ent']), i_episode)
            writer.add_scalar('rewards', scores_window[-1], i_episode)
            # writer.add_scalars('loss', {'loss_p': np.mean(loss_storage['p']),
            #                             'loss_v': np.mean(loss_storage['v']),
            #                             'loss_ent': np.mean(loss_storage['ent'])}, i_episode)
        

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores_window[-1]))
            torch.save(agent.policy.state_dict(), "%s/policy_%s_iter%05d.pth"%(savepath,env_name,i_episode))

        if np.mean(scores_window)>=best_score:    
            torch.save(agent.policy.state_dict(), "%s/policy_%s_best.pth"%(savepath,env_name))
            best_score = np.mean(scores_window)

    return mean_rewards, loss_storage

def train_stablebaseline():
    pass

if __name__=='__main__':
    # train(1000, 'CartPole-v1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, choices=['CartPole-v0','CartPole-v1','Pendulum-v0','takeoff-aviary-v0'])
    parser.add_argument('--multitask',action='store_true', help="Multitask")
    parser.add_argument('--randomize',action='store_true', help="Domain randomize")
    parser.add_argument('--tb_log', action='store_true', help="Tensorboard logging")
    parser.add_argument('--gpu', default='0', type=int, help="gpu number")
    args = parser.parse_args()
    if args.multitask and args.randomize:
        raise "Only give an option between multitask and randomize"
    train(args)