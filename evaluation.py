from ppo import PPODiscreteAgent, PPOContinuousAgent
import torch
import gym
import matplotlib
from matplotlib import animation, rc
from envs.customEnv import domainRandomize, domainRandeEnv
import argparse
import os
from utils import save_frames_as_gif
from pyvirtualdisplay import Display
import numpy as np
from copy import deepcopy
import pandas as pd

def evaluation(args):
    np.random.seed(args.seed)

    env_name = args.env
    
    # dyn_range = {
    #     # cartpole
    #     'masscart': [0.7,1.3], # original 1.0
    #     'masspole': [0.07, 0.13], # original 0.1
    #     'length': [0.35, 0.65], # original 0.5
    #     'force_mag': [7.0, 13.0], # original 10.0

    #     # pendulum
    #     'max_torque': [1.4, 2.6], # original 2.0
    #     'm': [0.7, 1.3], # original 1.0
    #     'l': [0.7, 1.3], # original 1.0
    # }
    dyn_range = {}
    default_ratio = args.rand_ratio

    device = torch.device("cuda:%d"%args.gpu if torch.cuda.is_available() else "cpu")

    envs = gym.make(env_name)
    envs.seed(args.seed)
    setattr(envs,'env_name', env_name)
    domainRandomize(envs, dyn_range,default_ratio, args.seed)

    if isinstance(envs.action_space, gym.spaces.Discrete):
        ppo_random_agent = PPODiscreteAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.n, 
                        seed=args.seed,
                        hidden_layers=[64,64],
                        lr_policy=1e-4, 
                        use_reset=True,
                        device=device)
        ppo_multitask_agent = PPODiscreteAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.n, 
                        seed=args.seed,
                        hidden_layers=[64,64],
                        lr_policy=1e-4, 
                        use_reset=True,
                        device=device)
        diffdac_agent = PPODiscreteAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.n, 
                        seed=args.seed,
                        hidden_layers=[64,64],
                        lr_policy=1e-4, 
                        use_reset=True,
                        device=device)
    elif isinstance(envs.action_space, gym.spaces.Box):
        ppo_random_agent = PPOContinuousAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.shape[0], 
                        seed=args.seed,
                        hidden_layers=[64,256],
                        lr_policy=1e-4, 
                        use_reset=False,
                        use_common=True,
                        device=device)
        ppo_multitask_agent = PPOContinuousAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.shape[0], 
                        seed=args.seed,
                        hidden_layers=[64,256],
                        lr_policy=1e-4, 
                        use_reset=False,
                        use_common=True,
                        device=device)
        diffdac_agent = PPOContinuousAgent(state_size=envs.observation_space.shape[0],
                        action_size=envs.action_space.shape[0], 
                        seed=args.seed,
                        hidden_layers=[64,256],
                        lr_policy=1e-4, 
                        use_reset=False,
                        use_common=True,
                        device=device)
    else:
        raise NotImplementedError
    
    if args.best:
        ppo_random_agent.policy.load_state_dict(torch.load('save/ppo/randomize/policy_%s_best.pth'%env_name, map_location=device))
        ppo_multitask_agent.policy.load_state_dict(torch.load('save/ppo/multitask/policy_%s_best.pth'%env_name, map_location=device))
        diffdac_agent.policy.load_state_dict(torch.load('save/diffdac/policy_%s_best.pth'%env_name, map_location=device))
    else:
        ppo_random_agent.policy.load_state_dict(torch.load('save/ppo/randomize/policy_%s.pth'%env_name, map_location=device))
        ppo_multitask_agent.policy.load_state_dict(torch.load('save/ppo/multitask/policy_%s.pth'%env_name, map_location=device))
        diffdac_agent.policy.load_state_dict(torch.load('save/diffdac/policy_%s.pth'%env_name, map_location=device))

    ppo_random_agent.policy.eval()
    ppo_multitask_agent.policy.eval()
    diffdac_agent.policy.eval()

    if "CartPole" in env_name:
        df = pd.DataFrame(columns=['algo','masscart','masspole','length','force_mag','reward_sum'])
    elif "Pendulum" in env_name:
        df = pd.DataFrame(columns=['algo','max_torque','m','l','reward_sum'])
    else:
        raise NotImplementedError

    with torch.no_grad():
        for i_eval in range(args.num_eval):

            ppo_random_env = gym.make(env_name)
            ppo_random_env.seed(args.seed+i_eval)
            setattr(ppo_random_env,'env_name', env_name)
            domainRandomize(ppo_random_env, dyn_range,default_ratio, args.seed+i_eval)            
            start_obs =ppo_random_env.reset()

            ppo_multitask_env = deepcopy(ppo_random_env)
            diffdac_env = deepcopy(ppo_random_env)

            

            env_list = [ppo_random_env, ppo_multitask_env, diffdac_env]
            agent_list = [ppo_random_agent, ppo_multitask_agent, diffdac_agent]
            algo_list = ['ppo_random', 'ppo_multitask', 'diffdac']

            for env,agent,algo in zip(env_list, agent_list, algo_list):
                
                done = False
                total_rew = 0
                frames = []
                agent.policy.eval()
                obs = deepcopy(start_obs)

                with torch.no_grad():
                    max_episode_len = 200
                    cartpole_step, cartpole_success = 0,1
                    pendulum_step, pendulum_success = 0,0
                    for _ in range(max_episode_len):
                        if "CartPole" in env_name: 
                            obs_input = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                            action = int(agent.policy.act(obs_input)['a'].cpu().numpy())
                            cartpole_step += 1
                        elif "Pendulum" in env_name:
                            obs_input = torch.from_numpy(obs).float().to(device).view(1,-1)
                            action = agent.policy.act(obs_input)['a'].cpu().numpy()
                            if obs_input[0,0] > 0.98:
                                pendulum_step += 1
                            else:
                                pendulum_step = 0
                            
                            if pendulum_step > 30:
                                pendulum_success = 1
                                break


                        obs, rew, is_done, _ = env.step(action)
                        total_rew += rew
                        if is_done:
                            if not cartpole_step == max_episode_len:
                                cartpole_success = 0
                            break
                
                if "CartPole" in env_name:
                    df = df.append({'algo': algo,
                                'masscart': env.env.masscart,
                                'masspole': env.env.masspole,
                                'length': env.env.length,
                                'force_mag': env.env.force_mag,
                                'reward_sum': total_rew,
                                'success':cartpole_success}, ignore_index=True)
                elif "Pendulum" in env_name:
                    df = df.append({'algo':algo,
                                'max_torque':env.max_torque,
                                'm':env.env.m,
                                'l':env.env.l,
                                'reward_sum':total_rew[0],
                                'success':pendulum_success}, ignore_index=True)
                else:
                    raise NotImplementedError
                
                env.close()
    
    if not os.path.isdir("evaluation"):
        os.mkdir('evaluation')
    df.to_pickle(os.path.join('evaluation',"%s_S%d_N%d_R%d.pkl"%(env_name, args.seed, args.num_eval, int(args.rand_ratio*100))))
    df.to_csv(os.path.join('evaluation',"%s_S%d_N%d_R%d.csv"%(env_name, args.seed, args.num_eval, int(args.rand_ratio*100))))

    print("%s evaluation result"%(env_name))
    print('------ reward sum --------')
    print("PPO-Randomize:",np.mean(df.loc[df['algo']=='ppo_random']['reward_sum']))
    print("PPO-Multitask:",np.mean(df.loc[df['algo']=='ppo_multitask']['reward_sum']))
    print("Diff-DAC:",np.mean(df.loc[df['algo']=='diffdac']['reward_sum']))
    print('------ success rate --------')
    print("PPO-Randomize:",np.mean(df.loc[df['algo']=='ppo_random']['success']))
    print("PPO-Multitask:",np.mean(df.loc[df['algo']=='ppo_multitask']['success']))
    print("Diff-DAC:",np.mean(df.loc[df['algo']=='diffdac']['success']))

def use_data():
    files = [file for file in os.listdir('evaluation') if '.pkl' in file]
    for file in files:
        env_name, seed, num_eval, rand_ratio = file.strip('.pkl').split("_")
        df = pd.read_pickle(os.path.join('evaluation', file))
        print("%s evaluation result (seed: %s, num_eval: %s, random ratio: %s)"%(env_name, seed, num_eval, rand_ratio))
        print('------ reward sum --------')
        print("PPO-Randomize:",np.mean(df.loc[df['algo']=='ppo_random']['reward_sum']))
        print("PPO-Multitask:",np.mean(df.loc[df['algo']=='ppo_multitask']['reward_sum']))
        print("Diff-DAC:",np.mean(df.loc[df['algo']=='diffdac']['reward_sum']))
        print('------ success rate --------')
        print("PPO-Randomize:",np.mean(df.loc[df['algo']=='ppo_random']['success']))
        print("PPO-Multitask:",np.mean(df.loc[df['algo']=='ppo_multitask']['success']))
        print("Diff-DAC:",np.mean(df.loc[df['algo']=='diffdac']['success']))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v0', choices=['CartPole-v0','CartPole-v1','Pendulum-v0'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_eval', default=100, type=int)
    parser.add_argument('--rand-ratio', default=0.3, type=float)
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--use-data', action='store_true')
    args = parser.parse_args()
    with Display(visible=False, size=(100, 60)) as disp:
        if not args.use_data:
            evaluation(args)
        else:
            use_data()