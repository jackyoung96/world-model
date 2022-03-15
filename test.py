from ppo import PPODiscreteAgent, PPOContinuousAgent
import torch
import gym
import matplotlib
from matplotlib import animation, rc
from customEnv import domainRandomize
import argparse

def test(args):
    env_name, randomize = args.env, args.randomize
    
    if randomize:
        dyn_range = {
        # cartpole
        'masscart': [1.3,1.3], # original 1.0
        'masspole': [0.13, 0.13], # original 0.1
        'length': [0.65, 0.65], # original 0.5
        'force_mag': [15.0, 15.0], # original 10.0

        # pendulum
        'max_torque': [2.6, 2.6], # original 2.0
        'm': [1.3, 1.3], # original 1.0
        'l': [1.3, 1.3], # original 1.0
        }
        default_ratio = 0.1
    else:
        dyn_range = {}
        default_ratio = 0.0

    envs= gym.make(env_name)
    domainRandomize(envs.env, dyn_range)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    agent.policy.load_state_dict(torch.load('save/policy_%s.pth'%env_name, map_location=lambda storage, loc: storage))

    obs = envs.reset()
    done = False
    count = 0
    while not done:
        envs.render()
        obs_input = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        action = agent.policy.act(obs_input)['a'].cpu().numpy()
        obs, _, is_done, _ = envs.step(int(action))
        count += 1
        if is_done:
            print("reward :", count)
            break
    
    envs.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, choices=['CartPole-v0','CartPole-v1','Pendulum-v0'])
    parser.add_argument('--randomize',action='store_true', help="Domain randomize")
    args = parser.parse_args()
    test(args)