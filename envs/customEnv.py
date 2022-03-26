from email.policy import default
import gym

import numpy as np
import gym
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from .customEnvDrone import customAviary, domainRandomAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
import gym_pybullet_drones

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        #logger.warn('Render not defined for %s' % self)
        pass
        
    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class parallelEnv(VecEnv):
    def __init__(self, env_name='PongDeterministic-v4',
                 n=4, seed=None,
                 spaces=None):

        env_fns = [ gym.make(env_name) for _ in range(n) ]

        if seed is not None:
            for i,e in enumerate(env_fns):
                e.seed(i+seed)
        
        """
        envs: list of gym environments to run in subprocesses
        adopted from openai baseline
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True




# Random physical properties env
def domainRandomize(env,dyn_range=dict(),default_ratio=0.1,seed=None):
    def default_range(dyn, ratio):
        return [dyn*(1-ratio), dyn*(1+ratio)]
    if seed is not None:
        np.random.seed(seed)
    env_name = env.env_name
    if 'CartPole' in env_name:
        # env.env.masscart = np.random.uniform(* (dyn_range.get('masscart',default_range(env.env.masscart,default_ratio))) ) 
        env.env.masscart = np.random.uniform(*default_range(env.env.masscart,dyn_range.get('masscart',default_ratio))) 
        # env.env.masspole = np.random.uniform(* (dyn_range.get('masspole',default_range(env.env.masspole,default_ratio))) )
        env.env.masspole = np.random.uniform(*default_range(env.env.masspole,dyn_range.get('masspole',default_ratio))) 
        env.env.total_mass = env.env.masspole + env.env.masscart
        # env.env.length = np.random.uniform(* (dyn_range.get('length',default_range(env.env.length,default_ratio))) )  # actually half the pole's length
        env.env.length = np.random.uniform(*default_range(env.env.length,dyn_range.get('length',default_ratio))) 
        env.env.polemass_length = env.env.masspole * env.env.length
        # env.env.force_mag = np.random.uniform(* (dyn_range.get('force_mag',default_range(env.env.force_mag,default_ratio))) )
        env.env.force_mag = np.random.uniform(*default_range(env.env.force_mag,dyn_range.get('force_mag',default_ratio))) 
    elif 'Pendulum' in env_name:
        # env.env.max_torque = np.random.uniform(* (dyn_range.get('max_torque',default_range(env.env.max_torque,default_ratio))) ) 
        env.env.max_torque = np.random.uniform(*default_range(env.env.max_torque,dyn_range.get('max_torque',default_ratio))) 
        # env.env.m = np.random.uniform(* (dyn_range.get('m',default_range(env.env.m,default_ratio))) ) 
        env.env.m = np.random.uniform(*default_range(env.env.m,dyn_range.get('m',default_ratio))) 
        # env.env.l = np.random.uniform(* (dyn_range.get('l',default_range(env.env.l,default_ratio))) ) 
        env.env.l = np.random.uniform(*default_range(env.env.l,dyn_range.get('l',default_ratio)))
    elif 'aviary' in env_name:
        env.random_urdf()
    else:
        raise NotImplementedError

# multithreading 
def workerDomainRand(remote, parent_remote, env_fn_wrapper, randomize, dyn_range, default_ratio, idx):
    parent_remote.close()
    env = env_fn_wrapper.x
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            if randomize:
                domainRandomize(env, dyn_range,default_ratio, idx+np.random.randint(2147483647))
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class domainRandeEnv(parallelEnv):
    '''
    Domain randomize environment
    '''
    def __init__(self, 
                env_name='CartPole-v1',
                tag='simple',
                n=4, seed=0,
                randomize=False, # True: domain randomize(=Randomize every reset)
                dyn_range=None, # physical properties range
                default_ratio=0.1
                ):
        
        self.env_name = env_name
        self.default_ratio = default_ratio
        if not 'aviary' in env_name:
            env_fns = [ gym.make(env_name) for _ in range(n) ]
            for i, env_fn in enumerate(env_fns):
                setattr(env_fn, 'env_name', env_name)
                domainRandomize(env=env_fn, dyn_range=dyn_range, default_ratio=default_ratio, seed=i+seed)
                env_fn.seed(i+seed)
        else:
            env_fns = []
            for idx in range(n):
                env = gym.make(id=env_name, # arbitrary environment that has state normalization and clipping
                    drone_model=DroneModel.CF2X,
                    initial_xyzs=np.array([[0.0,0.0,2.0]]),
                    initial_rpys=np.array([[0.0,0.0,0.0]]),
                    physics=Physics.PYB_GND_DRAG_DW,
                    freq=240,
                    aggregate_phy_steps=1,
                    gui=False,
                    record=False, 
                    obs=ObservationType.KIN,
                    act=ActionType.RPM)
                env = domainRandomAviary(env, tag, idx, seed,
                    observable=['pos', 'rotation', 'vel', 'angular_vel', 'rpm'],
                    frame_stack=1,
                    task='stabilize2',
                    reward_coeff={'xyz':0.2, 'vel':0.016, 'ang_vel':0.08, 'd_action':0.002},
                    episode_len_sec=2,
                    max_rpm=66535,
                    initial_xyz=[[0.0,0.0,50.0]], # Far from the ground
                    freq=200,
                    rpy_noise=1.2,
                    vel_noise=1.0,
                    angvel_noise=2.4,
                    mass_range=dyn_range.get('mass_range', 0.0),
                    cm_range=dyn_range.get('cm_range', 0.0),
                    kf_range=dyn_range.get('kf_range', 0.0),
                    km_range=dyn_range.get('km_range', 0.0))
                setattr(env, 'env_name', env_name)
                env_fns.append(env)
        
        """
        envs: list of gym environments to run in subprocesses
        adopted from openai baseline
        """
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [Process(target=workerDomainRand, args=(work_remote, remote, CloudpickleWrapper(env_fn),randomize,dyn_range,default_ratio, idx))
            for (work_remote, remote, env_fn, idx) in zip(self.work_remotes, self.remotes, env_fns, range(len(env_fns)))]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        self.env_step = 0
        

    def reset(self):
        self.env_step = 0
        return super().reset()

    def step(self, action):
        """
        Reward normalization
        """
        self.env_step += 1
        if "CartPole" in self.env_name:
            obs, reward, is_done, info = super().step(action)
            reward = reward - np.clip(np.power(obs[:,0]/2.4,2),0,1)
            return obs, reward, is_done, info
        elif "Pendulum" in self.env_name:
            obs, reward, is_done, info = super().step(action)
            reward = (reward + 8.1) / 8.1
            return obs, reward, is_done, info
        elif 'aviary' in self.env_name:
            action = action.reshape((self.nenvs,-1))
            obs, reward, is_done, info = super().step(action)
            return obs, reward, is_done, info
        else:
            raise NotImplementedError