from baselines.common.atari_wrappers import *
from gym import ObservationWrapper, Wrapper, ActionWrapper
from baselines.common.vec_env import VecEnvWrapper
from gym.spaces import Box
from dl.util import Monitor
from dl.util import logger
import gin, os
import torch
from collections import deque


class EpsilonGreedy(ActionWrapper):
    def __init__(self, env, epsilon):
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return action

class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, max_history=1000):
        super().__init__(venv)
        self.episode_rewards = deque(maxlen=max_history)
        self.episode_lengths = deque(maxlen=max_history)
        self.rews = np.zeros(self.num_envs, dtype=np.float32)
        self.lens = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        obs = self.venv.reset()
        self.rews = np.zeros(self.num_envs, dtype=np.float32)
        self.lens = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.rews += rews
        self.lens += 1
        for i,done in enumerate(dones):
            if done:
                self.episode_lengths.append(self.lens[i])
                self.episode_rewards.append(self.rews[i])
                self.lens[i] = 0
                self.rews[i] = 0.
        return obs, rews, dones, infos


class FrameStack(Wrapper):
    """
    Stack frames without lazy frames.
    """
    def __init__(self, env, k):
        """Stack k last frames.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        ospace = env.observation_space
        shp = ospace.shape
        self.shape = shp[0]
        self.frames = np.zeros((k*shp[0],*shp[1:]), dtype=ospace.dtype)
        low = np.repeat(ospace.low, self.k, axis=0)
        high = np.repeat(ospace.high, self.k, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=ospace.dtype)

    def reset(self):
        ob = self.env.reset()
        self.frames[:] = 0
        self.frames[-self.shape:] = ob
        return self.frames

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames[:-self.shape] = self.frames[self.shape:]
        self.frames[-self.shape:] = ob
        return self.frames, reward, done, info

class ImageTranspose(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, Box)
        assert len(self.observation_space.shape) == 3
        self.observation_space = Box(self.observation_space.low.transpose(2,0,1), \
                                     self.observation_space.high.transpose(2,0,1),
                                     dtype=self.observation_space.dtype)

    def observation(self, obs):
        return obs.transpose(2,0,1)

@gin.configurable(blacklist=['rank'])
def atari_env(game_name, seed=0, rank=0, sticky_actions=False, timelimit=True, noop=True, frameskip=4, episode_life=True, clip_rewards=True, frame_stack=1, scale=False):
    id = game_name + 'NoFrameskip'
    id += '-v0' if sticky_actions else '-v4'
    env = gym.make(id)
    if not timelimit:
        env = env.env
    assert 'NoFrameskip' in env.spec.id
    if noop:
        env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=frameskip)
    env.seed(seed + rank)
    env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=clip_rewards, frame_stack=False, scale=scale) # call frame stack after transpose
    env = ImageTranspose(env)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    return env



import unittest

class TestAtari(unittest.TestCase):
    def testWrapper(self):
        env = atari_env('Pong', 0, 0, sticky_actions=True)
        assert env.spec.id == 'PongNoFrameskip-v0'
        assert env.observation_space.shape == (1,84,84)
        assert env.reset().shape == (1,84,84)
        env.close()
        logger.reset()
        env = atari_env('Breakout', 0, 0, sticky_actions=False)
        assert env.spec.id == 'BreakoutNoFrameskip-v4'
        assert env.observation_space.shape == (1,84,84)
        assert env.reset().shape == (1,84,84)
        env.close()
        logger.reset()
        env = atari_env('Breakout', 0, 0, sticky_actions=False, frame_stack=4)
        assert env.spec.id == 'BreakoutNoFrameskip-v4'
        assert env.observation_space.shape == (4,84,84)
        assert env.reset().shape == (4,84,84)
        env.close()
        logger.reset()



if __name__ == '__main__':
    unittest.main()
