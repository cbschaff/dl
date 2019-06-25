"""
Environment Wrappers for logging episode stats.
"""

from gym import Wrapper
from baselines.common.vec_env import VecEnvWrapper
from dl import logger
import numpy as np
import time

class EpisodeInfo(Wrapper):
    """
    Pass episode stats through the info dict returned by step().
    If placed before wrappers which modify episode length and reward,
    this will provide easy access to unmodified episode stats.
    """
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0
        self.episode_length = 0
        self.needs_reset = True

    def reset(self):
        self.episode_reward = 0
        self.episode_length = 0
        self.needs_reset = False
        return self.env.reset()

    def step(self, action):
        assert not self.needs_reset, "Can't step environment when the episode ends. Please call reset() first."
        ob, r, done, info = self.env.step(action)
        if done:
            self.needs_reset = True
        self.episode_reward += r
        self.episode_length += 1
        assert 'episode_info' not in info, f"Can't save episode data. Another EpisodeInfo Wrapper exists."
        info['episode_info'] = {}
        info['episode_info']['reward'] = self.episode_reward
        info['episode_info']['length'] = self.episode_length
        info['episode_info']['done'] = done
        return ob, r, done, info


class EpisodeLogger(Wrapper):
    """
    Logs episode stats to TensorBoard. If the env has also been wrapped with
    an EpisodeInfo Wrapper, those stats will also be logged
    under env/unwrapped_episode_*.
    """
    def __init__(self, env, tstart=0):
        super().__init__(env)
        self.t = tstart
        self.eplen = 0
        self.eprew = 0.

    def reset(self, **kwargs):
        self.eplen = 0
        self.eprew = 0.
        return self.env.reset()

    def step(self, action):
        ob, r, done, info = self.env.step(action)
        self.t += 1
        self.eplen += 1
        self.eprew += r
        if done:
            logger.add_scalar('env/episode_length', self.eplen, self.t, time.time())
            logger.add_scalar('env/episode_reward', self.eprew, self.t, time.time())
            self.eplen = 0
            self.eprew = 0.
        # log unwrapped episode stats if they exist
        if 'episode_info' in info:
            epinfo = info['episode_info']
            if epinfo['done']:
                logger.add_scalar('env/unwrapped_episode_length', epinfo['length'], self.t, time.time())
                logger.add_scalar('env/unwrapped_episode_reward', epinfo['reward'], self.t, time.time())

        return ob, r, done, info

class VecEpisodeLogger(VecEnvWrapper):
    """
    EpisodeLogger for vecorized environments.
    """
    def __init__(self, venv, tstart=0):
        super().__init__(venv)
        self.t = tstart
        self.rews = np.zeros(self.num_envs, dtype=np.float32)
        self.lens = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        obs = self.venv.reset()
        self.t += sum(self.lens)
        self.rews = np.zeros(self.num_envs, dtype=np.float32)
        self.lens = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.t += self.num_envs
        self.lens += 1
        self.rews += rews
        for i,done in enumerate(dones):
            if done:
                logger.add_scalar('env/episode_length', self.lens[i], self.t, time.time())
                logger.add_scalar('env/episode_reward', self.rews[i], self.t, time.time())
                self.lens[i] = 0
                self.rews[i] = 0.
        # log unwrapped episode stats if they exist
        if 'episode_info' in infos[0]:
            for info in infos:
                epinfo = info['episode_info']
                if epinfo['done']:
                    logger.add_scalar('env/unwrapped_episode_length', epinfo['length'], self.t, time.time())
                    logger.add_scalar('env/unwrapped_episode_reward', epinfo['reward'], self.t, time.time())

        return obs, rews, dones, infos



if __name__ == '__main__':
    import unittest, shutil
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    import gym

    class Test(unittest.TestCase):
        def test_ep_info(self):
            env = gym.make('PongNoFrameskip-v4')
            env = EpisodeInfo(env)
            env.reset()
            rew = 0
            len = 0
            done = False
            while not done:
                _,r,done,info = env.step(env.action_space.sample())
                len += 1
                rew += r
                assert done == info['episode_info']['done']
                assert len == info['episode_info']['length']
                assert rew == info['episode_info']['reward']

        def test_logger(self):
            logger.configure('./.test')
            env = gym.make('PongNoFrameskip-v4')
            env = EpisodeInfo(env)
            env = EpisodeLogger(env)
            env.reset()
            done = False
            while not done:
                _,_,done,_ = env.step(env.action_space.sample())
            logger.flush()
            shutil.rmtree('./.test')

        def test_vec_logger(self):
            logger.configure('./.test')

            def env_fn(rank=0):
                env = gym.make('PongNoFrameskip-v4')
                env.seed(rank)
                return EpisodeInfo(env)

            def _env(rank):
                def _thunk():
                    return env_fn(rank=rank)
                return _thunk

            nenv = 4
            env = SubprocVecEnv([_env(i) for i in range(nenv)])
            env = VecEpisodeLogger(env)
            env.reset()
            for _ in range(5000):
                env.step([env.action_space.sample() for _ in range(nenv)])
            logger.flush()
            shutil.rmtree('./.test')


    unittest.main()
