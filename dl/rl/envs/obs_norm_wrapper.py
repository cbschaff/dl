"""
Environment Wrapper for normalizing observations.
"""
from baselines.common.vec_env import VecEnvWrapper
from gym import Wrapper
from dl import logger
from dl.rl.util import get_ob_norm
import numpy as np
import time
import gin

@gin.configurable(blacklist=['norm', 'find_norm_params'])
class ObsNorm(object):
    """
    Observation normalization for vecorized environments.
    Collects data from a random policy and computes a fixed normalization.
    """
    def __init__(self,
                 norm=True,
                 find_norm_params=True,
                 steps=10000,
                 mean=None,
                 std=None,
                 eps=1e-2,
                 log=True,
                 log_prob=0.01,
                ):
        self.steps = steps
        self.should_norm = norm
        self.eps = eps
        self.log = log
        self.log_prob = log_prob
        self.t = 0

        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        elif find_norm_params and self.should_norm:
            self.mean, self.std = get_ob_norm(self.env, self.steps)
        else:
            self.mean = np.zeros(self.observation_space.shape, dtype=np.float32)
            self.std = np.ones(self.observation_space.shape, dtype=np.float32)
        self.std = np.maximum(self.std, eps)

    def _normalize(self, obs):
        if not isinstance(obs, np.ndarray):
            obs = np.asarray(obs)
        if not obs.dtype in [np.float32, np.float64]:
            obs = obs.astype(np.float32)
        return (obs - self.mean) / self.std

    def state_dict(self):
        return {
            'mean': self.mean,
            'std': self.std,
            't': self.t,
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        self.t = state_dict['t']

    def norm_and_log(self, obs):
        if self.should_norm:
            obs = self._normalize(obs)
        if self.log and self.log_prob > np.random.rand():
            percentiles = {
                '10': np.quantile(obs, 0.1),
                '25': np.quantile(obs, 0.25),
                '50': np.quantile(obs, 0.5),
                '75': np.quantile(obs, 0.75),
                '90': np.quantile(obs, 0.9),
            }
            logger.add_scalars('ob_stats/percentiles', percentiles, self.t, time.time())
        return obs


class ObsNormWrapper(Wrapper, ObsNorm):
    def __init__(self, env, *args, **kwargs):
        Wrapper.__init__(self, env)
        ObsNorm.__init__(self, *args, **kwargs)

    def reset(self):
        ob = self.env.reset()
        return self._normalize(ob)

    def step(self, ac):
        ob, r, done, info = self.env.step(ac)
        self.t += 1
        return self.norm_and_log(ob), r, done, info


class VecObsNormWrapper(VecEnvWrapper, ObsNorm):
    def __init__(self, venv, *args, **kwargs):
        VecEnvWrapper.__init__(self, venv)
        self.env = self.venv # make naming consistent...
        ObsNorm.__init__(self, *args, **kwargs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.t += self.num_envs
        return self.norm_and_log(obs), rews, dones, infos

    def reset(self):
        obs = self.venv.reset()
        return self._normalize(obs)


if __name__ == '__main__':
    import unittest
    import shutil
    from dl.rl.envs import make_env
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    def make_vec_env(name, nenv):
        def _env(rank):
            def _thunk():
                return make_env(name, rank=rank)
            return _thunk
        env = SubprocVecEnv([_env(i) for i in range(nenv)])
        return env

    class TestObNorm(unittest.TestCase):
        def test_vec(self):
            logger.configure('./.test')
            env = make_vec_env('CartPole-v1', 10)
            env = VecObsNormWrapper(env, log_prob=1.)
            env.reset()
            assert env.t == 0
            for _ in range(100):
                env.step([env.action_space.sample() for _ in range(10)])
            assert env.t == 1000
            print(env.mean)
            print(env.std)
            shutil.rmtree('./.test')

        def test_env(self):
            logger.configure('./.test')
            env = make_env('CartPole-v1')
            env = ObsNormWrapper(env, log_prob=1.)
            env.reset()
            assert env.t == 0
            for _ in range(100):
                _,_,done,_ = env.step(env.action_space.sample())
                if done:
                    env.reset()
            assert env.t == 100
            print(env.mean)
            print(env.std)
            shutil.rmtree('./.test')

    unittest.main()
