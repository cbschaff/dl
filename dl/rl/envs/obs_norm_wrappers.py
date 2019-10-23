"""Environment Wrapper for normalizing observations."""
from baselines.common.vec_env import VecEnvWrapper
from gym import Wrapper
from dl import logger, nest
from dl.rl.util import get_ob_norm
import numpy as np
import time
import gin


@gin.configurable(blacklist=['norm'])
class ObsNorm(object):
    """Observation normalization for vecorized environments.

    Collects data from a random policy and computes a fixed normalization.
    Computing norm params is done lazily when normalization constants
    are needed and unknown.
    """

    def __init__(self,
                 norm=True,
                 steps=10000,
                 mean=None,
                 std=None,
                 eps=1e-2,
                 log=True,
                 log_prob=0.01):
        """Init."""
        self.steps = steps
        self.should_norm = norm
        self.eps = eps
        self.log = log
        self.log_prob = log_prob
        self.t = 0
        self._eval = False
        self.mean = None
        self.std = None

        if mean is not None and std is not None:
            if not nest.has_same_structure(mean, std):
                raise ValueError("mean and std must have the same structure.")
            self.mean = mean
            self.std = nest.map_structure(
                        lambda x: np.maximum(x, self.eps), std)

    def _env(self):
        return self.env if hasattr(self, 'env') else self.venv

    def find_norm_params(self):
        """Calculate mean and std with a random policy to collect data."""
        mean, std = get_ob_norm(self._env(), self.steps)
        self.mean = mean
        self.std = nest.map_structure(lambda x: np.maximum(x, self.eps), std)

    def _normalize(self, obs):
        if not self.should_norm:
            return obs
        if self.mean is None or self.std is None:
            self.find_norm_params()
        obs = nest.map_structure(np.asarray, obs)
        obs = nest.map_structure(np.float32, obs)
        if not nest.has_same_structure(self.mean, obs):
            raise ValueError("mean and obs do not have the same structure!")

        def norm(item):
            ob, mean, std = item
            return (ob - mean) / std
        return nest.map_structure(norm, nest.zip_structure(obs, self.mean,
                                                           self.std))

    def state_dict(self):
        """State dict."""
        return {
            'mean': self.mean,
            'std': self.std,
            't': self.t,
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        self.t = state_dict['t']

    def norm_and_log(self, obs):
        """Norm observations and log."""
        obs = self._normalize(obs)
        if not self._eval and self.log and self.log_prob > np.random.rand():
            for i, ob in enumerate(nest.flatten(obs)):
                percentiles = {
                    '00': np.quantile(ob, 0.0),
                    '10': np.quantile(ob, 0.1),
                    '25': np.quantile(ob, 0.25),
                    '50': np.quantile(ob, 0.5),
                    '75': np.quantile(ob, 0.75),
                    '90': np.quantile(ob, 0.9),
                    '100': np.quantile(ob, 1.0),
                }
                logger.add_scalars(f'ob_stats/{i}_percentiles', percentiles,
                                   self.t, time.time())
        return obs

    def eval(self):
        """Set the environment to eval mode.

        Eval mode disables logging and stops counting steps.
        """
        self._eval = True

    def train(self):
        """Set the environment to train mode.

        Train mode counts steps and logs obs distribution if self.log is True.
        """
        self._eval = False


class ObsNormWrapper(Wrapper, ObsNorm):
    """Environment wrapper which normalizes obsesrvations."""

    def __init__(self, env, *args, **kwargs):
        """Init."""
        Wrapper.__init__(self, env)
        ObsNorm.__init__(self, *args, **kwargs)

    def reset(self):
        """Reset."""
        ob = self.env.reset()
        return self._normalize(ob)

    def step(self, ac):
        """Step."""
        ob, r, done, info = self.env.step(ac)
        if not self._eval:
            self.t += 1
        return self.norm_and_log(ob), r, done, info


class VecObsNormWrapper(VecEnvWrapper, ObsNorm):
    """Vecotized environment wrapper which normalizes obsesrvations."""

    def __init__(self, venv, *args, **kwargs):
        """Init."""
        VecEnvWrapper.__init__(self, venv)
        ObsNorm.__init__(self, *args, **kwargs)

    def step_wait(self):
        """Step."""
        obs, rews, dones, infos = self.venv.step_wait()
        if not self._eval:
            self.t += self.num_envs
        return self.norm_and_log(obs), rews, dones, infos

    def reset(self):
        """Reset."""
        obs = self.venv.reset()
        return self._normalize(obs)


if __name__ == '__main__':
    import unittest
    import shutil
    from dl.rl.envs import make_env
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from gym import ObservationWrapper
    from gym.spaces import Tuple

    def make_vec_env(name, nenv):
        """Make env."""
        def _env(rank):
            def _thunk():
                return make_env(name, rank=rank)
            return _thunk
        env = SubprocVecEnv([_env(i) for i in range(nenv)])
        return env

    class NestedObWrapper(ObservationWrapper):
        """Nest observations."""

        def __init__(self, env):
            """Init."""
            super().__init__(env)
            self.observation_space = Tuple([self.observation_space,
                                            self.observation_space])

        def observation(self, observation):
            """Duplicate observation."""
            return (observation, observation)

    class TestObNorm(unittest.TestCase):
        """Test."""

        def test_vec(self):
            """Test vec wrapper."""
            logger.configure('./.test')
            nenv = 10
            env = make_vec_env('CartPole-v1', nenv)
            env = VecObsNormWrapper(env, log_prob=1.)
            env.reset()
            assert env.t == 0
            for _ in range(100):
                env.step([env.action_space.sample() for _ in range(nenv)])
            assert env.t == 1000
            state = env.state_dict()
            assert state['t'] == env.t
            assert np.allclose(state['mean'], env.mean)
            assert np.allclose(state['std'], env.std)
            state['t'] = 0
            env.load_state_dict(state)
            assert env.t == 0

            env.eval()
            env.reset()
            for _ in range(10):
                env.step([env.action_space.sample() for _ in range(nenv)])
            assert env.t == 0
            env.train()
            for _ in range(10):
                env.step([env.action_space.sample() for _ in range(nenv)])
            assert env.t == 10 * nenv
            print(env.mean)
            print(env.std)
            shutil.rmtree('./.test')

        def test_env(self):
            """Test wrapper."""
            logger.configure('./.test')
            env = make_env('CartPole-v1')
            env = ObsNormWrapper(env, log_prob=1.)
            env.reset()
            assert env.t == 0
            for _ in range(100):
                _, _, done, _ = env.step(env.action_space.sample())
                if done:
                    env.reset()
            assert env.t == 100
            state = env.state_dict()
            assert state['t'] == env.t
            assert np.allclose(state['mean'], env.mean)
            assert np.allclose(state['std'], env.std)
            state['t'] = 0
            env.load_state_dict(state)
            assert env.t == 0

            env.eval()
            env.reset()
            for _ in range(3):
                env.step(env.action_space.sample())
            assert env.t == 0
            env.train()
            for _ in range(3):
                env.step(env.action_space.sample())
            assert env.t == 3
            print(env.mean)
            print(env.std)
            shutil.rmtree('./.test')

        def test_nested_observations(self):
            """Test nested observations."""
            logger.configure('./.test')
            env = make_env('CartPole-v1')
            env = NestedObWrapper(env)
            env = NestedObWrapper(env)
            env = ObsNormWrapper(env, log_prob=1.)
            env.reset()
            assert env.t == 0
            for _ in range(100):
                _, _, done, _ = env.step(env.action_space.sample())
                if done:
                    env.reset()
            assert env.t == 100
            state = env.state_dict()
            assert state['t'] == env.t
            assert np.allclose(state['mean'], env.mean)
            assert np.allclose(state['std'], env.std)
            state['t'] = 0
            env.load_state_dict(state)
            assert env.t == 0

            env.eval()
            env.reset()
            for _ in range(3):
                env.step(env.action_space.sample())
            assert env.t == 0
            env.train()
            for _ in range(3):
                env.step(env.action_space.sample())
            assert env.t == 3
            print(env.mean)
            print(env.std)
            shutil.rmtree('./.test')

    unittest.main()
