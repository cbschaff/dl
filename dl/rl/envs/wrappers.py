"""Environment wrappers."""
from gym import Wrapper, ObservationWrapper, ActionWrapper
from baselines.common.vec_env import VecEnvWrapper
from gym.spaces import Box
import numpy as np


class ImageTranspose(ObservationWrapper):
    """Change from HWC to CHW or vise versa."""

    def __init__(self, env):
        """Init."""
        super().__init__(env)
        assert isinstance(self.observation_space, Box)
        assert len(self.observation_space.shape) == 3
        self.observation_space = Box(
            self.observation_space.low.transpose(2, 0, 1),
            self.observation_space.high.transpose(2, 0, 1),
            dtype=self.observation_space.dtype)

    def observation(self, obs):
        """Observation."""
        return obs.transpose(2, 0, 1)


class FrameStack(Wrapper):
    """Stack last k frames along the first dimension.

    For images this means you probably want observations in CHW format.
    """

    def __init__(self, env, k):
        """Init."""
        super().__init__(env)
        self.k = k
        ospace = env.observation_space
        shp = ospace.shape
        self.shape = shp[0]
        self.frames = np.zeros((k*shp[0], *shp[1:]), dtype=ospace.dtype)
        low = np.repeat(ospace.low, self.k, axis=0)
        high = np.repeat(ospace.high, self.k, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=ospace.dtype)

    def reset(self):
        """Reset."""
        ob = self.env.reset()
        self.frames[:] = 0
        self.frames[-self.shape:] = ob
        return self.frames

    def step(self, action):
        """Step."""
        ob, reward, done, info = self.env.step(action)
        self.frames[:-self.shape] = self.frames[self.shape:]
        self.frames[-self.shape:] = ob
        return self.frames, reward, done, info


class VecFrameStack(VecEnvWrapper):
    """Frame stack wrapper for vectorized environments."""

    def __init__(self, venv, k):
        """Init."""
        super().__init__(venv)
        self.k = k
        ospace = self.observation_space
        shp = ospace.shape
        self.shape = shp[0]
        self.frames = np.zeros((self.num_envs, k*shp[0], *shp[1:]),
                               dtype=ospace.dtype)
        low = np.repeat(ospace.low, self.k, axis=0)
        high = np.repeat(ospace.high, self.k, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=ospace.dtype)

    def reset(self):
        """Reset."""
        ob = self.venv.reset()
        self.frames[:] = 0
        self.frames[:, -self.shape:] = ob
        return self.frames

    def step_wait(self):
        """Step."""
        ob, reward, done, info = self.venv.step_wait()
        self.frames[:, :-self.shape] = self.frames[:, self.shape:]
        self.frames[:, -self.shape:] = ob
        return self.frames, reward, done, info


class EpsilonGreedy(ActionWrapper):
    """Epsilon greedy wrapper."""

    def __init__(self, env, epsilon):
        """Init."""
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        """Wrap actions."""
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return action


class VecEpsilonGreedy(VecEnvWrapper):
    """Epsilon greedy wrapper for vectorized environments."""

    def __init__(self, venv, epsilon):
        """Init."""
        super().__init__(venv)
        self.epsilon = epsilon

    def step(self, actions):
        """Wrap actions."""
        if np.random.rand() < self.epsilon:
            actions = [self.action_space.sample() for _ in range(self.num_envs)]
        self.venv.step(actions)

    def step_wait(self):
        """Step."""
        return self.venv.step_wait()

    def reset(self):
        """Reset."""
        return self.venv.reset()


if __name__ == '__main__':
    import unittest
    import gym
    from dl.rl.envs import VecEpisodeLogger, make_atari_env
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    def make_env(nenv):
        """Create a training environment."""
        def _env(rank):
            def _thunk():
                return make_atari_env("Pong", rank=rank)
            return _thunk
        env = SubprocVecEnv([_env(i) for i in range(nenv)],
                            context='fork')
        return VecEpisodeLogger(env)

    class Test(unittest.TestCase):
        """Test."""

        def test_image_transpose(self):
            """Test image transpose wrapper."""
            env = gym.make('PongNoFrameskip-v4')
            s = env.observation_space.shape
            env = ImageTranspose(env)
            ob = env.reset()
            assert ob.shape == (s[2], s[0], s[1])
            ob, _, _, _ = env.step(env.action_space.sample())
            assert ob.shape == (s[2], s[0], s[1])

        def test_frame_stack(self):
            """Test frame stack wrapper."""
            env = gym.make('PongNoFrameskip-v4')
            s = env.observation_space.shape
            env = FrameStack(env, 4)
            ob = env.reset()
            assert ob.shape == (4*s[0], s[1], s[2])
            ob, _, _, _ = env.step(env.action_space.sample())
            assert ob.shape == (4*s[0], s[1], s[2])

            env = gym.make('CartPole-v1')
            s = env.observation_space.shape
            env = FrameStack(env, 4)
            ob = env.reset()
            assert ob.shape == (4*s[0],)
            ob, _, _, _ = env.step(env.action_space.sample())
            assert ob.shape == (4*s[0],)

        def test_vec_frame_stack(self):
            """Test vec frame stack wrapper."""
            nenv = 2
            env = make_env(nenv)
            s = env.observation_space.shape
            env = VecFrameStack(env, 4)
            ob = env.reset()
            assert ob.shape == (nenv, 4*s[0], s[1], s[2])
            ob, _, _, _ = env.step([env.action_space.sample()
                                    for _ in range(nenv)])
            assert ob.shape == (nenv, 4*s[0], s[1], s[2])

    unittest.main()
