"""Environment wrappers."""
from gym import ObservationWrapper, ActionWrapper
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
        return self.venv.step(actions)

    def step_wait(self):
        """Step."""
        return self.venv.step_wait()

    def reset(self):
        """Reset."""
        return self.venv.reset()


if __name__ == '__main__':
    import unittest
    import gym

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

    unittest.main()
