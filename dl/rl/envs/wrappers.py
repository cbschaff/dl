from gym import Wrapper, ObservationWrapper, ActionWrapper
from gym.spaces import Box
import numpy as np



class ImageTranspose(ObservationWrapper):
    """
    Change from HWC to CHW or vise versa.
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, Box)
        assert len(self.observation_space.shape) == 3
        self.observation_space = Box(self.observation_space.low.transpose(2,0,1), \
                                     self.observation_space.high.transpose(2,0,1),
                                     dtype=self.observation_space.dtype)

    def observation(self, obs):
        return obs.transpose(2,0,1)



class FrameStack(Wrapper):
    """
    Stack last k frames along the first dimension. For images this means
    you probably want observations in CHW format.
    """
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        ospace = env.observation_space
        shp = ospace.shape
        self.shape = shp[0]
        self.frames = np.zeros((k*shp[0],*shp[1:]), dtype=ospace.dtype)
        low = np.repeat(ospace.low, self.k, axis=0)
        high = np.repeat(ospace.high, self.k, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=ospace.dtype)

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



class EpsilonGreedy(ActionWrapper):
    """
    Epsilon greedy wrapper
    """
    def __init__(self, env, epsilon):
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return action


if __name__ == '__main__':
    import unittest
    import gym

    class Test(unittest.TestCase):
        def test_image_transpose(self):
            env = gym.make('PongNoFrameskip-v4')
            s = env.observation_space.shape
            env = ImageTranspose(env)
            ob = env.reset()
            assert ob.shape == (s[2], s[0], s[1])
            ob,_,_,_ = env.step(env.action_space.sample())
            assert ob.shape == (s[2], s[0], s[1])

        def test_frame_stack(self):
            env = gym.make('PongNoFrameskip-v4')
            s = env.observation_space.shape
            env = FrameStack(env, 4)
            ob = env.reset()
            assert ob.shape == (4*s[0], s[1], s[2])
            ob,_,_,_ = env.step(env.action_space.sample())
            assert ob.shape == (4*s[0], s[1], s[2])

            env = gym.make('CartPole-v1')
            s = env.observation_space.shape
            env = FrameStack(env, 4)
            ob = env.reset()
            assert ob.shape == (4*s[0],)
            ob,_,_,_ = env.step(env.action_space.sample())
            assert ob.shape == (4*s[0],)



    unittest.main()
