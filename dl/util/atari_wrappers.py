from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from gym import ObservationWrapper
from gym.spaces import Box
from dl.util import Monitor
from dl.util import logger
import gin, os



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

@gin.configurable(blacklist=['game_name', 'seed', 'rank'])
def atari_env(game_name, seed=0, rank=0, sticky_actions=False, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    id = game_name + 'NoFrameskip'
    id += '-v0' if sticky_actions else '-v4'
    env = make_atari(id)
    env.seed(seed + rank)
    env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=clip_rewards, frame_stack=frame_stack, scale=scale)
    return ImageTranspose(env)



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



if __name__ == '__main__':
    unittest.main()
