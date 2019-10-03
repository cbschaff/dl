"""Set up gin.configurables for environment creation."""
import baselines.common.atari_wrappers as atari_wrappers
from dl.rl.envs.logging_wrappers import EpisodeInfo
from dl.rl.envs.wrappers import FrameStack, ImageTranspose
import gin
import gym


@gin.configurable(blacklist=['rank'])
def make_atari_env(game_name, seed=0, rank=0, sticky_actions=True,
                   timelimit=True, noop=False, frameskip=4, episode_life=True,
                   clip_rewards=True, frame_stack=1, scale=False):
    """Create an Atari environment."""
    id = game_name + 'NoFrameskip'
    id += '-v0' if sticky_actions else '-v4'
    env = gym.make(id)
    if not timelimit:
        env = env.env
    assert 'NoFrameskip' in env.spec.id
    if noop:
        env = atari_wrappers.NoopResetEnv(env, noop_max=30)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=frameskip)
    env = EpisodeInfo(env)
    env.seed(seed + rank)
    env = atari_wrappers.wrap_deepmind(
        env, episode_life=episode_life, clip_rewards=clip_rewards,
        frame_stack=False, scale=scale)  # call frame stack after transpose
    env = ImageTranspose(env)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    return env


@gin.configurable(blacklist=['rank'])
def make_env(env_id, seed=0, rank=0):
    """Create an environment."""
    env = gym.make(env_id)
    env = EpisodeInfo(env)
    env.seed(seed + rank)
    return env


if __name__ == '__main__':
    import unittest

    class TestEnvFns(unittest.TestCase):
        """Test."""

        def test_atari(self):
            """Test atari fn."""
            env = make_atari_env('Pong', 0, 0, sticky_actions=True)
            assert env.spec.id == 'PongNoFrameskip-v0'
            assert env.observation_space.shape == (1, 84, 84)
            assert env.reset().shape == (1, 84, 84)
            assert 'episode_info' in env.step(env.action_space.sample())[3]
            env.close()
            env = make_atari_env('Breakout', 0, 0, sticky_actions=False)
            assert env.spec.id == 'BreakoutNoFrameskip-v4'
            assert env.observation_space.shape == (1, 84, 84)
            assert env.reset().shape == (1, 84, 84)
            assert 'episode_info' in env.step(env.action_space.sample())[3]
            env.close()
            env = make_atari_env('Breakout', 0, 0, sticky_actions=False,
                                 frame_stack=4)
            assert env.spec.id == 'BreakoutNoFrameskip-v4'
            assert env.observation_space.shape == (4, 84, 84)
            assert env.reset().shape == (4, 84, 84)
            assert 'episode_info' in env.step(env.action_space.sample())[3]
            env.close()

        def test_env(self):
            """Test env fn."""
            env = make_env('CartPole-v1')
            env.reset()
            assert 'episode_info' in env.step(env.action_space.sample())[3]

    unittest.main()
