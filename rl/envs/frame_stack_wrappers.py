"""Frame stack environment wrappers."""
from baselines.common.vec_env import VecEnvWrapper
from gym.spaces import Box, Tuple
import numpy as np
from dl import nest


class VecFrameStack(VecEnvWrapper):
    """Frame stack wrapper for vectorized environments."""

    def __init__(self, venv, k):
        """Init."""
        super().__init__(venv)
        self.k = k
        self.observation_space, self.frames = self._get_ob_space_and_frames(
                                                        self.observation_space)

    def _get_ob_space_and_frames(self, ob_space):
        if isinstance(ob_space, Tuple):
            spaces, frames = list(zip(*[self._get_ob_space_and_frames(s)
                                        for s in ob_space.spaces]))
            return Tuple(spaces), frames
        elif isinstance(ob_space, Box):
            shp = ob_space.shape
            frames = np.zeros((self.num_envs, self.k*shp[0], *shp[1:]),
                              dtype=ob_space.dtype)
            low = np.repeat(ob_space.low, self.k, axis=0)
            high = np.repeat(ob_space.high, self.k, axis=0)
            space = Box(low=low, high=high, dtype=ob_space.dtype)
            return space, frames
        else:
            raise ValueError("Observation Space must be Box or Tuple."
                             f" Found {type(ob_space)}.")

    def _add_new_observation(self, frames, ob):
        shape = ob.shape[1]
        frames[:, :-shape] = frames[:, shape:]
        frames[:, -shape:] = ob
        return frames

    def reset(self):
        """Reset."""
        ob = self.venv.reset()

        def _zero_frames(frames):
            frames[:] = 0
            return frames

        def _add_ob(item):
            return self._add_new_observation(*item)

        self.frames = nest.map_structure(_zero_frames, self.frames)
        self.frames = nest.map_structure(_add_ob,
                                         nest.zip_structure(self.frames, ob))
        return nest.map_structure(lambda x: x.copy(), self.frames)

    def step(self, action):
        """Step."""
        ob, reward, done, info = self.venv.step(action)

        def _zero_frames(frames):
            for i, d in enumerate(done):
                if d:
                    frames[i] = 0
            return frames

        def _add_ob(item):
            return self._add_new_observation(*item)

        self.frames = nest.map_structure(_zero_frames, self.frames)
        self.frames = nest.map_structure(_add_ob,
                                         nest.zip_structure(self.frames, ob))
        ob = nest.map_structure(lambda x: x.copy(), self.frames)
        return ob, reward, done, info

    def step_wait(self):
        """Step wait."""
        return self.venv.step_wait()

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


if __name__ == '__main__':
    import unittest
    from dl.rl.envs import make_atari_env
    from baselines.common.vec_env import VecEnvWrapper

    def make_env(nenv):
        """Create a training environment."""
        return make_atari_env("Pong", nenv)

    class NestedVecObWrapper(VecEnvWrapper):
        """Nest observations."""

        def __init__(self, venv):
            """Init."""
            super().__init__(venv)
            self.observation_space = Tuple([self.observation_space,
                                            self.observation_space])

        def reset(self):
            """Reset."""
            ob = self.venv.reset()
            return (ob, ob)

        def step_wait(self):
            """Step."""
            ob, r, done, info = self.venv.step_wait()
            return (ob, ob), r, done, info

    class Test(unittest.TestCase):
        """Test."""

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
            while True:
                ob, _, done, _ = env.step([env.action_space.sample()
                                           for _ in range(nenv)])
                assert ob.shape == (nenv, 4*s[0], s[1], s[2])

                for i, d in enumerate(done):
                    if d:
                        assert np.allclose(ob[i, :-1], 0)
                if np.any(done):
                    break

        def test_nested_vec_frame_stack(self):
            """Test vec frame stack wrapper."""
            nenv = 2
            env = make_env(nenv)
            env = NestedVecObWrapper(env)
            env = NestedVecObWrapper(env)
            s = env.observation_space.spaces[0].spaces[0].shape
            env = VecFrameStack(env, 4)
            ob = env.reset()
            assert ob[0][0].shape == (nenv, 4*s[0], s[1], s[2])
            assert ob[0][1].shape == (nenv, 4*s[0], s[1], s[2])
            assert ob[1][0].shape == (nenv, 4*s[0], s[1], s[2])
            assert ob[1][1].shape == (nenv, 4*s[0], s[1], s[2])
            while True:
                ob, _, done, _ = env.step([env.action_space.sample()
                                           for _ in range(nenv)])
                assert ob[0][0].shape == (nenv, 4*s[0], s[1], s[2])
                assert ob[0][1].shape == (nenv, 4*s[0], s[1], s[2])
                assert ob[1][0].shape == (nenv, 4*s[0], s[1], s[2])
                assert ob[1][1].shape == (nenv, 4*s[0], s[1], s[2])

                for i, d in enumerate(done):
                    if d:
                        assert np.allclose(ob[0][0][i, :-1], 0)
                        assert np.allclose(ob[1][0][i, :-1], 0)
                        assert np.allclose(ob[0][1][i, :-1], 0)
                        assert np.allclose(ob[1][1][i, :-1], 0)
                if np.any(done):
                    break

    unittest.main()
