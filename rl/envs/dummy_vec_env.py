"""Adapt SubprocVecEnv to save environment state."""

import numpy as np
from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from dl.rl import env_state_dict, env_load_state_dict

class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        """
        Arguments:
        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.spec = self.envs[0].spec

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            # if isinstance(self.envs[e].action_space, spaces.Discrete):
            #    action = int(action)

            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def state_dict(self):
        env_states = []
        for e in self.envs:
            env_states.append(env_state_dict(e))
        return {'env_states': env_states}

    def load_state_dict(self, state_dict):
        for e, state in zip(self.envs, state_dict['env_states']):
            env_load_state_dict(e, state)


if __name__ == "__main__":
    import unittest
    import gym
    from gym import Wrapper

    class StateWrapper(Wrapper):
        # hack to save random state from lunar lander env.
        def __init__(self, env):
            super().__init__(env)

        def step(self, action):
            return self.env.step(action)

        def state_dict(self):
            return {'rng': self.env.np_random.get_state()}

        def load_state_dict(self, state_dict):
            self.env.np_random.set_state(state_dict['rng'])

    def make_env(nenv, seed=0):
        def _env(rank):
            def _thunk():
                env = gym.make('LunarLander-v2')
                env = StateWrapper(env)
                env.seed(seed + rank)
                return env
            return _thunk
        return DummyVecEnv([_env(i) for i in range(nenv)])

    class TestSubprocVecEnv(unittest.TestCase):
        """Test SubprocVecEnv"""

        def test(self):
            nenv = 4
            env = make_env(nenv)
            obs = env.reset()
            env2 = make_env(nenv)
            obs2 = env2.reset()
            env3 = make_env(nenv, seed=1)
            obs3 = env3.reset()

            assert np.allclose(obs, obs2)
            assert not np.allclose(obs, obs3)

            for _ in range(100):
                actions = [env.action_space.sample() for _ in range(nenv)]
                ob, r, done, _ = env.step(actions)
                ob2, r2, done2, _ = env2.step(actions)
                assert np.allclose(ob, ob2)
                assert np.allclose(r, r2)
                assert np.allclose(done, done2)

            env3.load_state_dict(env.state_dict())
            ob = env.reset()
            ob3 = env3.reset()
            assert np.allclose(ob, ob3)

            for _ in range(100):
                actions = [env.action_space.sample() for _ in range(nenv)]
                ob, r, done, _ = env.step(actions)
                ob3, r3, done3, _ = env3.step(actions)
                assert np.allclose(ob, ob3)
                assert np.allclose(r, r3)
                assert np.allclose(done, done3)

    unittest.main()
