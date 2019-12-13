"""Adapt SubprocVecEnv to save random state and environment state."""

import multiprocessing as mp

import numpy as np
from baselines.common.vec_env.vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars
from dl import rng
from dl.rl import env_state_dict, env_load_state_dict


def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, info

    parent_remote.close()
    rng.seed(0)
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                remote.send([env.render(mode='rgb_array') for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(CloudpickleWrapper((envs[0].observation_space, envs[0].action_space, envs[0].spec)))
            elif cmd == 'get_rng':
                remote.send(rng.get_state())
            elif cmd == 'get_state':
                remote.send([env_state_dict(env) for env in envs])
            elif cmd == 'set_rng':
                rng.set_state(data)
            elif cmd == 'set_state':
                for env, state in zip(envs, data):
                    env_load_state_dict(env, state)
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn', in_series=1):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = _flatten_list(imgs)
        return imgs

    def state_dict(self):
        for remote in self.remotes:
            remote.send(('get_rng', None))
        rng_states = [remote.recv() for remote in self.remotes]

        for remote in self.remotes:
            remote.send(('get_state', None))
        env_states = [remote.recv() for remote in self.remotes]

        return {'rng_states': rng_states,  # nremotes rng_states
                'env_states': env_states}  # nenv env_states

    def load_state_dict(self, state_dict):
        for remote, rng_state in zip(self.remotes, state_dict['rng_states']):
            remote.send(('set_rng', rng_state))
        for remote, env_states in zip(self.remotes, state_dict['env_states']):
            remote.send(('set_state', env_states))

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()


def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)


def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]


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
        return SubprocVecEnv([_env(i) for i in range(nenv)], context='fork')

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
            env.reset()
            env3.reset()
            for _ in range(100):
                actions = [env.action_space.sample() for _ in range(nenv)]
                ob, r, done, _ = env.step(actions)
                ob3, r3, done3, _ = env3.step(actions)
                assert np.allclose(ob, ob3)
                assert np.allclose(r, r3)
                assert np.allclose(done, done3)

    unittest.main()
