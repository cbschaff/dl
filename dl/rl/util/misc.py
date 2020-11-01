"""Misc utilities."""
import numpy as np
from dl.rl.util.vec_env import VecEnv, VecEnvWrapper
from dl import nest
from dl import rl
import torch
from functools import partial


def _discount(x, gamma):
    n = x.shape[0]
    out = torch.zeros_like(x)
    out[-1] = x[-1]
    for ind in reversed(range(n - 1)):
        out[ind] = x[ind] + gamma * out[ind + 1]
    return out


def discount(x, gamma):
    """Return the discounted sum of a sequence."""
    return nest.map_structure(partial(_discount, gamma=gamma), x)


class RewardForwardFilter(object):
    """Transform rewards to running estimate of returns."""

    def __init__(self, gamma):
        self.gamma = gamma
        self.ret = None

    def __call__(self, rews, updates=None):
        if self.ret is None:
            self.ret = rews
        else:
            if updates is None:
                self.ret = self.gamma * self.ret + rews
            else:
                self.ret[updates] *= self.gamma
                self.ret[updates] += rews[updates]
        return self.ret


def conv_out_shape(in_shape, conv):
    """Compute the output shape of a conv layer."""
    w = np.array(in_shape)
    f = np.array(conv.kernel_size)
    d = np.array(conv.dilation)
    p = np.array(conv.padding)
    s = np.array(conv.stride)
    df = (f - 1) * d + 1
    return (w - df + 2*p) // s + 1


def find_wrapper(env, cls):
    """Find an environment wrapper."""
    if isinstance(env, cls):
        return env
    while hasattr(env, 'env') or hasattr(env, 'venv'):
        if hasattr(env, 'env'):
            env = env.env
        else:
            env = env.venv
        if isinstance(env, cls):
            return env
    return None


def is_vec_env(env):
    """Check if env is a VecEnv."""
    return isinstance(env, (VecEnvWrapper, VecEnv))


def ensure_vec_env(env):
    """Wrap env with DummyVecEnv if it is not a VecEnv."""
    if not is_vec_env(env):
        env = rl.envs.DummyVecEnv([lambda: env])
    return env


def _get_env_ob_norm(env, steps):
    ob = env.reset()
    obs = [ob]
    for _ in range(steps):
        ob, _, done, _ = env.step(env.action_space.sample())
        if done:
            ob = env.reset()
        obs.append(ob)
    obs = nest.map_structure(np.stack, nest.zip_structure(*obs))
    mean = nest.map_structure(lambda x: np.mean(x, axis=0), obs)
    std = nest.map_structure(lambda x: np.std(x, axis=0), obs)
    return mean, std


def _get_venv_ob_norm(env, steps):
    ob = env.reset()
    obs = [ob]
    for _ in range(steps // env.num_envs):
        ob, _, done, _ = env.step(
            np.array([env.action_space.sample() for _ in range(env.num_envs)]))
        if np.any(done):
            ob = env.reset(force=False)
        obs.append(ob)
    obs = nest.map_structure(np.concatenate, nest.zip_structure(*obs))
    mean = nest.map_structure(lambda x: np.mean(x, axis=0), obs)
    std = nest.map_structure(lambda x: np.std(x, axis=0), obs)
    return mean, std


def get_ob_norm(env, steps):
    """Get observation normalization constants."""
    if is_vec_env(env):
        return _get_venv_ob_norm(env, steps)
    else:
        return _get_env_ob_norm(env, steps)


def set_env_to_eval_mode(env):
    """Set env and all wrappers to eval mode if available."""
    if hasattr(env, 'eval'):
        env.eval()
    if hasattr(env, 'venv'):
        set_env_to_eval_mode(env.venv)
    elif hasattr(env, 'env'):
        set_env_to_eval_mode(env.env)


def set_env_to_train_mode(env):
    """Set env and all wrappers to train mode if available."""
    if hasattr(env, 'train'):
        env.train()
    if hasattr(env, 'venv'):
        set_env_to_train_mode(env.venv)
    elif hasattr(env, 'env'):
        set_env_to_train_mode(env.env)


def env_state_dict(env):
    def _env_state_dict(env, state_dict, ind):
        """Gather the state of env and all its wrappers into one dict."""
        if hasattr(env, 'state_dict'):
            state_dict[ind] = env.state_dict()
        if hasattr(env, 'venv'):
            state_dict = _env_state_dict(env.venv, state_dict, ind+1)
        elif hasattr(env, 'env'):
            state_dict = _env_state_dict(env.env, state_dict, ind+1)
        return state_dict
    return _env_state_dict(env, {}, 0)


def env_load_state_dict(env, state_dict, ind=0):
    """Load the state of env and its wrapprs."""
    if hasattr(env, 'load_state_dict'):
        env.load_state_dict(state_dict[ind])
    if hasattr(env, 'venv'):
        env_load_state_dict(env.venv, state_dict, ind+1)
    elif hasattr(env, 'env'):
        env_load_state_dict(env.env, state_dict, ind+1)


if __name__ == '__main__':
    import unittest
    from dl.rl.envs import VecEpisodeLogger, VecObsNormWrapper, make_atari_env

    def make_env(nenv):
        """Create a training environment."""
        return VecEpisodeLogger(VecObsNormWrapper(make_atari_env("Pong", nenv)))

    class TestMisc(unittest.TestCase):
        """Test Case."""

        def test_state_and_eval_mode(self):
            """Test."""
            env = make_env(2)
            env.reset()
            assert env.venv.mean.shape == (1, 84, 84)
            assert env.venv.std.shape == (1, 84, 84)
            state = env_state_dict(env)
            assert 0 in state and 1 in state
            state[1]['mean'] = 5
            env_load_state_dict(env, state)
            assert env.venv.mean == 5

            assert not env._eval
            assert not env.venv._eval
            set_env_to_eval_mode(env)
            assert env._eval
            assert env.venv._eval
            set_env_to_train_mode(env)
            assert not env._eval
            assert not env.venv._eval

    unittest.main()
