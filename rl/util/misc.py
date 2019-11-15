"""Misc utilities."""
import numpy as np
from baselines.common.vec_env import VecEnv, VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from dl import nest


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
        env = DummyVecEnv([lambda: env])
    return env


def _get_env_ob_norm(env, steps):
    ob = env.reset()
    obs = [ob]
    for _ in range(steps):
        ob, _, done, _ = env.step(env.action_space.sample())
        obs.append(ob)
        if done:
            obs.append(env.reset())
    obs = nest.map_structure(np.stack, nest.zip_structure(*obs))
    mean = nest.map_structure(lambda x: np.mean(x, axis=0), obs)
    std = nest.map_structure(lambda x: np.std(x, axis=0), obs)
    return mean, std


def _get_venv_ob_norm(env, steps):
    ob = env.reset()
    obs = [ob]
    for _ in range(steps):
        ob, r, _, _ = env.step(
            [env.action_space.sample() for _ in range(env.num_envs)])
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


def env_state_dict(env, state_dict={}, ind=0):
    """Gather the state of env and all its wrappers into one dict."""
    if hasattr(env, 'state_dict'):
        state_dict[ind] = env.state_dict()
    if hasattr(env, 'venv'):
        state_dict = env_state_dict(env.venv, state_dict, ind+1)
    elif hasattr(env, 'env'):
        state_dict = env_state_dict(env.env, state_dict, ind+1)
    return state_dict


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
