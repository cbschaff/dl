"""Misc utilities."""
import numpy as np
from baselines.common.vec_env import VecEnv, VecEnvWrapper


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


def _get_env_ob_norm(env, steps):
    ob = env.reset()
    obs = [ob]
    for _ in range(steps):
        ob, _, done, _ = env.step(env.action_space.sample())
        obs.append(ob)
        if done:
            obs.append(env.reset())
    obs = np.stack(obs, axis=0)
    return obs.mean(axis=0), obs.std(axis=0)


def _get_venv_ob_norm(env, steps):
    ob = env.reset()
    obs = [ob]
    for _ in range(steps):
        ob, r, _, _ = env.step(
            [env.action_space.sample() for _ in range(env.num_envs)])
        obs.append(ob)
    obs = np.concatenate(obs, axis=0)
    return obs.mean(axis=0), obs.std(axis=0)


def get_ob_norm(env, steps):
    """Get observation normalization constants."""
    if is_vec_env(env):
        return _get_venv_ob_norm(env, steps)
    else:
        return _get_env_ob_norm(env, steps)
