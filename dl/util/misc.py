import numpy as np
import gin
from dl.util import Monitor


def conv_out_shape(in_shape, conv):
    w = np.array(in_shape)
    f = np.array(conv.kernel_size)
    d = np.array(conv.dilation)
    p = np.array(conv.padding)
    s = np.array(conv.stride)
    df = (f - 1) * d + 1
    return (w - df + 2*p) // s + 1


def load_gin_configs(gin_files, gin_bindings=[]):
    """Loads gin configuration files.
    Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
    """
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False)



def find_wrapper(env, cls):
    if isinstance(env, cls):
        return env
    while hasattr(env, 'env'):
        env = env.env
        if isinstance(env, cls):
            return env
    return None


def find_monitor(env):
    return find_wrapper(env, Monitor)
