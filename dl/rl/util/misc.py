import numpy as np


def conv_out_shape(in_shape, conv):
    w = np.array(in_shape)
    f = np.array(conv.kernel_size)
    d = np.array(conv.dilation)
    p = np.array(conv.padding)
    s = np.array(conv.stride)
    df = (f - 1) * d + 1
    return (w - df + 2*p) // s + 1



def find_wrapper(env, cls):
    if isinstance(env, cls):
        return env
    while hasattr(env, 'env'):
        env = env.env
        if isinstance(env, cls):
            return env
    return None
