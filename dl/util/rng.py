import random
import numpy as np
import torch

SEED = None

def seed(seed):
    global SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    SEED = seed

def get_state():
    s = {}
    s['torch']  = torch.get_rng_state()
    s['numpy']  = np.random.get_state()
    s['random'] = random.getstate()
    return s

def set_state(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['random'])


import unittest
class TestRandom(unittest.TestCase):
    def test(self):
        seed(0)
        state = get_state()
        r1 = torch.rand([10])
        r2 = np.random.rand(10)
        r3 = [random.random() for _ in range(10)]

        set_state(state)
        r1_copy = torch.rand([10])
        r2_copy = np.random.rand(10)
        r3_copy = [random.random() for _ in range(10)]

        assert np.allclose(r1.numpy(),r1_copy.numpy())
        assert np.allclose(r2, r2_copy)
        assert np.allclose(r3, r3_copy)


if __name__ == '__main__':
    unittest.main()
