import torch.nn as nn
import gin
from dl import logger
from dl.rl.modules.base import ValueFunctionBase
from collections import namedtuple

@gin.configurable(whitelist=['base'])
class ValueFunction(nn.Module):
    def __init__(self, env, base):
        super().__init__()
        self.base = base(env.observation_space, env.action_space)
        assert isinstance(self.base, ValueFunctionBase)
        self.outputs = namedtuple('Outputs', ['value'])

    def log_graph(self, ob):
        if logger.get_summary_writer() is None:
            return
        logger.add_graph(self, ob)

    def forward(self, ob):
        value = self.base(ob).squeeze(-1)
        return self.outputs(value=value)



if __name__ == '__main__':
    import unittest
    import torch, gym
    import numpy as np

    class TestVF(unittest.TestCase):
        def test(self):
            class Base(ValueFunctionBase):
                def forward(self, ob):
                    return torch.from_numpy(np.zeros((ob.shape[0], 1)))

            env = gym.make('CartPole-v1')
            vf = ValueFunction(env, Base)
            ob = env.reset()

            assert np.allclose(vf(ob[None]).value, 0)

    unittest.main()
