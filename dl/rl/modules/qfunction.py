import torch.nn as nn
import gin
from dl import logger
from dl.rl.modules.base import DiscreteQFunctionBase, ContinuousQFunctionBase
from collections import namedtuple




@gin.configurable(whitelist=['base'])
class QFunction(nn.Module):
    def __init__(self, env, base):
        super().__init__()
        self.base = base(env.observation_space, env.action_space)
        assert isinstance(self.base, (DiscreteQFunctionBase, ContinuousQFunctionBase))
        self.discrete = isinstance(self.base, DiscreteQFunctionBase)
        self.outputs = namedtuple('Outputs', ['action', 'value', 'max_a', 'max_q', 'qvals'])

    def log_graph(self, ob, ac=None):
        if logger.get_summary_writer() is None:
            return
        if self.discrete:
            logger.add_graph(self.base, ob)
        else:
            assert ac is not None, "You must provide an action for a continuous action space"
            logger.add_graph(self.base, (ob,ac))

    def forward(self, ob, action=None):
        """
        Computes Q-value.
        Returns:
            out (namedtuple):
                out.action: If an action is specified, out.action is the same, otherwise it is the argmax of the Q-values
                out.value: The q value of (x, out.action)
                out.max_a:  The argmax of the Q-values   (only available for discrete action spaces)
                out.max_q:  The max of the Q-values      (only available for discrete action spaces)
                out.qvals:  The Q-value for each action  (only available for discrete action spaces)
        """
        if action is None:
            assert self.discrete, "You must provide an action for a continuous action space"
            qvals = self.base(ob)
            maxq, maxa = qvals.max(dim=-1)
            return self.outputs(action=maxa, value=maxq, max_a=maxa, max_q=maxq, qvals=qvals)
        elif self.discrete:
            qvals = self.base(ob)
            maxq, maxa = qvals.max(dim=-1)
            action = action.long().unsqueeze(1) if len(action.shape) == 1 else action.long()
            value = qvals.gather(1, action).squeeze(1)
            return self.outputs(action=action.squeeze(1), value=value, max_a=maxa, max_q=maxq, qvals=qvals)
        else:
            value = self.base(ob, action).squeeze(1)
            return self.outputs(action=action, value=value, max_a=None, max_q=None, qvals=None)


if __name__ == '__main__':
    import unittest
    import torch, gym
    import numpy as np

    class TestQF(unittest.TestCase):
        def test_discrete(self):
            class Base(DiscreteQFunctionBase):
                def forward(self, ob):
                    return torch.from_numpy(np.random.rand(ob.shape[0], self.action_space.n))

            env = gym.make('CartPole-v1')
            q = QFunction(env, Base)
            ob = env.reset()

            outs = q(ob[None])
            assert np.allclose(outs.action, outs.max_a)
            assert np.allclose(outs.value, outs.max_q)
            assert outs.action.shape == (1,)

            outs = q(ob[None], torch.from_numpy(np.array([0])))
            assert np.allclose(outs.action, 0)
            assert np.allclose(outs.value, outs.qvals[:,0])
            assert outs.action.shape == (1,)

            outs = q(ob[None], torch.from_numpy(np.array([[0]])))
            assert np.allclose(outs.action, 0)
            assert np.allclose(outs.value, outs.qvals[:,0])
            assert outs.action.shape == (1,)

        def test_continuous(self):
            class Base(ContinuousQFunctionBase):
                def forward(self, ob, ac):
                    return torch.from_numpy(np.random.rand(ob.shape[0], 1))

            env = gym.make('CartPole-v1')
            q = QFunction(env, Base)
            ob = env.reset()

            ac = torch.from_numpy(np.array([[1]]))
            outs = q(ob[None], ac)
            assert outs.max_a == None
            assert outs.max_q == None
            assert outs.action.shape == (1,1)



    unittest.main()
