"""Policy Subclass which unnormalizes actions."""
from dl.rl.modules import Policy
import torch
import gin
from collections import namedtuple


@gin.configurable(whitelist=['base'])
class UnnormActionPolicy(Policy):
    """Unnormalize the action based on action space bounds.

    Requires that action space is an instance of "Box".
    """

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self._outputs_normed = namedtuple('Outputs', [
                                    'action', 'value', 'dist', 'state_out',
                                    'batch_sizes', 'sorted_indices',
                                    'unsorted_indices', 'normed_action'])

    def forward(self, *args, **kwargs):
        """Forward."""
        outs = super().forward(*args, **kwargs)
        if self.base.action_space.__class__.__name__ == 'Box':
            if not hasattr(self, 'low'):
                self._get_bounds_on_device(outs.action)
            if self.low is not None and self.high is not None:
                ac = self.low + 0.5 * (outs.action + 1) * (self.high - self.low)
            else:
                ac = outs.action
            outs = self._outputs_normed(action=ac, value=outs.value,
                                        dist=outs.dist,
                                        state_out=outs.state_out,
                                        normed_action=outs.action,
                                        batch_sizes=outs.batch_sizes,
                                        sorted_indices=outs.sorted_indices,
                                        unsorted_indices=outs.unsorted_indices)
        else:
            raise ValueError("Action space must be an instance of 'Box'")
        return outs

    def _get_bounds_on_device(self, action):
        low = self.base.action_space.low
        high = self.base.action_space.high
        if low is not None and high is not None:
            self.low = torch.from_numpy(low).to(action.device)
            self.high = torch.from_numpy(high).to(action.device)
        else:
            self.low = None
            self.high = None


if __name__ == "__main__":
    import unittest
    import gym
    from dl.modules import DeltaDist
    from dl.rl.modules import PolicyBase

    class TestPolicy(unittest.TestCase):
        """Test Policy module."""

        def test_policy_base(self):
            """Test Policy base."""
            class Base(PolicyBase):
                def forward(self, ob):
                    return DeltaDist(torch.ones([2], dtype=torch.float32))

            env = gym.make('LunarLanderContinuous-v2')
            pi = Policy(Base(env.observation_space, env.action_space))
            pi2 = UnnormActionPolicy(Base(env.observation_space,
                                          env.action_space))
            ob = env.reset()

            outs = pi(ob[None])
            assert outs.value is None
            assert outs.state_out is None

            outs2 = pi2(ob[None])
            assert outs.value is None
            assert outs.state_out is None

            ac_normed = 2 * (outs2.action - pi2.low) / (pi2.high - pi2.low) - 1.
            assert torch.allclose(ac_normed, outs.action)
            assert torch.allclose(outs2.normed_action, outs.action)

    unittest.main()
