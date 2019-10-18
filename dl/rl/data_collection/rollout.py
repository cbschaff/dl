"""Data Storage for Rollouts.

Loosely based on
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/storage.py
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    """Rollout Storage."""

    def __init__(self, num_steps, num_processes, device='cpu', other_keys=[],
                 recurrent_state_keys=[]):
        """Init."""
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.recurrent_state_keys = recurrent_state_keys
        self.keys = (['obs', 'action', 'reward', 'mask', 'vpred']
                     + list(other_keys) + list(recurrent_state_keys))
        self.data = None
        self.device = device
        self.step = 0

    def init_data(self, step_data):
        """Initialize data storage."""
        self.data = {}
        assert step_data['reward'].shape == step_data['vpred'].shape
        assert step_data['reward'].shape == step_data['mask'].shape
        for k in self.keys:
            if k in self.recurrent_state_keys:
                shape = [1] + list(step_data[k].shape)
            else:
                shape = [self.num_steps] + list(step_data[k].shape)
            self.data[k] = torch.zeros(size=shape, dtype=step_data[k].dtype,
                                       device=self.device)
        shape = [self.num_steps] + list(step_data['vpred'].shape)
        self.data['vtarg'] = torch.zeros(size=shape,
                                         dtype=step_data['vpred'].dtype,
                                         device=self.device)
        self.data['atarg'] = torch.zeros_like(self.data['vtarg'])

    def insert(self, step_data):
        """Insert new data into storage.

        Transfers to the correct device if needed.
        """
        if self.data is None:
            self.init_data(step_data)
        for k in self.keys:
            batch_dim = -2 if k in self.recurrent_state_keys else 0
            assert step_data[k].shape[batch_dim] == self.num_processes, \
                ("inserted data is expected to have its first dimension equal "
                 f"to the numper of processes: {self.num_processes}")
            if k in self.recurrent_state_keys and self.step > 0:
                continue
            if step_data[k].device != self.device:
                self.data[k][self.step].copy_(step_data[k].to(self.device))
            else:
                self.data[k][self.step].copy_(step_data[k])

        self.step = (self.step + 1) % self.num_steps

    def compute_targets(self, next_value, next_mask, gamma, use_gae=True,
                        lambda_=1.0, norm_advantages=False):
        """Compute advantage targets."""
        if use_gae:
            gae = (self.data['reward'][-1] + gamma * next_mask * next_value
                   - self.data['vpred'][-1])
            self.data['vtarg'][-1] = gae + self.data['vpred'][-1]
            for step in reversed(range(self.num_steps - 1)):
                delta = (self.data['reward'][step]
                         + (gamma * self.data['mask'][step + 1]
                            * self.data['vpred'][step + 1])
                         - self.data['vpred'][step])
                gae = delta + (gamma * lambda_ * self.data['mask'][step + 1]
                               * gae)
                self.data['vtarg'][step] = gae + self.data['vpred'][step]
        else:
            self.data['vtarg'][-1] = self.data['reward'][-1] + (gamma
                                                                * next_mask
                                                                * next_value)
            for step in reversed(range(self.num_steps - 1)):
                self.data['vtarg'][step] = self.data['reward'][step] + (
                    gamma
                    * self.data['mask'][step + 1]
                    * self.data['vtarg'][step + 1]
                )
        self.data['atarg'] = self.data['vtarg'] - self.data['vpred']
        if norm_advantages:
            self.data['atarg'] = (self.data['atarg']
                                  - self.data['atarg'].mean()) / (
                                        self.data['atarg'].std() + 1e-5)

    def feed_forward_generator(self, batch_size):
        """Iterate over rollout data."""
        assert self.step == 0, (
            f"Insert exactly {self.num_steps} transitions before calling a "
            f"generator. Only {self.step} insertions were made.")
        assert len(self.recurrent_state_keys) == 0, (
            "Call self.recurrent_generator when using a recurrent model.")
        N = self.num_processes * self.num_steps
        assert batch_size <= N
        sampler = BatchSampler(SubsetRandomSampler(range(N)), batch_size,
                               drop_last=False)
        for indices in sampler:
            batch = {}
            for k, v in self.data.items():
                batch[k] = v.view(-1, *v.shape[2:])[indices]
            yield batch

    def recurrent_generator(self, batch_size):
        """Iterate over rollout data."""
        assert self.step == 0, (
            f"Insert exactly {self.num_steps} transitions before calling a "
            f"generator. Only {self.step} insertions were made.")
        N = self.num_processes
        assert batch_size <= N
        sampler = BatchSampler(SubsetRandomSampler(range(N)), batch_size,
                               drop_last=False)
        for indices in sampler:
            batch = {}
            for k, v in self.data.items():
                if k in self.recurrent_state_keys:
                    batch[k] = v[..., indices, :][0]
                else:
                    batch[k] = v[:, indices].view(-1, *v.shape[2:])
            yield batch


if __name__ == '__main__':
    import unittest
    import numpy as np

    class TestRollout(unittest.TestCase):
        """Test."""

        def test_feed_forward(self):
            """Test feeed forward generator."""
            def _gen_data(np, x):
                data = {}
                data['obs'] = x*torch.ones(size=(np, 1, 84, 84))
                data['action'] = torch.zeros(size=(np, 1))
                data['reward'] = torch.zeros(size=(np, 1))
                data['mask'] = torch.ones(size=(np, 1))
                data['vpred'] = x*torch.ones(size=(np, 1))
                data['logp'] = torch.zeros(size=(np, 1))
                return data
            r = RolloutStorage(10, 2, other_keys=['logp'])
            for i in range(10):
                r.insert(_gen_data(2, i))
                if i < 9:
                    try:
                        r.feed_forward_generator(6)
                        assert False
                    except Exception:
                        pass
            r.compute_targets(torch.ones(size=(2, 1)), torch.ones(size=(2, 1)),
                              gamma=0.99, use_gae=True, lambda_=1.0,
                              norm_advantages=True)
            assert (np.allclose(r.data['atarg'].mean(), 0., atol=1e-6)
                    and np.allclose(r.data['atarg'].std(), 1., atol=1e-6))
            for batch in r.feed_forward_generator(6):
                assert (batch['obs'].shape == (6, 1, 84, 84)
                        or batch['obs'].shape == (2, 1, 84, 84))
            try:
                r.recurrent_generator(1)
                assert False
            except Exception:
                pass

        def test_recurrent(self):
            """Test recurreent generator."""
            def _gen_data(np, x):
                data = {}
                data['obs'] = x*torch.ones(size=(np, 1, 84, 84))
                data['action'] = torch.zeros(size=(np, 1))
                data['reward'] = torch.zeros(size=(np, 1))
                data['mask'] = torch.ones(size=(np, 1))
                data['vpred'] = x*torch.ones(size=(np, 1))
                data['logp'] = torch.zeros(size=(np, 1))
                data['state'] = torch.zeros(size=(np, 5))
                return data
            r = RolloutStorage(10, 4, other_keys=['logp'],
                               recurrent_state_keys=['state'])
            for i in range(10):
                r.insert(_gen_data(4, i))
                if i < 9:
                    try:
                        r.recurrent_generator(2)
                        assert False
                    except Exception:
                        pass
            r.compute_targets(torch.ones(size=(4, 1)), torch.ones(size=(4, 1)),
                              gamma=0.99, use_gae=True, lambda_=1.0)
            for batch in r.recurrent_generator(2):
                assert batch['obs'].shape == (20, 1, 84, 84)
                assert batch['atarg'].shape == (20, 1)
                assert batch['vtarg'].shape == (20, 1)
                assert batch['mask'].shape == (20, 1)
                assert batch['reward'].shape == (20, 1)
                assert batch['state'].shape == (2, 5)

    unittest.main()
