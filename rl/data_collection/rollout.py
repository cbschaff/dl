"""Data Storage for Rollouts.

Loosely based on
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/storage.py
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from dl import nest
from functools import partial


class RolloutStorage(object):
    """Rollout Storage.

    This class stores data from rollouts with an environment.

    Data is provided by passing a dictionary to the 'insert(data)' method.
    The data dictionary must have the keys:
        'obs', 'action', 'reward', 'mask', and 'vpred'
    If the recurrent flag is set to True, the 'state' key must also exist.
    Any amount of additional data can be provided.

    The data with 'obs' and 'state' keys may be arbitrarily nested torch
    tensors. All other data is assumed to be a single torch tensor.

    Once all rollout data has been stored, it can be batched and iterated over
    by calling the 'sampler(batch_size)' method.
    """

    def __init__(self, num_steps, num_processes, device='cpu', recurrent=False):
        """Init."""
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.recurrent = recurrent
        self.required_keys = ['obs', 'action', 'reward', 'mask', 'vpred']
        if recurrent:
            self.required_keys.append('state')
        self.keys = None
        self.data = None
        self.device = device
        self.step = 0

    def init_data(self, step_data):
        """Initialize data storage."""
        for k in self.required_keys:
            if k not in step_data:
                raise ValueError(f"Key {k} must be provided in step_data.")
        self.keys = set(step_data.keys())
        self.data = {}
        if step_data['reward'].shape != step_data['vpred'].shape:
            raise ValueError('reward and vpred must have the same shape!')
        if step_data['reward'].shape != step_data['mask'].shape:
            raise ValueError('reward and mask must have the same shape!')

        def _make_storage(arr, recurrent_key=False):
            if recurrent_key:
                shape = [1] + list(arr.shape)
            else:
                shape = [self.num_steps] + list(arr.shape)
            return torch.zeros(size=shape, dtype=arr.dtype, device=self.device)

        for k in self.keys:
            if k == 'obs':
                self.data['obs'] = nest.map_structure(_make_storage,
                                                      step_data['obs'])
            elif k == 'state':
                if not self.recurrent:
                    raise ValueError("The 'state' key is reserved for storing "
                                     "the recurrent state of a model. To store "
                                     "recurrent states, set recurrent=True "
                                     "when constructing the RolloutStorage.")
                f = partial(_make_storage, recurrent_key=True)
                self.data['state'] = nest.map_structure(f, step_data['state'])
            else:
                self.data[k] = _make_storage(step_data[k])
        self.data['vtarg'] = _make_storage(step_data['vpred'])
        self.data['atarg'] = _make_storage(step_data['vpred'])

    def insert(self, step_data):
        """Insert new data into storage.

        Transfers to the correct device if needed.
        """
        if self.data is None:
            self.init_data(step_data)

        if set(step_data.keys()) != self.keys:
            raise ValueError("The same data must be provided at every step.")

        def _copy_data(item):
            storage, step_data = item
            if step_data.device != self.device:
                storage[self.step].copy_(step_data.to(self.device))
            else:
                storage[self.step].copy_(step_data)

        def _check_shape(data, key):
            batch_dim = -2 if key == 'state' else 0
            if data.shape[batch_dim] != self.num_processes:
                raise ValueError(f"data '{key}' is expected to have its "
                                 f"{batch_dim} dimension equal to the number "
                                 f"of processes: {self.num_processes}")

        for k in self.keys:
            if k == 'state' and self.step > 0:
                continue
            if k in ['state', 'obs']:
                nest.map_structure(partial(_check_shape, key=k), step_data[k])
                nest.map_structure(_copy_data, nest.zip_structure(self.data[k],
                                                                  step_data[k]))
            else:
                _check_shape(step_data[k], key=k)
                _copy_data((self.data[k], step_data[k]))

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

    def _feed_forward_generator(self, batch_size):
        if self.step != 0:
            raise ValueError(f"Insert exactly {self.num_steps} transitions "
                             f"before calling a generator. Only {self.step} "
                             "insertions were made.")
        N = self.num_processes * self.num_steps
        if batch_size > N:
            raise ValueError(f"Batch size ({batch_size}) is bigger than the "
                             f"number of samples ({N}).")
        sampler = BatchSampler(SubsetRandomSampler(range(N)), batch_size,
                               drop_last=False)

        def _batch_data(data, indices):
            return data.view(-1, *data.shape[2:])[indices]

        for indices in sampler:
            batch = {}
            for k, v in self.data.items():
                if k == 'obs':
                    f = partial(_batch_data, indices=indices)
                    batch['obs'] = nest.map_structure(f, v)
                else:
                    batch[k] = _batch_data(v, indices=indices)
            yield batch

    def _recurrent_generator(self, batch_size):
        if self.step != 0:
            raise ValueError(f"Insert exactly {self.num_steps} transitions "
                             f"before calling a generator. Only {self.step} "
                             "insertions were made.")
        N = self.num_processes
        if batch_size > N:
            raise ValueError(f"Batch size ({batch_size}) is bigger than the "
                             f"number of samples ({N}).")
        sampler = BatchSampler(SubsetRandomSampler(range(N)), batch_size,
                               drop_last=False)

        def _batch_data(data, indices, recurrent):
            if recurrent:
                return data[..., indices, :][0]
            else:
                return data[:, indices].view(-1, *data.shape[2:])

        for indices in sampler:
            batch = {}
            for k, v in self.data.items():
                if k == 'state':
                    f = partial(_batch_data, indices=indices, recurrent=True)
                    batch['state'] = nest.map_structure(f, v)
                elif k == 'obs':
                    f = partial(_batch_data, indices=indices, recurrent=False)
                    batch['obs'] = nest.map_structure(f, v)
                else:
                    batch[k] = _batch_data(v, indices=indices, recurrent=False)
            yield batch

    def sampler(self, batch_size):
        """Iterate over rollout data."""
        if self.recurrent:
            return self._recurrent_generator(batch_size)
        else:
            return self._feed_forward_generator(batch_size)


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
            r = RolloutStorage(10, 2, recurrent=False)
            for i in range(10):
                r.insert(_gen_data(2, i))
                if i < 9:
                    try:
                        r.sampler(6)
                        assert False
                    except Exception:
                        pass
            r.compute_targets(torch.ones(size=(2, 1)), torch.ones(size=(2, 1)),
                              gamma=0.99, use_gae=True, lambda_=1.0,
                              norm_advantages=True)
            assert (np.allclose(r.data['atarg'].mean(), 0., atol=1e-6)
                    and np.allclose(r.data['atarg'].std(), 1., atol=1e-6))
            for batch in r.sampler(6):
                assert (batch['obs'].shape == (6, 1, 84, 84)
                        or batch['obs'].shape == (2, 1, 84, 84))

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
            r = RolloutStorage(10, 4, recurrent=True)
            for i in range(10):
                r.insert(_gen_data(4, i))
                if i < 9:
                    try:
                        r.sampler(2)
                        assert False
                    except Exception:
                        pass
            r.compute_targets(torch.ones(size=(4, 1)), torch.ones(size=(4, 1)),
                              gamma=0.99, use_gae=True, lambda_=1.0)
            for batch in r.sampler(2):
                assert batch['obs'].shape == (20, 1, 84, 84)
                assert batch['atarg'].shape == (20, 1)
                assert batch['vtarg'].shape == (20, 1)
                assert batch['mask'].shape == (20, 1)
                assert batch['reward'].shape == (20, 1)
                assert batch['state'].shape == (2, 5)

        def test_nested_feed_forward(self):
            """Test feeed forward generator."""
            def _gen_data(np, x):
                data = {}
                ob = x*torch.ones(size=(np, 1, 84, 84))
                data['obs'] = [ob, {'ob1': ob, 'ob2': ob}]
                data['action'] = torch.zeros(size=(np, 1))
                data['reward'] = torch.zeros(size=(np, 1))
                data['mask'] = torch.ones(size=(np, 1))
                data['vpred'] = x*torch.ones(size=(np, 1))
                data['logp'] = torch.zeros(size=(np, 1))
                return data
            r = RolloutStorage(10, 2, recurrent=False)
            for i in range(10):
                r.insert(_gen_data(2, i))
                if i < 9:
                    try:
                        r.sampler(6)
                        assert False
                    except Exception:
                        pass
            r.compute_targets(torch.ones(size=(2, 1)), torch.ones(size=(2, 1)),
                              gamma=0.99, use_gae=True, lambda_=1.0,
                              norm_advantages=True)
            assert (np.allclose(r.data['atarg'].mean(), 0., atol=1e-6)
                    and np.allclose(r.data['atarg'].std(), 1., atol=1e-6))
            for batch in r.sampler(6):
                assert (batch['obs'][0].shape == (6, 1, 84, 84)
                        or batch['obs'][0].shape == (2, 1, 84, 84))
                assert (batch['obs'][1]['ob1'].shape == (6, 1, 84, 84)
                        or batch['obs'][1]['ob1'].shape == (2, 1, 84, 84))
                assert (batch['obs'][1]['ob2'].shape == (6, 1, 84, 84)
                        or batch['obs'][1]['ob2'].shape == (2, 1, 84, 84))

        def test_nested_recurrent(self):
            """Test recurreent generator."""
            def _gen_data(np, x):
                data = {}
                ob = x*torch.ones(size=(np, 1, 84, 84))
                data['obs'] = [ob, {'ob1': ob, 'ob2': ob}]
                data['action'] = torch.zeros(size=(np, 1))
                data['reward'] = torch.zeros(size=(np, 1))
                data['mask'] = torch.ones(size=(np, 1))
                data['vpred'] = x*torch.ones(size=(np, 1))
                data['logp'] = torch.zeros(size=(np, 1))
                state = torch.zeros(size=(np, 5))
                data['state'] = {'1': state, '2': [state, state]}
                return data
            r = RolloutStorage(10, 4, recurrent=True)
            for i in range(10):
                r.insert(_gen_data(4, i))
                if i < 9:
                    try:
                        r.sampler(2)
                        assert False
                    except Exception:
                        pass
            r.compute_targets(torch.ones(size=(4, 1)), torch.ones(size=(4, 1)),
                              gamma=0.99, use_gae=True, lambda_=1.0)
            for batch in r.sampler(2):
                assert batch['obs'][0].shape == (20, 1, 84, 84)
                assert batch['obs'][1]['ob1'].shape == (20, 1, 84, 84)
                assert batch['obs'][1]['ob2'].shape == (20, 1, 84, 84)
                assert batch['atarg'].shape == (20, 1)
                assert batch['vtarg'].shape == (20, 1)
                assert batch['mask'].shape == (20, 1)
                assert batch['reward'].shape == (20, 1)
                assert batch['state']['1'].shape == (2, 5)
                assert batch['state']['2'][0].shape == (2, 5)
                assert batch['state']['2'][1].shape == (2, 5)

    unittest.main()
