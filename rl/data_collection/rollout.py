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
        'obs', 'action', 'reward', 'done', and 'vpred'
    Any amount of additional data can be provided.

    'reward', 'done', and 'vpred' are assumed to be a single torch tensor.
    All other data may be arbitrarily nested torch tensors.

    Once all rollout data has been stored, it can be batched and iterated over
    by calling the 'sampler(batch_size)' method.
    """

    def __init__(self, num_processes, num_steps=128, device='cpu'):
        """Init."""
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.required_keys = ['obs', 'action', 'reward', 'done', 'vpred']
        self.keys = None
        self.data = None
        self.device = device
        self.rollout_complete = False

    def init_data(self, step_data):
        """Initialize data storage."""
        for k in self.required_keys:
            if k not in step_data:
                raise ValueError(f"Key {k} must be provided in step_data.")
        self.keys = set(step_data.keys())
        self.data = {}
        if step_data['reward'].shape != step_data['vpred'].shape:
            raise ValueError('reward and vpred must have the same shape!')
        if step_data['reward'].shape != step_data['done'].shape:
            raise ValueError('reward and done must have the same shape!')

        def _make_storage(arr, recurrent_key=False):
            shape = [self.num_steps] + list(arr.shape)
            return torch.zeros(size=shape, dtype=arr.dtype, device=self.device)

        for k in self.keys:
            self.data[k] = nest.map_structure(_make_storage, step_data[k])
        self.data['vtarg'] = _make_storage(step_data['vpred'])
        self.data['atarg'] = _make_storage(step_data['vpred'])
        self.data['return'] = _make_storage(step_data['vpred'])
        self.data['q_mc'] = _make_storage(step_data['vpred'])
        self.nenv = step_data['reward'].shape[0]
        self.reset()

    def reset(self):
        self.step = 0
        self.sequence_lengths = torch.ones(self.num_processes)
        self.rollout_complete = False

    def extend_storage(self):
        # automatically grow storage to accommodate episode length.
        def _extend(arr):
            shape = [2 * self.num_steps] + list(arr.shape[1:])
            data = torch.zeros(size=shape, dtype=arr.dtype, device=self.device)
            data[:self.num_steps].copy_(arr)
            return data

        self.data = nest.map_structure(_extend, self.data)
        self.num_steps *= 2

    def insert(self, step_data):
        """Insert new data into storage.

        Transfers to the correct device if needed.
        """
        if self.data is None:
            self.init_data(step_data)

        if self.rollout_complete:
            raise ValueError("Tried to insert data when the rollout is "
                             " complete. Call rollout.reset() to reset.")

        if self.step >= self.num_steps:
            self.extend_storage()

        if set(step_data.keys()) != self.keys:
            raise ValueError("The same data must be provided at every step.")

        def _copy_data(item):
            storage, step_data = item
            if step_data.device != self.device:
                storage[self.step].copy_(step_data.to(self.device))
            else:
                storage[self.step].copy_(step_data)

        def _check_shape(data, key):
            if data.shape[0] != self.num_processes:
                raise ValueError(f"data '{key}' is expected to have its "
                                 f"0th dimension equal to the number "
                                 f"of processes: {self.num_processes}")

        for k in self.keys:
            nest.map_structure(partial(_check_shape, key=k), step_data[k])
            nest.map_structure(_copy_data, nest.zip_structure(self.data[k],
                                                              step_data[k]))

        if self.step == 0:
            self.data['return'].fill_(0.)
            self.data['q_mc'].fill_(0.)
            done = torch.zeros_like(self.data['done'][0])
        else:
            done = self.data['done'][self.step - 1]
        r = torch.logical_not(done) * step_data['reward'].to(self.device)
        self.data['return'] += r

        self.sequence_lengths += torch.logical_not(step_data['done'].cpu())
        self.step = self.step + 1
        self.rollout_complete = bool(torch.all(step_data['done']))

    def compute_targets(self, gamma, lambda_=1.0, norm_advantages=False):
        """Compute advantage targets."""
        if not self.rollout_complete:
            raise ValueError("Rollout should be complete before computing "
                             "targets.")
        max_step = int(torch.max(self.sequence_lengths))
        self.data['vtarg'][max_step-1] = self.data['reward'][max_step-1]
        gae = self.data['reward'][max_step-1] - self.data['vpred'][max_step-1]
        self.data['q_mc'][-1] = self.data['reward'][-1]
        for step in reversed(range(max_step - 1)):
            mask = torch.logical_not(self.data['done'][step])
            if step == 0:
                mask_prev = torch.ones_like(mask)
            else:
                mask_prev = torch.logical_not(self.data['done'][step - 1])
            delta = (self.data['reward'][step]
                     + (gamma * mask * self.data['vpred'][step + 1])
                     - self.data['vpred'][step])
            gae = mask_prev * (delta + gamma * lambda_ * gae)
            self.data['atarg'][step].copy_(gae)
            self.data['q_mc'][step] = (self.data['reward'][step] + mask * gamma
                                       * self.data['q_mc'][step + 1])
        self.data['vtarg'] = self.data['atarg'] + self.data['vpred']
        if norm_advantages:
            self.data['atarg'] = (self.data['atarg']
                                  - self.data['atarg'].mean()) / (
                                        self.data['atarg'].std() + 1e-5)

    def get_rollout(self, inds=None):
        if inds is None:
            inds = range(self.num_processes)

        def _pack(x):
            return torch.nn.utils.rnn.pack_padded_sequence(
                                x[:, inds], self.sequence_lengths[inds],
                                enforce_sorted=False)
        return nest.map_structure(_pack, self.data)

    def rollout_length(self):
        return int(sum(self.sequence_lengths))

    def _feed_forward_generator(self, batch_size):
        if not self.rollout_complete:
            raise ValueError(f"Finish rollout before batching data.")

        rollout = nest.map_structure(lambda x: x.data, self.get_rollout())
        n = len(rollout['reward'])
        if batch_size > n:
            raise ValueError(f"Batch size ({batch_size}) is bigger than the "
                             f"number of samples ({n}).")
        sampler = BatchSampler(SubsetRandomSampler(range(n)), batch_size,
                               drop_last=False)

        def _batch_data(data, indices):
            return data[indices]

        for indices in sampler:
            batch = {}
            for k, v in rollout.items():
                f = partial(_batch_data, indices=indices)
                batch[k] = nest.map_structure(f, v)
            yield batch

    def _recurrent_generator(self, batch_size):
        """Leave obs as a packed sequence and extract data from other keys."""

        if not self.rollout_complete:
            raise ValueError(f"Finish rollout before batching data.")

        n = self.num_processes
        if batch_size > n:
            raise ValueError(f"Batch size ({batch_size}) is bigger than the "
                             f"number of samples ({n}).")
        sampler = BatchSampler(SubsetRandomSampler(range(n)), batch_size,
                               drop_last=False)

        for indices in sampler:
            batch = self.get_rollout(indices)
            for k, v in batch.items():
                if k != 'obs':
                    batch[k] = nest.map_structure(lambda x: x.data, v)
            yield batch

    def sampler(self, batch_size, recurrent=False):
        """Iterate over rollout data."""
        if recurrent:
            return self._recurrent_generator(batch_size)
        else:
            return self._feed_forward_generator(batch_size)


if __name__ == '__main__':
    import unittest
    import numpy as np

    class TestRollout(unittest.TestCase):
        """Test."""

        def test(self):
            """Test feeed forward generator."""
            def _gen_data(np, x, dones):
                data = {}
                data['obs'] = x*torch.ones(size=(np, 1, 84, 84))
                data['action'] = torch.zeros(size=(np, 1))
                data['reward'] = torch.ones(size=(np,))
                data['done'] = torch.Tensor(dones).bool()
                data['vpred'] = x*torch.ones(size=(np,))
                data['logp'] = torch.zeros(size=(np,))
                return data
            r = RolloutStorage(3, 10)
            for i in range(10):
                r.insert(_gen_data(3, i, [i >= 7, i >= 9, i >= 7]))
                if i < 9:
                    try:
                        r.compute_targets(gamma=0.99, lambda_=1.0,
                                          norm_advantages=True)
                        assert False
                    except Exception:
                        pass
            r.compute_targets(gamma=0.99, lambda_=1.0, norm_advantages=True)
            assert (np.allclose(r.data['atarg'].mean(), 0., atol=1e-6)
                    and np.allclose(r.data['atarg'].std(), 1., atol=1e-6))

            for batch in r.sampler(2):
                assert batch['obs'].shape == (2, 1, 84, 84)
                assert batch['atarg'].shape == (2,)
                assert batch['vtarg'].shape == (2,)
                assert batch['done'].shape == (2,)
                assert batch['reward'].shape == (2,)
                assert batch['return'].shape == (2,)
                assert batch['q_mc'].shape == (2,)
                print(batch['return'])
                print(batch['q_mc'])

            for batch in r.sampler(2, recurrent=True):
                n = batch['obs'].data.shape[0]
                print(batch['done'])
                assert batch['atarg'].shape == (n,)
                assert batch['vtarg'].shape == (n,)
                assert batch['done'].shape == (n,)
                assert batch['reward'].shape == (n,)
                assert batch['return'].shape == (n,)
                assert batch['q_mc'].shape == (n,)
                print(batch['return'])
                print(batch['q_mc'])

    unittest.main()
