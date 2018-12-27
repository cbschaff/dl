"""
Modified from OpenAI baselines.
https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""
from dl.util import ReplayBuffer
from dl.util.segment_tree import SumSegmentTree, MinSegmentTree
from dl.util.buffer import sample_n_unique
import random


class PrioritizedReplayBuffer(object):
    """
    Implementation of https://arxiv.org/abs/1511.05952, using the "proportional sampling" algorithm.
    """
    def __init__(self, buffer, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        assert alpha >= 0
        self._alpha = alpha
        self.buffer = buffer

        it_capacity = 1
        while it_capacity < buffer.size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def can_sample(self, batch_size):
        return self.buffer.can_sample(batch_size)

    def _sample_proportional(self):
        mass = random.random() * self._it_sum.sum(0, self.buffer.num_in_buffer - 2)
        return self._it_sum.find_prefixsum_idx(mass)

    def _encode_sample(self, idxes):
        return self.buffer._encode_sample(idxes)

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(self._sample_proportional, batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.buffer.num_in_buffer) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.buffer.num_in_buffer) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self.buffer._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def encode_recent_observation(self):
        return self.buffer.encode_recent_observation()

    def store_frame(self, frame):
        idx = self.buffer.store_frame(frame)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
        return idx

    def store_effect(self, *args, **kwargs):
        return self.buffer.store_effect(*args, **kwargs)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        priorities = np.minimum(priorities, self._max_priority)
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.buffer.num_in_buffer - 1
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)




"""
Unit Tests
"""

import unittest
import gym, numpy as np

class TestPrioritizedBuffer(unittest.TestCase):
    def test(self):
        buffer = ReplayBuffer(10, 4)
        buffer = PrioritizedReplayBuffer(buffer, alpha=0.5)
        env = gym.make('PongNoFrameskip-v4')
        init_obs = env.reset()
        idx = buffer.store_frame(init_obs)
        assert np.allclose(buffer.encode_recent_observation()[:-3], 0)
        for i in range(10):
            ac = env.action_space.sample()
            obs, r, done, _ = env.step(ac)
            buffer.store_effect(idx, ac, r, done)
            idx = buffer.store_frame(obs)

        # Check sample shapes
        s = buffer.sample(2, beta=1.)
        assert len(s) == 7
        assert s[0].shape == (2,12,210,160)
        assert s[3].shape == (2,12,210,160)
        s = buffer._encode_sample([4,5])
        # Check observation stacking
        assert np.allclose(s[0][0][-3:],   s[3][0][-6:-3])
        assert np.allclose(s[0][0][-6:-3], s[3][0][-9:-6])
        assert np.allclose(s[0][0][-9:-6], s[3][0][:3])

        # Check sequential samples
        assert np.allclose(s[0][0][-3:],   s[0][1][-6:-3])

        # check priorities
        buffer.update_priorities([4,5],[0.5,2])
        assert buffer._it_sum[4] == 0.5 ** buffer._alpha
        assert buffer._it_sum[5] == 1.0 ** buffer._alpha
        assert buffer._it_min[4] == 0.5 ** buffer._alpha
        assert buffer._it_min[5] == 1.0 ** buffer._alpha
        assert buffer._max_priority == 1.0


        # Check for wrap around when buffer is full
        s = buffer._encode_sample([0])
        assert not np.allclose(s[0][0][:-3], 0)





if __name__=='__main__':
    unittest.main()
