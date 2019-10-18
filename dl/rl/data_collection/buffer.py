"""Replay buffer.

This file is apdated from
https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
Edits were made to make the buffer more flexible in the data it could store.
"""
import numpy as np
import random


def sample_n_unique(sampling_f, n):
    """Sample n unique outputs from sampling_f.

    Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class ReplayBuffer(object):
    """Replay Buffer."""

    def __init__(self, size, frame_history_len):
        """Memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frame_t and frame_(t+1) in the same buffer.
        For the typical use case in Atari Deep RL buffer with 1M frames the
        total memory footprint of this buffer is
        10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Warning! Observations are concatenated along the first dimension.
        For images, this means that the data format should be (C,H,W).
        Parameters

        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.

        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.data = {}
        self.required_keys = ['action', 'reward', 'done']

    def _init_obs_data(self, frame):
        dtype = np.float32 if frame.dtype == np.float64 else frame.dtype
        self.obs = np.empty([self.size] + list(frame.shape), dtype=dtype)

    def _init_replay_data(self, step_data):
        for k in self.required_keys:
            if k not in step_data:
                raise ValueError("action, reward, and done must be keys in the"
                                 "dict passed to buffer.store_effect.")
        for k in step_data.keys():
            data_k = np.asarray(step_data[k])
            dtype = data_k.dtype if data_k.dtype == np.float64 else np.float32
            self.data[k] = np.empty([self.size] + list(data_k.shape),
                                    dtype=dtype)

    def can_sample(self, batch_size):
        """Check if a batch_size can be sampled.

        Returns true if `batch_size` different transitions can be sampled
        from the buffer.
        """
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        batch = {}
        obs_batch = np.concatenate(
            [self._encode_observation(idx)[np.newaxis, :] for idx in idxes], 0)
        for k in self.data.keys():
            batch[k] = self.data[k][idxes]
        next_obs_batch = np.concatenate(
            [self._encode_observation(idx + 1)[np.newaxis, :] for idx in idxes],
            0)
        batch['obs'] = obs_batch
        batch['next_obs'] = next_obs_batch
        return batch

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        batched data: dict
            a dictionary containing batched observations, next_observations,
            action, reward, done, and other data stored in the replay buffer.

        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(
            lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`

        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx = idx + 1  # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.data['done'][idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [
                np.zeros_like(self.obs[0]) for _ in range(missing_context)
            ]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0)
        else:
            # this optimization has potential to saves about 30% compute time
            s = self.obs.shape[2:]
            return self.obs[start_idx:end_idx].reshape(-1, *s)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index.

        Overwrites old frames if necessary.
        Parameters
        ----------
        frame: np.array
            Array of shape (img_c, img_h, img_w) and dtype np.uint8
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect`
            later.

        """
        if self.obs is None:
            self._init_obs_data(frame)

        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, step_data):
        """Store effects of action taken after obeserving frame stored at idx.

        The reason `store_frame` and `store_effect` is broken
        up into two functions is so that one can call
        `encode_recent_observation` in between.
        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame
            (returned by `store_frame`).
        data: dict
            The data to store in the buffer.
        """
        if self.data == {}:
            self._init_replay_data(step_data)
        if set(self.data.keys()) != set(step_data.keys()):
            print(self.data.keys(), step_data.keys())
            raise ValueError("The data passed to ReplayBuffer must the same"
                             " at all time steps.")
        for k, v in step_data.items():
            self.data[k][idx] = v

    def env_reset(self):
        """Update buffer based on early environment resest.

        Allow environment resets for the most recent transition after it has
        been stored. This is useful when loading a saved replay buffer.
        """
        if self.num_in_buffer > 0:
            self.data['done'][(self.next_idx-1) % self.size] = True

    def state_dict(self):
        """State dict."""
        return {
            'obs': self.obs,
            'data': self.data,
            'num_in_buffer': self.num_in_buffer,
            'next_idx': self.next_idx,
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.obs = state_dict['obs']
        self.data = state_dict['data'].item()
        self.num_in_buffer = state_dict['num_in_buffer']
        self.next_idx = state_dict['next_idx']


"""
Unit Tests
"""


if __name__ == '__main__':
    import unittest
    from dl.rl.envs import make_atari_env as atari_env

    class TestBuffer(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            buffer = ReplayBuffer(10, 4)
            env = atari_env('Pong')
            init_obs = env.reset()
            idx = buffer.store_frame(init_obs)
            assert np.allclose(buffer.encode_recent_observation()[:-3], 0)
            for i in range(10):
                ac = env.action_space.sample()
                obs, r, done, _ = env.step(ac)
                data = {'action': ac, 'reward': r, 'done': done, 'key1': 0}
                buffer.store_effect(idx, data)
                idx = buffer.store_frame(obs)

            # Check sample shapes
            s = buffer.sample(2)
            assert s['obs'].shape == (2, 4, 84, 84)
            assert s['next_obs'].shape == (2, 4, 84, 84)
            assert s['key1'].shape == (2,)
            s = buffer._encode_sample([4, 5])
            # Check observation stacking
            assert np.allclose(s['obs'][0][3], s['next_obs'][0][2])
            assert np.allclose(s['obs'][0][2], s['next_obs'][0][1])
            assert np.allclose(s['obs'][0][1], s['next_obs'][0][0])

            # Check sequential samples
            assert np.allclose(s['obs'][0][3], s['obs'][1][2])

            # Check for wrap around when buffer is full
            s = buffer._encode_sample([0])
            assert not np.allclose(s['obs'][0][:-3], 0)

            # Check env reset
            buffer.env_reset()
            assert buffer.data['done'][buffer.next_idx - 1 % buffer.size]

            # Check saving and loading
            state = buffer.state_dict()
            buffer2 = ReplayBuffer(10, 4)
            buffer2.load_state_dict(state)

            s1 = buffer._encode_sample([1, 3, 5])
            s2 = buffer2._encode_sample([1, 3, 5])
            for k in s1:
                assert np.allclose(s1[k], s2[k])

            for i in range(10):
                ac = env.action_space.sample()
                obs, r, done, _ = env.step(ac)
                data = {'action': ac, 'reward': r, 'done': done, 'key1': 0}
                buffer.store_effect(idx, data)
                buffer2.store_effect(idx, data)
                idx = buffer.store_frame(obs)
                idx2 = buffer2.store_frame(obs)
                assert idx == idx2

            s1 = buffer._encode_sample([1, 3, 5])
            s2 = buffer2._encode_sample([1, 3, 5])
            for k in s1:
                assert np.allclose(s1[k], s2[k])

    unittest.main()
