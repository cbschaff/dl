"""
    This file is apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
    Minor edits were made to allow for easier subclassing and changing the type of data stored.
"""
import numpy as np
import random

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """This is a memory efficient implementation of the replay buffer.
        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.
        For the typical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes
        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.
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

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.action   = None

    def _init_obs_data(self, frame):
        dtype = np.float32 if frame.dtype == np.float64 else frame.dtype
        self.obs      = np.empty([self.size] + list(frame.shape), dtype=dtype)

    def _init_replay_data(self, action_shape, action_dtype):
        self.action   = np.empty([self.size] + list(action_shape), dtype=action_dtype)
        self.reward   = np.empty([self.size],                      dtype=np.float32)
        self.done     = np.empty([self.size],                      dtype=np.bool)

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[np.newaxis, :] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[np.newaxis, :] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        """Sample `batch_size` different transitions.
        i-th sample transition is the following:
        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
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
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            s = self.obs.shape[2:]
            return self.obs[start_idx:end_idx].reshape(-1, *s)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.
        Parameters
        ----------
        frame: np.array
            Array of shape (img_c, img_h, img_w) and dtype np.uint8
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self._init_obs_data(frame)

        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that one can call `encode_recent_observation`
        in between.
        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        if self.action is None:
            self._init_replay_data(action.shape, action.dtype)
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

    def env_reset(self):
        """
        Allow environment resets for the most recent transition after it has
        been stored. This is useful when loading a saved replay buffer.
        """
        if self.num_in_buffer > 0:
            self.done[(self.next_idx-1) % self.size] = True

    def state_dict(self):
        return {
            'obs': self.obs,
            'action': self.action,
            'reward': self.reward,
            'done': self.done,
            'num_in_buffer': self.num_in_buffer,
            'next_idx': self.next_idx,
        }

    def load_state_dict(self, state_dict):
        self.obs = state_dict['obs']
        self.action = state_dict['action']
        self.reward = state_dict['reward']
        self.done = state_dict['done']
        self.num_in_buffer = state_dict['num_in_buffer']
        self.next_idx = state_dict['next_idx']




"""
Unit Tests
"""

import unittest
import gym, numpy as np

class TestBuffer(unittest.TestCase):
    def test(self):
        buffer = ReplayBuffer(10, 4)
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
        s = buffer.sample(2)
        assert s[0].shape == (2,12,210,160)
        assert s[3].shape == (2,12,210,160)
        s = buffer._encode_sample([4,5])
        # Check observation stacking
        assert np.allclose(s[0][0][-3:],   s[3][0][-6:-3])
        assert np.allclose(s[0][0][-6:-3], s[3][0][-9:-6])
        assert np.allclose(s[0][0][-9:-6], s[3][0][:3])

        # Check sequential samples
        assert np.allclose(s[0][0][-3:],   s[0][1][-6:-3])

        # Check for wrap around when buffer is full
        s = buffer._encode_sample([0])
        assert not np.allclose(s[0][0][:-3], 0)

        #Check env reset
        buffer.env_reset()
        assert buffer.done[buffer.next_idx - 1 % buffer.size] == True

        # Check saving and loading
        state = buffer.state_dict()
        buffer2 = ReplayBuffer(10, 4)
        buffer2.load_state_dict(state)

        s1 = buffer._encode_sample([1,3,5])
        s2 = buffer2._encode_sample([1,3,5])
        for i,x in enumerate(s1):
            assert np.allclose(x, s2[i])


        for i in range(10):
            ac = env.action_space.sample()
            obs, r, done, _ = env.step(ac)
            buffer.store_effect(idx, ac, r, done)
            buffer2.store_effect(idx, ac, r, done)
            idx = buffer.store_frame(obs)
            idx2 = buffer2.store_frame(obs)
            assert idx == idx2

        s1 = buffer._encode_sample([1,3,5])
        s2 = buffer2._encode_sample([1,3,5])
        for i,x in enumerate(s1):
            assert np.allclose(x, s2[i])





if __name__=='__main__':
    unittest.main()
