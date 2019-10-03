"""Extends RLTrainer with utils for a ReplayBuffer."""
from dl.rl.trainers import RLTrainer
from dl.rl.util import ReplayBuffer
import gin
import os
import torch
import numpy as np


@gin.configurable(blacklist=['logdir'])
class ReplayBufferTrainer(RLTrainer):
    """Extends RLTrainer with functionality for replay buffers.

    The resposibilities of this class are:
    - Replay Buffer creation, saving, and loading.
    - Adding data to the replay buffer.
    - Stepping the environment.
    """

    def __init__(self,
                 logdir,
                 env_fn,
                 buffer_size=10000,
                 frame_stack=1,
                 learning_starts=1000,
                 update_period=1,
                 **kwargs):
        """Init."""
        super().__init__(logdir, env_fn, **kwargs)
        assert self.nenv == 1
        self.buffer = ReplayBuffer(buffer_size, frame_stack)
        self.frame_stack = frame_stack
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.update_period = update_period
        self._reset()

    def _reset(self):
        self.buffer.env_reset()
        self._ob = self.env.reset()

    def _save(self, state_dict):
        # save buffer seperately and only once (because it can be huge)
        np.savez(os.path.join(self.ckptr.ckptdir, 'buffer.npz'),
                 **self.buffer.state_dict())
        super()._save(state_dict)

    def _load(self, state_dict):
        self.buffer.load_state_dict(np.load(os.path.join(self.ckptr.ckptdir,
                                                         'buffer.npz')))
        self._reset()
        super()._load(state_dict)

    def env_step_and_store_transition(self):
        """Step env and store transition in replay buffer."""
        idx = self.buffer.store_frame(self._ob[0])  # remove batch dimension
        ob = self.buffer.encode_recent_observation()
        with torch.no_grad():
            ob = torch.from_numpy(ob).to(self.device)
            ac = self.act(ob[None]).cpu().numpy()
        self._ob, r, done, _ = self.env.step(ac)
        self.store_effect(idx, ac[0], r, done)
        if done:
            self._ob = self.env.reset()
        self.t += 1

    def step_until_update(self):
        """Step env untiil update."""
        for _ in range(self.update_period):
            self.env_step_and_store_transition()
        while self.buffer.num_in_buffer < min(self.learning_starts,
                                              self.buffer.size):
            self.env_step_and_store_transition()

    def act(self, ob):
        """Compute decision from model.

        Override in subclasses.
        Returns:
            out: action to take

        """
        raise NotImplementedError

    def store_effect(self, idx, ac, r, done):
        """Store the effect of an action in the replay buffer."""
        self.buffer.store_effect(idx, ac, r, done)


if __name__ == '__main__':
    import unittest
    import shutil
    from dl.rl.modules import QFunction, DiscreteQFunctionBase
    from dl.rl.envs import make_env
    from dl.modules import FeedForwardNet

    class FeedForwardBase(DiscreteQFunctionBase):
        """Feed forward network."""

        def build(self):
            """Build network."""
            inshape = self.observation_space.shape[0]
            self.net = FeedForwardNet(inshape, [32, 32, self.action_space.n])

        def forward(self, ob):
            """Forward."""
            return self.net(ob.float())

    class T(ReplayBufferTrainer):
        """Test trainer."""

        def __init__(self, *args, base=None, **kwargs):
            """Init."""
            super().__init__(*args, **kwargs)
            self.qf = QFunction(self.env, base)

        def act(self, ob):
            """Act."""
            return self.qf(ob).action

        def step(self):
            """Step."""
            self.step_until_update()
            batch = self.buffer.sample(32)
            ob, ac, rew, next_ob, done = [torch.from_numpy(x).to(self.device)
                                          for x in batch]
            self.act(ob)
            assert ac.shape == rew.shape
            assert ac.shape == done.shape
            assert ob.shape == next_ob.shape
            assert len(ob.shape) == 2
            assert len(ac.shape) == 1

        def state_dict(self):
            """State dict."""
            return {}

        def load_state_dict(self, state_dict):
            """Load state dict."""
            pass

    def env(rank):
        """Env function."""
        return make_env('CartPole-v1', rank=rank)

    class TestOffPolicyTrainer(unittest.TestCase):
        """Test case."""

        def test(self):
            """Test."""
            t = T('./test', env, buffer_size=2000, learning_starts=50,
                  frame_stack=1, update_period=2, base=FeedForwardBase,
                  maxt=1000)
            t.train()
            t = T('./test', env, buffer_size=2000, learning_starts=50,
                  frame_stack=1, update_period=2, base=FeedForwardBase,
                  maxt=1000)
            t.train()
            assert t.buffer.num_in_buffer == 1000
            shutil.rmtree('./test')

    unittest.main()
