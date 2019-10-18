"""DQN algorithm.

https://www.nature.com/articles/nature14236
"""
from dl.rl.trainers import RLTrainer
from dl.rl.data_collection import ReplayBufferDataManager, ReplayBuffer
from dl.rl.modules import QFunction
from dl import logger
import gin
import os
import time
import torch
import torch.nn as nn
import numpy as np
from dl.rl.envs import VecFrameStack
from baselines.common.schedules import LinearSchedule


class EpsilonGreedyActor(object):
    """Epsilon Greedy actor."""

    def __init__(self, qf, epsilon_schedule, action_space):
        """Init."""
        self.qf = qf
        self.eps = epsilon_schedule
        self.action_space = action_space
        self.t = 0

    def __call__(self, obs):
        """Epsilon greedy action."""
        if self.eps.value(self.t) > np.random.rand():
            action = torch.from_numpy(
                np.array(self.action_space.sample()))[None]
        else:
            action = self.qf(obs).action
        self.t += 1
        return {'action': action}

    def state_dict(self):
        """State dict."""
        return {'t': self.t}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.t = state_dict['t']


@gin.configurable(blacklist=['logdir'])
class DQN(RLTrainer):
    """DQN algorithm."""

    def __init__(self,
                 logdir,
                 env_fn,
                 qf_fn,
                 optimizer=torch.optim.RMSprop,
                 buffer_size=100000,
                 frame_stack=1,
                 learning_starts=10000,
                 update_period=1,
                 gamma=0.99,
                 huber_loss=True,
                 exploration_timesteps=1000000,
                 final_eps=0.1,
                 target_update_period=10000,
                 batch_size=32,
                 log_period=10,
                 **kwargs):
        """Init."""
        super().__init__(logdir, env_fn, **kwargs)
        self.gamma = gamma
        self.frame_stack = frame_stack
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_period = update_period
        self.target_update_period = target_update_period - (
            target_update_period % self.update_period)
        self.log_period = log_period
        self.eps_schedule = LinearSchedule(exploration_timesteps, final_eps,
                                           1.0)

        self.qf = qf_fn(VecFrameStack(self.env,
                                      self.frame_stack)).to(self.device)
        self.qf_targ = qf_fn(VecFrameStack(self.env,
                                           self.frame_stack)).to(self.device)
        self.opt = optimizer(self.qf.parameters())
        if huber_loss:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
        else:
            self.criterion = torch.nn.MSELoss(reduction='none')
        self._actor = EpsilonGreedyActor(self.qf, self.eps_schedule,
                                         self.env.action_space)

        self.buffer = ReplayBuffer(self.buffer_size, self.frame_stack)
        self.data_manager = ReplayBufferDataManager(self.buffer,
                                                    self.env,
                                                    self._actor,
                                                    self.device,
                                                    self.learning_starts,
                                                    self.update_period)

    def state_dict(self):
        """State dict."""
        return {
            'qf': self.qf.state_dict(),
            'qf_targ': self.qf.state_dict(),
            'opt': self.opt.state_dict(),
            '_actor': self._actor.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load dict."""
        self.qf.load_state_dict(state_dict['qf'])
        self.qf_targ.load_state_dict(state_dict['qf_targ'])
        self.opt.load_state_dict(state_dict['opt'])
        self._actor.load_state_dict(state_dict['_actor'])

    def _compute_target(self, rew, next_ob, done):
        qtarg = self.qf_targ(next_ob).max_q
        return rew + (1.0 - done) * self.gamma * qtarg

    def _get_batch(self):
        return self.buffer.sample(self.batch_size)

    def loss(self, batch):
        """Compute loss."""
        for k in batch:
            batch[k] = torch.from_numpy(batch[k]).to(self.device)

        q = self.qf(batch['obs'], batch['action']).value

        with torch.no_grad():
            target = self._compute_target(batch['reward'], batch['next_obs'],
                                          batch['done'])

        assert target.shape == q.shape
        loss = self.criterion(target, q).mean()
        if self.t % self.log_period < self.update_period:
            logger.add_scalar('alg/maxq', torch.max(q).detach().cpu().numpy(),
                              self.t, time.time())
            logger.add_scalar('alg/loss', loss.detach().cpu().numpy(), self.t,
                              time.time())
            logger.add_scalar('alg/epsilon',
                              self.eps_schedule.value(self._actor.t),
                              self.t, time.time())
        return loss

    def step(self):
        """Step."""
        self.t += self.data_manager.step_until_update()
        if self.t % self.target_update_period == 0:
            self.qf_targ.load_state_dict(self.qf.state_dict())

        self.opt.zero_grad()
        loss = self.loss(self._get_batch())
        loss.backward()
        self.opt.step()

    def evaluate(self):
        """Evaluate."""
        eval_env = VecFrameStack(self.env, self.frame_stack)
        self.rl_evaluate(eval_env, self.qf)
        self.rl_record(eval_env, self.qf)

    def _save(self, state_dict):
        # save buffer seperately and only once (because it can be huge)
        np.savez(os.path.join(self.ckptr.ckptdir, 'buffer.npz'),
                 **self.buffer.state_dict())
        super()._save(state_dict)

    def _load(self, state_dict):
        self.buffer.load_state_dict(np.load(os.path.join(self.ckptr.ckptdir,
                                                         'buffer.npz')))
        super()._load(state_dict)
        self.data_manager.manual_reset()


if __name__ == '__main__':
    import unittest
    import shutil
    from dl.rl.envs import make_atari_env
    from dl.rl.modules import DiscreteQFunctionBase
    from dl.rl.util import conv_out_shape
    import torch.nn.functional as F

    class NatureDQN(DiscreteQFunctionBase):
        """Deep network from https://www.nature.com/articles/nature14236."""

        def build(self):
            """Build."""
            self.conv1 = nn.Conv2d(4, 32, 8, 4)
            self.conv2 = nn.Conv2d(32, 64, 4, 2)
            self.conv3 = nn.Conv2d(64, 64, 3, 1)
            shape = self.observation_space.shape[1:]
            for c in [self.conv1, self.conv2, self.conv3]:
                shape = conv_out_shape(shape, c)
            self.nunits = 64 * np.prod(shape)
            self.fc = nn.Linear(self.nunits, 512)
            self.qf = nn.Linear(512, self.action_space.n)

        def forward(self, x):
            """Forward."""
            x = x.float() / 255.
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.fc(x.view(-1, self.nunits)))
            return self.qf(x)

    class TestDQN(unittest.TestCase):
        """Test case."""

        def test_ql(self):
            """Test."""
            def env_fn(rank):
                return make_atari_env('Pong', rank=rank, frame_stack=4)

            def qf_fn(env):
                return QFunction(NatureDQN(env.observation_space,
                                           env.action_space))

            ql = DQN('logs',
                     env_fn,
                     qf_fn,
                     learning_starts=100,
                     buffer_size=200,
                     update_period=4,
                     exploration_timesteps=500,
                     target_update_period=100,
                     maxt=1000,
                     eval=True,
                     eval_period=1000)
            ql.train()
            assert np.allclose(ql.eps_schedule.value(ql.t), 0.1)
            shutil.rmtree('logs')

    unittest.main()
