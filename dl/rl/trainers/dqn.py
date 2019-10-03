"""DQN algorithm.

https://www.nature.com/articles/nature14236
"""
from dl.rl.trainers import ReplayBufferTrainer
from dl.rl.modules import QFunction
from dl import logger
import gin
import time
import torch
import torch.nn as nn
import numpy as np
from dl.rl.envs import FrameStack, EpsilonGreedy
from baselines.common.schedules import LinearSchedule


@gin.configurable(blacklist=['logdir'])
class DQN(ReplayBufferTrainer):
    """DQN algorithm."""

    def __init__(self,
                 logdir,
                 env_fn,
                 qfunction=QFunction,
                 optimizer=torch.optim.RMSprop,
                 gamma=0.99,
                 huber_loss=True,
                 exploration_timesteps=1000000,
                 final_eps=0.1,
                 eval_eps=0.05,
                 target_update_period=10000,
                 batch_size=32,
                 log_period=10,
                 **kwargs):
        """Init."""
        super().__init__(logdir, env_fn, **kwargs)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_period = target_update_period - (
            target_update_period % self.update_period)
        self.eval_eps = eval_eps
        self.log_period = log_period
        self.eps_schedule = LinearSchedule(exploration_timesteps, final_eps,
                                           1.0)

        self.eval_env = self.make_eval_env()
        self.qf = qfunction(self.eval_env).to(self.device)
        self.qf_targ = qfunction(self.eval_env).to(self.device)
        self.opt = optimizer(self.qf.parameters())
        if huber_loss:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
        else:
            self.criterion = torch.nn.MSELoss(reduction='none')

    def _make_eval_env(self):
        return EpsilonGreedy(FrameStack(self.env_fn(rank=1),
                                        self.frame_stack),
                             self.eval_eps)

    def state_dict(self):
        """State dict."""
        return {
            'qf': self.qf.state_dict(),
            'qf_targ': self.qf.state_dict(),
            'opt': self.opt.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load dict."""
        self.qf.load_state_dict(state_dict['qf'])
        self.qf_targ.load_state_dict(state_dict['qf_targ'])
        self.opt.load_state_dict(state_dict['opt'])

    def act(self, ob):
        """Epsilon greedy action."""
        if self.eps_schedule.value(self.t) > np.random.rand():
            return torch.from_numpy(
                np.array(self.env.action_space.sample()))[None]
        else:
            return self.qf(ob).action

    def _compute_target(self, rew, next_ob, done):
        qtarg = self.qf_targ(next_ob).max_q
        return rew + (1.0 - done) * self.gamma * qtarg

    def _get_batch(self):
        return self.buffer.sample(self.batch_size)

    def loss(self, batch):
        """Compute loss."""
        ob, ac, rew, next_ob, done = [torch.from_numpy(x).to(self.device)
                                      for x in batch]

        q = self.qf(ob, ac).value

        with torch.no_grad():
            target = self._compute_target(rew, next_ob, done)

        assert target.shape == q.shape
        loss = self.criterion(target, q).mean()
        if self.t % self.log_period < self.update_period:
            logger.add_scalar('alg/maxq', torch.max(q).detach().cpu().numpy(),
                              self.t, time.time())
            logger.add_scalar('alg/loss', loss.detach().cpu().numpy(), self.t,
                              time.time())
            logger.add_scalar('alg/epsilon', self.eps_schedule.value(self.t),
                              self.t, time.time())
        return loss

    def step(self):
        """Step."""
        self.step_until_update()
        if self.t % self.target_update_period == 0:
            self.qf_targ.load_state_dict(self.qf.state_dict())

        if self.t % self.update_period == 0:
            self.opt.zero_grad()
            loss = self.loss(self._get_batch())
            loss.backward()
            self.opt.step()

    def evaluate(self):
        """Evaluate."""
        self.rl_evaluate(self.qf)
        self.rl_record(self.qf)


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
            ql = DQN('logs',
                     env_fn,
                     qfunction=lambda env: QFunction(env, NatureDQN),
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
