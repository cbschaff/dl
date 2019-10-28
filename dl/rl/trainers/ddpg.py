"""DDPG algorithm.

https://arxiv.org/abs/1509.02971
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


def soft_target_update(target_net, net, tau):
    """Soft update totarget network."""
    for tp, p in zip(target_net.parameters(), net.parameters()):
        tp.data.copy_((1. - tau) * tp.data + tau * p.data)


class OrnsteinUhlenbeck(object):
    """Ornstein-Uhlenbeck process for directed exploration.

    Default parameters are chosen from the DDPg paper.
    """

    def __init__(self, shape, device, theta=0.15, sigma=0.2):
        """Init."""
        self.theta = theta
        self.sigma = sigma
        self.x = torch.zeros(shape, device=device, dtype=torch.float32,
                             requires_grad=False)
        self.dist = torch.distributions.Normal(0., sigma)

    def __call__(self):
        """Sample."""
        with torch.no_grad():
            self.x = (1. - self.theta) * self.x + self.dist.sample()
        return self.x


class DDPGActor(object):
    """DDPG actor."""

    def __init__(self, pi, action_space):
        """Init."""
        self.pi = pi
        self.noise = None
        self.action_space = action_space

    def __call__(self, obs):
        """Act."""
        with torch.no_grad():
            action = self.pi(obs, deterministic=True).action
            if self.noise is None:
                self.noise = OrnsteinUhlenbeck(action.shape, action.device)
                self.low = torch.from_numpy(self.action_space.low).to(
                                                                action.device)
                self.high = torch.from_numpy(self.action_space.high).to(
                                                                action.device)
            normed_action = (action - self.low) / (self.high - self.low)
            noisy_normed_action = normed_action + self.noise()
            noisy_action = noisy_normed_action * (
                                            (self.high - self.low) + self.low)
            clipped_action = torch.max(torch.min(noisy_action,
                                                 self.high),
                                       self.low)
            return {'action': clipped_action}


@gin.configurable(blacklist=['logdir'])
class DDPG(RLTrainer):
    """DDPG algorithm."""

    def __init__(self,
                 logdir,
                 env_fn,
                 policy_fn,
                 qf_fn,
                 optimizer=torch.optim.Adam,
                 buffer_size=10000,
                 frame_stack=1,
                 learning_starts=1000,
                 update_period=1,
                 batch_size=256,
                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 gamma=0.99,
                 target_update_period=1,
                 target_smoothing_coef=0.005,
                 reward_scale=1,
                 log_period=1000,
                 **kwargs):
        """Init."""
        super().__init__(logdir, env_fn, **kwargs)
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.frame_stack = frame_stack
        self.learning_starts = learning_starts
        self.update_period = update_period
        self.batch_size = batch_size
        if target_update_period < self.update_period:
            self.target_update_period = self.update_period
        else:
            self.target_update_period = target_update_period - (
                                target_update_period % self.update_period)
        self.reward_scale = reward_scale
        self.target_smoothing_coef = target_smoothing_coef
        self.log_period = log_period

        eval_env = VecFrameStack(self.env, self.frame_stack)
        self.pi = policy_fn(eval_env)
        self.qf = qf_fn(eval_env)
        self.target_pi = policy_fn(eval_env)
        self.target_qf = qf_fn(eval_env)

        self.pi.to(self.device)
        self.qf.to(self.device)
        self.target_pi.to(self.device)
        self.target_qf.to(self.device)

        self.opt_pi = optimizer(self.pi.parameters(), lr=policy_lr)
        self.opt_qf = optimizer(self.qf.parameters(), lr=qf_lr)

        self.target_pi.load_state_dict(self.pi.state_dict())
        self.target_qf.load_state_dict(self.qf.state_dict())

        self.buffer = ReplayBuffer(buffer_size, frame_stack)
        self.data_manager = ReplayBufferDataManager(self.buffer,
                                                    self.env,
                                                    DDPGActor(
                                                        self.pi,
                                                        self.env.action_space),
                                                    self.device,
                                                    self.learning_starts,
                                                    self.update_period)

        self.qf_criterion = torch.nn.MSELoss()
        if self.env.action_space.__class__.__name__ == 'Discrete':
            raise ValueError("Action space must be continuous!")

        self.low = torch.from_numpy(self.env.action_space.low).to(self.device)
        self.high = torch.from_numpy(self.env.action_space.high).to(self.device)

    def state_dict(self):
        """State dict."""
        return {
            'pi': self.pi.state_dict(),
            'qf': self.qf.state_dict(),
            'target_pi': self.target_pi.state_dict(),
            'target_qf': self.target_qf.state_dict(),
            'opt_pi': self.opt_pi.state_dict(),
            'opt_qf': self.opt_qf.state_dict(),
            }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.pi.load_state_dict(state_dict['pi'])
        self.qf.load_state_dict(state_dict['qf'])
        self.target_pi.load_state_dict(state_dict['target_pi'])
        self.target_qf.load_state_dict(state_dict['target_qf'])

        self.opt_pi.load_state_dict(state_dict['opt_pi'])
        self.opt_qf.load_state_dict(state_dict['opt_qf'])

    def _norm_actions(self, ac):
        if self.low is not None and self.high is not None:
            return 2 * (ac - self.low) / (self.high - self.low) - 1.0
        else:
            return ac

    def loss(self, batch):
        """Loss function."""
        # compute QFunction loss.
        with torch.no_grad():
            target_action = self.target_pi(batch['next_obs']).dist.mode()
            target_q = self.target_qf(batch['next_obs'], target_action).value
            qtarg = self.reward_scale * batch['reward'] + (
                    (1.0 - batch['done']) * self.gamma * target_q)

        q = self.qf(batch['obs'], self._norm_actions(batch['action'])).value
        assert qtarg.shape == q.shape
        qf_loss = self.qf_criterion(q, qtarg)

        # compute policy loss
        action = self.pi(batch['obs']).dist.mode()
        q = self.qf(batch['obs'], action).value
        pi_loss = -q.mean()

        # log losses
        if self.t % self.log_period < self.update_period:
            logger.add_scalar('loss/qf', qf_loss, self.t, time.time())
            logger.add_scalar('loss/pi', pi_loss, self.t, time.time())
        return pi_loss, qf_loss

    def step(self):
        """Step optimization."""
        self.t += self.data_manager.step_until_update()
        if self.t % self.target_update_period == 0:
            soft_target_update(self.target_pi, self.pi,
                               self.target_smoothing_coef)
            soft_target_update(self.target_qf, self.qf,
                               self.target_smoothing_coef)

        if self.t % self.update_period == 0:
            batch = self.data_manager.sample(self.batch_size)

            pi_loss, qf_loss = self.loss(batch)

            # update
            self.opt_qf.zero_grad()
            qf_loss.backward()
            self.opt_qf.step()

            self.opt_pi.zero_grad()
            pi_loss.backward()
            self.opt_pi.step()

    def evaluate(self):
        """Evaluate."""
        eval_env = VecFrameStack(self.env, self.frame_stack)
        self.rl_evaluate(eval_env, self.pi)
        self.rl_record(eval_env, self.pi)
        if self.data_manager:
            self.data_manager.manual_reset()

    def _save(self, state_dict):
        # save buffer seperately and only once (because it can be huge)
        np.savez(os.path.join(self.ckptr.ckptdir, 'buffer.npz'),
                 **self.buffer.state_dict())
        super()._save(state_dict)

    def _load(self, state_dict):
        self.buffer.load_state_dict(np.load(os.path.join(self.ckptr.ckptdir,
                                                         'buffer.npz')))
        super()._load(state_dict)
        if self.data_manager:
            self.data_manager.manual_reset()


if __name__ == '__main__':
    import unittest
    import shutil
    from dl.rl.envs import make_env
    from dl.rl.modules import PolicyBase, ContinuousQFunctionBase
    from dl.rl.modules import UnnormActionPolicy
    from dl.modules import Delta
    import torch.nn.functional as F

    class PiBase(PolicyBase):
        """Policy network."""

        def build(self):
            """Build Network."""
            self.fc1 = nn.Linear(self.observation_space.shape[0], 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.dist = Delta(32, self.action_space.shape[0])

        def forward(self, x):
            """Forward."""
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.dist(x)

    class QFBase(ContinuousQFunctionBase):
        """Q network."""

        def build(self):
            """Build Network."""
            nin = self.observation_space.shape[0] + self.action_space.shape[0]
            self.fc1 = nn.Linear(nin, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.qvalue = nn.Linear(32, 1)

        def forward(self, x, a):
            """Forward."""
            x = F.relu(self.fc1(torch.cat([x, a], dim=1)))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.qvalue(x)

    def env_fn(rank):
        """Environment function."""
        return make_env('LunarLanderContinuous-v2', rank=rank)

    def policy_fn(env):
        """Create a policy."""
        return UnnormActionPolicy(PiBase(env.observation_space,
                                         env.action_space))

    def qf_fn(env):
        """Create a qfunction."""
        return QFunction(QFBase(env.observation_space, env.action_space))

    class TestDDPG(unittest.TestCase):
        """Test case."""

        def test_sac(self):
            """Test."""
            ddpg = DDPG('logs',
                        env_fn,
                        policy_fn,
                        qf_fn,
                        learning_starts=300,
                        eval_num_episodes=1,
                        buffer_size=500,
                        target_update_period=100,
                        maxt=1000,
                        eval=False,
                        eval_period=1000)
            ddpg.train()
            shutil.rmtree('logs')

    unittest.main()
