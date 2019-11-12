"""TD3 algorithm.

https://arxiv.org/abs/1802.09477

This algorithm improves on DDPG by adding:
- A form of Double Q Learning to reduce overestimation
- Noise to target policy to avoid propagating unrealistic value estimates
- Delay policy updates to have better value estimates
"""
from dl.rl.trainers import RLTrainer
from dl.rl.data_collection import ReplayBufferDataManager, ReplayBuffer
from dl import logger, nest
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


class TD3Actor(object):
    """TD3 actor."""

    def __init__(self, pi, action_space, sigma):
        """Init."""
        self.pi = pi
        self.noise = None
        self.action_space = action_space
        self.sigma = sigma
        self.low = None
        self.high = None

    def __call__(self, obs):
        """Act."""
        with torch.no_grad():
            action = self.pi(obs, deterministic=True).normed_action
            noisy_action = self.add_noise_to_action(action)
            return {'action': self.unnorm_action(noisy_action)}

    def add_noise_to_action(self, action):
        """Add exploration noise."""
        noise = torch.randn_like(action) * self.sigma
        return (action + noise).clamp(-1., 1.)

    def unnorm_action(self, action):
        """Unnormalize action."""
        if self.low is None:
            self.low = torch.from_numpy(self.action_space.low).to(
                                                            action.device)
            self.high = torch.from_numpy(self.action_space.high).to(
                                                            action.device)
        return (action + 1.0) / 2.0 * (self.high - self.low) + self.low

    def update_sigma(self, sigma):
        """Update noise standard deviation."""
        self.sigma = sigma


@gin.configurable(blacklist=['logdir'])
class TD3(RLTrainer):
    """TD3 algorithm."""

    def __init__(self,
                 logdir,
                 env_fn,
                 policy_fn,
                 qf_fn,
                 optimizer=torch.optim.Adam,
                 buffer_size=int(1e6),
                 frame_stack=1,
                 learning_starts=10000,
                 update_period=1,
                 batch_size=256,
                 lr=3e-4,
                 policy_update_period=2,
                 target_smoothing_coef=0.005,
                 reward_scale=1,
                 gamma=0.99,
                 exploration_noise=0.1,
                 policy_noise=0.2,
                 policy_noise_clip=0.5,
                 log_period=1000,
                 **kwargs):
        """Init."""
        super().__init__(logdir, env_fn, **kwargs)
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.frame_stack = frame_stack
        self.learning_starts = learning_starts
        self.update_period = update_period
        if policy_update_period < self.update_period:
            self.policy_update_period = self.update_period
        else:
            self.policy_update_period = policy_update_period - (
                                policy_update_period % self.update_period)
        self.reward_scale = reward_scale
        self.target_smoothing_coef = target_smoothing_coef
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.log_period = log_period

        self.policy_fn = policy_fn
        self.qf_fn = qf_fn
        eval_env = VecFrameStack(self.env, self.frame_stack)
        self.pi = policy_fn(eval_env)
        self.qf1 = qf_fn(eval_env)
        self.qf2 = qf_fn(eval_env)
        self.target_pi = policy_fn(eval_env)
        self.target_qf1 = qf_fn(eval_env)
        self.target_qf2 = qf_fn(eval_env)

        self.pi.to(self.device)
        self.qf1.to(self.device)
        self.qf2.to(self.device)
        self.target_pi.to(self.device)
        self.target_qf1.to(self.device)
        self.target_qf2.to(self.device)

        self.optimizer = optimizer
        self.lr = lr
        self.opt_pi = optimizer(self.pi.parameters(), lr=lr)
        self.opt_qf = optimizer(list(self.qf1.parameters())
                                + list(self.qf2.parameters()), lr=lr)

        self.target_pi.load_state_dict(self.pi.state_dict())
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        self._actor = TD3Actor(self.pi, self.env.action_space,
                               exploration_noise)
        self.buffer = ReplayBuffer(buffer_size, frame_stack)
        self.data_manager = ReplayBufferDataManager(self.buffer,
                                                    self.env,
                                                    self._actor,
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
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            'target_pi': self.target_pi.state_dict(),
            'target_qf1': self.target_qf1.state_dict(),
            'target_qf2': self.target_qf2.state_dict(),
            'opt_pi': self.opt_pi.state_dict(),
            'opt_qf': self.opt_qf.state_dict(),
            }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.pi.load_state_dict(state_dict['pi'])
        self.qf1.load_state_dict(state_dict['qf1'])
        self.qf2.load_state_dict(state_dict['qf2'])
        self.target_pi.load_state_dict(state_dict['target_pi'])
        self.target_qf1.load_state_dict(state_dict['target_qf1'])
        self.target_qf2.load_state_dict(state_dict['target_qf2'])

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
            target_action = self.target_pi(batch['next_obs']).normed_action
            noise = (
                     torch.randn_like(target_action) * self.policy_noise
            ).clamp(-self.policy_noise_clip, self.policy_noise_clip)
            target_action = (target_action + noise).clamp(-1., 1.)
            target_q1 = self.target_qf1(batch['next_obs'], target_action).value
            target_q2 = self.target_qf2(batch['next_obs'], target_action).value
            target_q = torch.min(target_q1, target_q2)
            qtarg = self.reward_scale * batch['reward'] + (
                    (1.0 - batch['done']) * self.gamma * target_q)

        normed_action = self._norm_actions(batch['action'])
        q1 = self.qf1(batch['obs'], normed_action).value
        q2 = self.qf2(batch['obs'], normed_action).value
        assert qtarg.shape == q1.shape
        assert qtarg.shape == q2.shape
        qf_loss = self.qf_criterion(q1, qtarg) + self.qf_criterion(q2, qtarg)

        # compute policy loss
        if self.t % self.policy_update_period == 0:
            action = self.pi(batch['obs'], deterministic=True).normed_action
            q = self.qf1(batch['obs'], action).value
            pi_loss = -q.mean()
        else:
            pi_loss = torch.zeros_like(qf_loss)

        # log losses
        if self.t % self.log_period < self.update_period:
            logger.add_scalar('loss/qf', qf_loss, self.t, time.time())
            if self.t % self.policy_update_period == 0:
                logger.add_scalar('loss/pi', pi_loss, self.t, time.time())
        return pi_loss, qf_loss

    def step(self):
        """Step optimization."""
        self.t += self.data_manager.step_until_update()
        batch = self.data_manager.sample(self.batch_size)

        pi_loss, qf_loss = self.loss(batch)

        # update
        self.opt_qf.zero_grad()
        qf_loss.backward()
        self.opt_qf.step()

        if self.t % self.policy_update_period == 0:
            self.opt_pi.zero_grad()
            pi_loss.backward()
            self.opt_pi.step()

            # update target networks
            soft_target_update(self.target_pi, self.pi,
                               self.target_smoothing_coef)
            soft_target_update(self.target_qf1, self.qf1,
                               self.target_smoothing_coef)
            soft_target_update(self.target_qf2, self.qf2,
                               self.target_smoothing_coef)

    def evaluate(self):
        """Evaluate."""
        eval_env = VecFrameStack(self.env, self.frame_stack)
        self.rl_evaluate(eval_env, self.pi)
        self.rl_record(eval_env, self.pi)
        if self.data_manager:
            self.data_manager.manual_reset()

    def _save(self, state_dict):
        # save buffer seperately and only once (because it can be huge)
        buffer_dict = self.buffer.state_dict()
        np.savez(os.path.join(self.ckptr.ckptdir, 'buffer.npz'),
                 *nest.flatten(buffer_dict))
        super()._save(state_dict)

    def _load(self, state_dict):
        super()._load(state_dict)
        # initialize data format of buffer if needed.
        self.data_manager.env_step_and_store_transition()
        buffer_dict = self.buffer.state_dict()
        buffer_state = dict(np.load(os.path.join(self.ckptr.ckptdir,
                                                 'buffer.npz')))
        buffer_state = nest.flatten(buffer_state)
        self.buffer.load_state_dict(nest.pack_sequence_as(buffer_state,
                                                          buffer_dict))
        if self.data_manager:
            self.data_manager.manual_reset()


if __name__ == '__main__':
    import unittest
    import shutil
    from dl.rl.envs import make_env
    from dl.rl.modules import QFunction
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
            td3 = TD3('logs',
                      env_fn,
                      policy_fn,
                      qf_fn,
                      learning_starts=300,
                      eval_num_episodes=1,
                      buffer_size=500,
                      policy_update_period=2,
                      maxt=1000,
                      eval=False,
                      eval_period=1000)
            td3.train()
            td3.load()
            shutil.rmtree('logs')

    unittest.main()
