"""SAC algorithm.

https://arxiv.org/abs/1801.01290
"""
from dl.rl.trainers import ReplayBufferTrainer
from dl.rl.modules import QFunction, Policy, ValueFunction
from dl.modules import TanhNormal
from dl import logger
import gin
import time
import torch
import torch.nn as nn
import numpy as np
from dl.rl.envs import FrameStack


def soft_target_update(target_net, net, tau):
    """Soft update totarget network."""
    for tp, p in zip(target_net.parameters(), net.parameters()):
        tp.data.copy_((1. - tau) * tp.data + tau * p.data)


@gin.configurable(whitelist=['base'])
class UnnormActionPolicy(Policy):
    """Unnormalize the outputs of a TanhNormal distribution."""

    def forward(self, *args, **kwargs):
        """Forward."""
        outs = super().forward(*args, **kwargs)
        if self.base.action_space.__class__.__name__ == 'Box':
            low = self.base.action_space.low
            high = self.base.action_space.high
            if low is not None and high is not None:
                low = torch.from_numpy(low)
                high = torch.from_numpy(high)
                ac = low + 0.5 * (outs.action + 1) * (high - low)
                outs = self.outputs(action=ac, value=outs.value, dist=outs.dist,
                                    state_out=outs.state_out)
        return outs


@gin.configurable(blacklist=['logdir'])
class SAC(ReplayBufferTrainer):
    """SAC algorithm."""

    def __init__(self,
                 logdir,
                 env_fn,
                 policy_fn,
                 qf_fn,
                 vf_fn,
                 optimizer=torch.optim.Adam,
                 batch_size=256,
                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 vf_lr=1e-3,
                 policy_mean_reg_weight=1e-3,
                 gamma=0.99,
                 target_update_period=1,
                 policy_update_period=1,
                 target_smoothing_coef=0.005,
                 automatic_entropy_tuning=True,
                 reparameterization_trick=True,
                 target_entropy=None,
                 reward_scale=1,
                 log_period=1000,
                 **kwargs):
        """Init."""
        super().__init__(logdir, env_fn, **kwargs)
        self.gamma = gamma
        self.batch_size = batch_size
        if target_update_period < self.update_period:
            self.target_update_period = self.update_period
        else:
            self.target_update_period = target_update_period - (
                                target_update_period % self.update_period)
        if policy_update_period < self.update_period:
            self.policy_update_period = self.update_period
        else:
            self.policy_update_period = policy_update_period - (
                                policy_update_period % self.update_period)
        self.rsample = reparameterization_trick
        self.reward_scale = reward_scale
        self.target_smoothing_coef = target_smoothing_coef
        self.log_period = log_period

        self.eval_env = self.make_eval_env()
        self.pi = policy_fn(self.eval_env)
        self.qf1 = qf_fn(self.eval_env)
        self.qf2 = qf_fn(self.eval_env)
        self.vf = vf_fn(self.eval_env)
        self.target_vf = vf_fn(self.eval_env)

        self.pi.to(self.device)
        self.qf1.to(self.device)
        self.qf2.to(self.device)
        self.vf.to(self.device)
        self.target_vf.to(self.device)

        self.opt_pi = optimizer(self.pi.parameters(), lr=policy_lr)
        self.opt_qf1 = optimizer(self.qf1.parameters(), lr=qf_lr)
        self.opt_qf2 = optimizer(self.qf2.parameters(), lr=qf_lr)
        self.opt_vf = optimizer(self.vf.parameters(), lr=vf_lr)
        self.policy_mean_reg_weight = policy_mean_reg_weight

        self.target_vf.load_state_dict(self.vf.state_dict())

        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # heuristic value from Tuomas
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True,
                                         device=self.device)
            self.opt_alpha = optimizer([self.log_alpha], lr=policy_lr)
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.opt_alpha = None

        self.qf_criterion = torch.nn.MSELoss()
        self.vf_criterion = torch.nn.MSELoss()
        self.discrete = self.env.action_space.__class__.__name__ == 'Discrete'

    def _make_eval_env(self):
        return FrameStack(self.env_fn(rank=1), self.frame_stack)

    def state_dict(self):
        """State dict."""
        return {
            'pi': self.pi.state_dict(),
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            'vf': self.vf.state_dict(),
            'opt_pi': self.opt_pi.state_dict(),
            'opt_qf1': self.opt_qf1.state_dict(),
            'opt_qf2': self.opt_qf2.state_dict(),
            'opt_vf': self.opt_vf.state_dict(),
            'log_alpha': (self.log_alpha if self.automatic_entropy_tuning
                          else None),
            'opt_alpha': (self.opt_alpha.state_dict()
                          if self.automatic_entropy_tuning else None)}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.pi.load_state_dict(state_dict['pi'])
        self.qf1.load_state_dict(state_dict['qf1'])
        self.qf2.load_state_dict(state_dict['qf2'])
        self.vf.load_state_dict(state_dict['vf'])
        self.target_vf.load_state_dict(state_dict['vf'])

        self.opt_pi.load_state_dict(state_dict['opt_pi'])
        self.opt_qf1.load_state_dict(state_dict['opt_qf1'])
        self.opt_qf2.load_state_dict(state_dict['opt_qf2'])
        self.opt_vf.load_state_dict(state_dict['opt_vf'])

        if state_dict['log_alpha']:
            with torch.no_grad():
                self.log_alpha.copy_(state_dict['log_alpha'])
        self.opt_vf.load_state_dict(state_dict['opt_vf'])

    def act(self, ob):
        """Get decision from policy."""
        return self.pi(ob).action

    def loss(self, batch):
        """Loss function."""
        ob, ac, rew, next_ob, done = [
            torch.from_numpy(x).to(self.device) for x in batch]

        pi_out = self.pi(ob, reparameterization_trick=self.rsample)
        if self.discrete:
            new_ac = pi_out.action
            logp = pi_out.logp
        else:
            assert isinstance(pi_out.dist, TanhNormal), (
                "It is strongly encouraged that you use a TanhNormal "
                "action distribution for continuous action spaces.")
            if self.rsample:
                new_ac, new_pth_ac = pi_out.dist.rsample(
                                                    return_pretanh_value=True)
            else:
                new_ac, new_pth_ac = pi_out.dist.sample(
                                                    return_pretanh_value=True)
            logp = pi_out.dist.log_prob(new_ac, new_pth_ac)
        q1 = self.qf1(ob, ac).value
        q2 = self.qf2(ob, ac).value
        v = self.vf(ob).value

        # alpha loss
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (
                            logp + self.target_entropy).detach()).mean()
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        # qf loss
        vtarg = self.target_vf(next_ob).value
        qtarg = self.reward_scale * rew + (1.0 - done) * self.gamma * vtarg
        assert qtarg.shape == q1.shape
        assert qtarg.shape == q2.shape
        qf1_loss = self.qf_criterion(q1, qtarg.detach())
        qf2_loss = self.qf_criterion(q2, qtarg.detach())

        # vf loss
        q1_new = self.qf1(ob, new_ac).value
        q2_new = self.qf2(ob, new_ac).value
        q = torch.min(q1_new, q2_new)
        vtarg = q - alpha * logp
        assert v.shape == vtarg.shape
        vf_loss = self.vf_criterion(v, vtarg.detach())

        # pi loss
        pi_loss = None
        if self.t % self.policy_update_period == 0:
            if self.rsample:
                assert q.shape == logp.shape
                pi_loss = (alpha*logp - q).mean()
            else:
                pi_targ = q - v
                assert pi_targ.shape == logp.shape
                pi_loss = (logp * (alpha * logp - pi_targ).detach()).mean()

            if not self.discrete:  # continuous action space.
                pi_loss += self.policy_mean_reg_weight * (
                                            pi_out.dist.normal.mean**2).mean()

            # log pi loss about as frequently as other losses
            if self.t % self.log_period < self.policy_update_period:
                logger.add_scalar('loss/pi', pi_loss, self.t, time.time())

        if self.t % self.log_period < self.update_period:
            if self.automatic_entropy_tuning:
                logger.add_scalar('ent/log_alpha',
                                  self.log_alpha.detach().cpu().numpy(), self.t,
                                  time.time())
                scalars = {"target": self.target_entropy,
                           "entropy": -torch.mean(
                                        logp.detach()).cpu().numpy().item()}
                logger.add_scalars('ent/entropy', scalars, self.t, time.time())
            else:
                logger.add_scalar(
                        'ent/entropy',
                        -torch.mean(logp.detach()).cpu().numpy().item(),
                        self.t, time.time())
            logger.add_scalar('loss/qf1', qf1_loss, self.t, time.time())
            logger.add_scalar('loss/qf2', qf2_loss, self.t, time.time())
            logger.add_scalar('loss/vf', vf_loss, self.t, time.time())
        return pi_loss, qf1_loss, qf2_loss, vf_loss

    def step(self):
        """Step optimization."""
        self.step_until_update()
        if self.t % self.target_update_period == 0:
            soft_target_update(self.target_vf, self.vf,
                               self.target_smoothing_coef)

        if self.t % self.update_period == 0:
            batch = self.buffer.sample(self.batch_size)

            pi_loss, qf1_loss, qf2_loss, vf_loss = self.loss(batch)

            # update
            self.opt_qf1.zero_grad()
            qf1_loss.backward()
            self.opt_qf1.step()

            self.opt_qf2.zero_grad()
            qf2_loss.backward()
            self.opt_qf2.step()

            self.opt_vf.zero_grad()
            vf_loss.backward()
            self.opt_vf.step()

            if pi_loss:
                self.opt_pi.zero_grad()
                pi_loss.backward()
                self.opt_pi.step()

    def evaluate(self):
        """Evaluate."""
        self.rl_evaluate(self.pi)
        self.rl_record(self.pi)


if __name__ == '__main__':
    import unittest
    import shutil
    from dl.rl.envs import make_env
    from dl.rl.modules import PolicyBase, ContinuousQFunctionBase
    from dl.rl.modules import ValueFunctionBase
    from dl.modules import TanhDiagGaussian
    import torch.nn.functional as F

    class PiBase(PolicyBase):
        """Policy network."""

        def build(self):
            """Build Network."""
            self.fc1 = nn.Linear(self.observation_space.shape[0], 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.dist = TanhDiagGaussian(32, self.action_space.shape[0])

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

    class VFBase(ValueFunctionBase):
        """Value network."""

        def build(self):
            """Build Network."""
            self.fc1 = nn.Linear(self.observation_space.shape[0], 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.value = nn.Linear(32, 1)

        def forward(self, x):
            """Forward."""
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.value(x)

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

    def vf_fn(env):
        """Create a value function."""
        return ValueFunction(VFBase(env.observation_space, env.action_space))

    class TestSAC(unittest.TestCase):
        """Test case."""

        def test_sac(self):
            """Test."""
            sac = SAC('logs',
                      env_fn,
                      policy_fn,
                      qf_fn,
                      vf_fn,
                      learning_starts=300,
                      eval_num_episodes=1,
                      buffer_size=500,
                      target_update_period=100,
                      maxt=1000,
                      eval=False,
                      eval_period=1000,
                      reparameterization_trick=True)
            sac.train()
            shutil.rmtree('logs')

    unittest.main()
