"""
Prioritized Replay DQN algorithm
https://arxiv.org/abs/1511.05952
"""

from dl.rl.trainers import DoubleDQN
from dl.rl.util import PrioritizedReplayBuffer
from baselines.common.schedules import LinearSchedule
import gin, torch, time
from dl import logger

@gin.configurable(blacklist=['logdir'])
class PrioritizedReplayDQN(DoubleDQN):
    def __init__(self,
                 logdir,
                 replay_alpha=0.6,
                 replay_beta=0.4,
                 t_beta_max=int(1e7),
                 **kwargs
                ):
        super().__init__(logdir, **kwargs)
        self.buffer = PrioritizedReplayBuffer(self.buffer, alpha=replay_alpha)
        self.beta_schedule = LinearSchedule(t_beta_max, 1.0, replay_beta)

    def _get_batch(self):
        beta = self.beta_schedule.value(self.t)
        return self.buffer.sample(self.batch_size, beta)

    def loss(self, batch):
        idx = batch[-1]
        ob, ac, rew, next_ob, done, weight = [torch.from_numpy(x).to(self.device) for x in batch[:-1]]

        q = self.qf(ob, ac).value

        with torch.no_grad():
            target = self._compute_target(rew, next_ob, done)

        assert target.shape == q.shape
        err = self.criterion(target, q)
        self.buffer.update_priorities(idx, err.detach().cpu().numpy() + 1e-6)
        assert err.shape == weight.shape
        err = weight * err
        loss = err.mean()

        if self.t % self.log_period < self.update_period:
            logger.add_scalar('alg/maxq', torch.max(q).detach().cpu().numpy(), self.t, time.time())
            logger.add_scalar('alg/loss', loss.detach().cpu().numpy(), self.t, time.time())
            logger.add_scalar('alg/epsilon', self.eps_schedule.value(self.t), self.t, time.time())
        return loss
