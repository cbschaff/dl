from dl import Trainer
from dl.modules import QFunction
from dl.util import ReplayBuffer, PrioritizedReplayBuffer, Checkpointer
from dl.util import logger, find_monitor
from baselines.common.schedules import LinearSchedule
import gin, os, time
import torch
import numpy as np
from collections import deque


@gin.configurable(blacklist=['logdir','env'])
class QLearning(Trainer):
    def __init__(self,
                 logdir,
                 env,
                 optimizer,
                 gamma=0.99,
                 batch_size=32,
                 update_period=4,
                 frame_stack=4,
                 huber_loss=True,
                 learning_starts=50000,
                 exploration_timesteps=1000000,
                 final_eps=0.1,
                 eval_eps=0.05,
                 eval_nepisodes=100,
                 target_update_period=10000,
                 double_dqn=False,
                 buffer=ReplayBuffer,
                 buffer_size=1000000,
                 prioritized_replay=False,
                 replay_alpha=0.6,
                 replay_beta=0.4,
                 t_beta_max=int(1e7),
                 gpu=True,
                 log_period=1000,
                 trainer_kwargs={}
                 ):
        super().__init__(logdir, **trainer_kwargs)
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_period = update_period
        self.frame_stack = frame_stack
        self.learning_starts = learning_starts
        self.target_update_period = target_update_period
        self.double_dqn = double_dqn
        self.eval_eps = eval_eps
        self.log_period = log_period
        self.prioritized_replay = prioritized_replay
        self.buffer = buffer(buffer_size, frame_stack)
        if prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(self.buffer, alpha=replay_alpha)
            self.beta_schedule = LinearSchedule(t_beta_max, 1.0, replay_beta)
        self.eps_schedule = LinearSchedule(exploration_timesteps, final_eps, 1.0)

        s = env.observation_space.shape
        ob_shape = (s[0] * self.frame_stack, *s[1:])
        self.net = QFunction(ob_shape, env.action_space)
        self.target_net = QFunction(ob_shape, env.action_space)
        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.target_net.to(self.device)
        self.opt = optimizer(self.net.parameters())
        if huber_loss:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
        else:
            self.criterion = torch.nn.MSELoss(reduction='none')
        self.t, self.t_start = 0,0
        logger.configure(os.path.join(logdir, 'logs'), ['stdout', 'log', 'json'])
        self.losses = []

        self._reset()

    def _reset(self):
        self.buffer.env_reset()
        self._ob = self.env.reset()

    def state_dict(self):
        return {
            'net': self.net.state_dict(),
            'opt': self.opt.state_dict(),
            't':   self.t,
            'buffer': self.buffer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict['net'])
        self.target_net.load_state_dict(state_dict['net'])
        self.opt.load_state_dict(state_dict['opt'])
        self.buffer.load_state_dict(state_dict['buffer'])
        self.t = state_dict['t']
        self._reset()

    def save(self):
        state = self.state_dict()
        # save buffer seperately and only once (because it is huge)
        buffer_state_dict = state['buffer']
        state_without_buffer = dict(state)
        del state_without_buffer['buffer']
        self.ckptr.save(state_without_buffer, self.t)
        np.savez(os.path.join(self.ckptr.ckptdir, 'buffer.npz'), **buffer_state_dict)

    def load(self, t=None):
        state = self.ckptr.load(t)
        state['buffer'] = np.load(os.path.join(self.ckptr.ckptdir, 'buffer.npz'))
        self.load_state_dict(state)
        self.t_start = 0 if t is None else t

    def act(self):
        idx = self.buffer.store_frame(self._ob)
        if self.eps_schedule.value(self.t) > np.random.rand():
            ac = self.env.action_space.sample()
        else:
            x = self.buffer.encode_recent_observation()
            with torch.no_grad():
                x = torch.from_numpy(x).to(self.device)
                ac = self.net(x[None]).action.cpu().numpy()
        self._ob, r, done, _ = self.env.step(ac)
        self.buffer.store_effect(idx, ac, r, done)
        if done:
            self._ob = self.env.reset()
        self.t += 1

    def loss(self, batch):
        if self.prioritized_replay:
            idx = batch[-1]
            ob, ac, rew, next_ob, done, weight = [torch.from_numpy(x).to(self.device) for x in batch[:-1]]
        else:
            ob, ac, rew, next_ob, done = [torch.from_numpy(x).to(self.device) for x in batch]

        qs = self.net(ob).qvals
        q = qs.gather(1, ac.long().unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_ac = self.net(next_ob).action
                qtargs = self.target_net(next_ob).qvals
                qtarg = qtargs.gather(1, next_ac.long().unsqueeze(1)).squeeze(1)
            else:
                qtarg = self.target_net(next_ob).maxq
            assert rew.shape == qtarg.shape
            target = rew + (1.0 - done) * self.gamma * qtarg

        assert target.shape == q.shape
        err = self.criterion(target, q)

        if self.prioritized_replay:
            self.buffer.update_priorities(idx, err.detach().cpu().numpy())
            assert err.shape == weight.shape
            err = weight * err
        return err.mean()

    def step(self):
        self.act()
        while self.buffer.num_in_buffer < min(self.learning_starts, self.buffer.size):
            self.act()
        if self.t % self.target_update_period == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        if self.t % self.update_period == 0:
            if self.prioritized_replay:
                beta = self.beta_schedule.value(self.t)
                batch = self.buffer.sample(self.batch_size, beta)
            else:
                batch = self.buffer.sample(self.batch_size)

            self.opt.zero_grad()
            loss = self.loss(batch)
            loss.backward()
            self.opt.step()

            self.losses.append(loss.detach())

        if self.t % self.log_period == 0 and self.t > 0:
            with torch.no_grad():
                meanloss = (sum(self.losses) / self.log_period).cpu().numpy()
            self.losses = []
            logger.log("========================|  Timestep: {}  |========================".format(self.t))
            # Logging stats...
            logger.logkv('Loss', meanloss)
            logger.logkv('timesteps', self.t)
            logger.logkv('fps', int((self.t - self.t_start) / (time.time() - self.time_start)))
            logger.logkv('time_elapsed', time.time() - self.time_start)
            logger.logkv('time spent exploring', self.eps_schedule.value(self.t))

            monitor = find_monitor(self.env)
            if monitor is not None:
                logger.logkv('mean episode length', np.mean(monitor.episode_lengths[-100:]))
                logger.logkv('mean episode reward', np.mean(monitor.episode_rewards[-100:]))
            logger.dumpkvs()

    def evaluate(self):
        import json
        frames = deque(maxlen=self.frame_stack)
        def reset():
            self._reset()
            for _ in range(self.frame_stack - 1):
                frames.append(np.zeros_like(self._ob))
            frames.append(self._ob)

        def get_ob():
            return np.concatenate(frames, axis=0)

        def eps_greedy():
            if self.eval_eps > np.random.rand():
                ac = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    x = torch.from_numpy(get_ob()).to(self.device)
                    ac = self.net(x[None]).action
            return ac

        ep_lengths = []
        ep_rewards = []
        for i in range(self.eval_nepisodes):
            reset()
            done = False
            ep_lengths.append(0)
            ep_rewards.append(0)
            while not done:
                ob, r, done, _ = self.env.step(eps_greedy())
                frames.append(ob)
                ep_lengths[-1] += 1
                ep_rewards[-1] += r
        self._reset()

        outs = {
            'episode_lengths': ep_lengths,
            'episode_rewards': ep_rewards,
            'mean_length': np.mean(ep_lengths),
            'mean_reward': np.mean(ep_rewards),
        }

        os.makedirs(os.path.join(self.logdir, 'eval'))
        with open(os.path.join(self.logdir, f'eval/{self.t}.json'), 'w') as f:
            json.dump(outs, f)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
        logger.reset()


import unittest, shutil
from dl.util import atari_env, load_gin_configs

class TestQLearning(unittest.TestCase):
    def test_ql(self):
        env = atari_env('Pong')
        ql = QLearning('logs', env, learning_starts=100, eval_nepisodes=1, target_update_period=100, trainer_kwargs={'maxt': 1000, 'eval':True, 'eval_period':1000})
        ql.train()
        env = atari_env('Pong')
        ql = QLearning('logs', env, learning_starts=100, eval_nepisodes=1, trainer_kwargs={'maxt': 1000, 'eval':True, 'eval_period':1000})
        ql.train() # loads checkpoint
        assert ql.buffer.num_in_buffer == 1000
        shutil.rmtree('logs')


    def test_double_ql(self):
        env = atari_env('Pong')
        ql = QLearning('logs', env, learning_starts=100, double_dqn=True, trainer_kwargs={'maxt': 1000})
        ql.train()
        shutil.rmtree('logs')

    def test_prioritized_ql(self):
        env = atari_env('Pong')
        ql = QLearning('logs', env, learning_starts=100, double_dqn=True, prioritized_replay=True, trainer_kwargs={'maxt': 1000})
        ql.train()
        shutil.rmtree('logs')

    def test_gpu(self):
        if not torch.cuda.is_available():
            return
        env = atari_env('Pong')
        ql = QLearning('logs', env, learning_starts=100, gpu=True, trainer_kwargs={'maxt': 1000})
        ql.train()
        shutil.rmtree('logs')


if __name__=='__main__':
    load_gin_configs(['../configs/dqn.gin'])
    unittest.main()
