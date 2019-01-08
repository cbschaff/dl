
"""
PPO RL algorithm.
https://arxiv.org/abs/1707.06347
"""
from dl import Trainer
from dl.modules import Policy
from dl.util import RolloutStorage
from dl.util import logger, find_monitor, VecMonitor
from baselines.common.schedules import LinearSchedule
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import gin, os, time, json
import torch
import torch.nn as nn
import numpy as np



@gin.configurable(blacklist=['logdir'])
class PPO(Trainer):
    def __init__(self,
                 logdir,
                 env_fn,
                 nenv,
                 optimizer,
                 batch_size=32,
                 steps_per_iter=128,
                 epochs_per_iter=10,
                 max_grad_norm=None,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 clip_param=0.2,
                 gamma=0.99,
                 lambda_=1.0,
                 huber_loss=True,
                 norm_observations=True,
                 norm_advantages=True,
                 eval_nepisodes=100,
                 gpu=True,
                 **trainer_kwargs
    ):
        super().__init__(logdir, **trainer_kwargs)
        logger.configure(os.path.join(logdir, 'logs'), ['stdout', 'log', 'json'])
        def _env(rank):
            def _thunk():
                return env_fn(rank)
            return _thunk
        if nenv > 1:
            self.env = SubprocVecEnv([_env(i) for i in range(nenv)])
        else:
            self.env = DummyVecEnv([_env(0)])
        self.env = VecMonitor(self.env, max_history=100)
        self.nenv = nenv
        self.env_fn = env_fn
        self.batch_size = batch_size
        self.steps_per_iter = steps_per_iter
        self.epochs_per_iter = epochs_per_iter
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.clip_param = clip_param
        self.gamma = gamma
        self.lambda_ = lambda_
        self.norm_observations = norm_observations
        self.norm_advantages = norm_advantages
        self.eval_nepisodes = eval_nepisodes

        self.net = self._make_policy(self.env.observation_space.shape, self.env.action_space)
        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.opt = optimizer(self.net.parameters())
        self.recurrent_state_size = self.net.recurrent_state_size()
        if self.recurrent_state_size is None:
            self.recurrent = False
            self.rollout = RolloutStorage(steps_per_iter, self.nenv, device=self.device, other_keys=['logp'])
            self.init_state = None
        else:
            self.recurrent_keys = [f'state{i}' for i in range(len(self.recurrent_state_size))]
            self.rollout = RolloutStorage(steps_per_iter, self.nenv, device=self.device, other_keys=['logp'], recurrent_state_keys=self.recurrent_keys)
            self.init_state = []
            for i,state in enumerate(self.recurrent_state_size):
                self.init_state.append(torch.zeros(size=[self.nenv] + list(state.shape), device=self.device))

        if huber_loss:
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
        else:
            self.criterion = torch.nn.MSELoss(reduction='none')
        self.t, self.t_start = 0,0
        self.losses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}
        self.meanlosses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}

        self._ob = torch.from_numpy(self.env.reset()).to(self.device)
        self._mask = torch.FloatTensor([0. for _ in range(self.nenv)], device=self.device)
        self._state = self.init_state

    def _make_policy(self, ob_shape, action_space):
        return Policy(ob_shape, action_space, norm_observations=self.norm_observations)

    def state_dict(self):
        state = {}
        state['net'] = self.net.state_dict()
        state['opt'] = self.opt.state_dict()
        state['t']   = self.t

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict['net'])
        self.opt.load_state_dict(state_dict['opt'])
        self.t = state_dict['t']

    def load(self, t=None):
        super().load(t)
        self.t_start = self.t

    def act(self):
        with torch.no_grad():
            if self.recurrent:
                outs = self.net(self._ob, mask=self._mask, state_in=self._state)
            else:
                outs = self.net(self._ob)
        ob, r, done, _ = self.env.step(outs.action.cpu().numpy())
        data = {}
        data['ob'] = self._ob
        data['ac'] = outs.action
        data['r']  = torch.from_numpy(r).float()
        data['mask'] = self._mask
        data['vpred'] = outs.value
        data['logp'] = outs.logp
        if self.recurrent:
            for i,name in self.recurrent_keys:
                data[name] = self._state[i]
        self.rollout.insert(data)
        self._ob = torch.from_numpy(ob).to(self.device)
        self._mask = torch.FloatTensor([0.0 if done_ else 1.0 for done_ in done], device=self.device)
        self._state = outs.state_out
        self.t += self.nenv

    def loss(self, batch):
        if self.recurrent:
            state = [batch[k] for k in self.recurrent_keys]
            outs = self.net(batch['ob'], mask=batch['mask'], state_in=state)
        else:
            outs = self.net(batch['ob'])

        # compute policy loss
        assert outs.logp.shape == batch['logp'].shape
        ratio = torch.exp(outs.logp - batch['logp'])
        assert ratio.shape == batch['atarg'].shape
        ploss1 = ratio * batch['atarg']
        ploss2 = torch.clamp(ratio, 1.0-self.clip_param, 1.0+self.clip_param) * batch['atarg']
        pi_loss = -torch.min(ploss1, ploss2).mean()
        self.losses['pi'].append(pi_loss)

        # compute value loss
        vloss1 = self.criterion(outs.value, batch['vtarg'])
        vpred_clipped = batch['vpred'] + (outs.value - batch['vpred']).clamp(-self.clip_param, self.clip_param)
        vloss2 = self.criterion(vpred_clipped, batch['vtarg'])
        vf_loss = torch.max(vloss1, vloss2).mean()
        self.losses['value'].append(vf_loss)

        # compute entropy loss
        ent_loss = outs.dist.entropy().mean()
        self.losses['ent'].append(ent_loss)

        loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss
        self.losses['tot'].append(loss)
        return loss

    def step(self):
        logger.log("========================|  Iteration: {}  |========================".format(self.t // (self.steps_per_iter*self.nenv)))

        # collect rollout data
        for _ in range(self.steps_per_iter):
            self.act()

        # compute advatage and value targets
        with torch.no_grad():
            if self.recurrent:
                next_value = self.net(self._ob, mask=self._mask, state_in=self._state).value
            else:
                next_value = self.net(self._ob).value
        self.rollout.compute_targets(next_value, self._mask, self.gamma, use_gae=True, lambda_=self.lambda_, norm_advantages=self.norm_advantages)

        # update running norm
        if self.norm_observations:
            batch_mean, batch_var, batch_count = self.rollout.compute_ob_stats()
            self.net.running_norm.update(batch_mean, batch_var, batch_count)

        # update model
        for _ in range(self.epochs_per_iter):
            if self.recurrent:
                sampler = self.rollout.recurrent_generator(self.batch_size)
            else:
                sampler = self.rollout.feed_forward_generator(self.batch_size)

            for batch in sampler:
                self.opt.zero_grad()
                loss = self.loss(batch)
                loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
            self.log_losses()
        self.log()

    def log_losses(self):
        s = 'Losses:  '
        for ln in ['tot', 'pi', 'value', 'ent']:
            with torch.no_grad():
                self.meanlosses[ln].append((sum(self.losses[ln]) / len(self.losses[ln])).cpu().numpy())
            s += '{}: {:08f}  '.format(ln, self.meanlosses[ln][-1])
        logger.log(s)
        self.losses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}

    def log(self):
        with torch.no_grad():
            logger.logkv('Loss - Total', np.mean(self.meanlosses['tot']))
            logger.logkv('Loss - Policy', np.mean(self.meanlosses['pi']))
            logger.logkv('Loss - Value', np.mean(self.meanlosses['value']))
            logger.logkv('Loss - Entropy', np.mean(self.meanlosses['ent']))
        self.meanlosses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}
        # Logging stats...
        logger.logkv('timesteps', self.t)
        logger.logkv('fps', int((self.t - self.t_start) / (time.time() - self.time_start)))
        logger.logkv('time_elapsed', time.time() - self.time_start)

        logger.logkv('mean episode length', np.mean(self.env.episode_lengths))
        logger.logkv('mean episode reward', np.mean(self.env.episode_rewards))
        logger.dumpkvs()

    def evaluate(self):
        # create new env to access true reward function and episode lenghts from the Monitor wrapper (if it exists)
        eval_env = self.env_fn(rank=self.nenv+1)
        ep_lengths = []
        ep_rewards = []
        monitor = find_monitor(eval_env)
        for i in range(self.eval_nepisodes):
            ob = eval_env.reset()
            if self.recurrent:
                mask = torch.FloatTensor([0.]).to(self.device)
                state = self.init_state.to(self.device)
            done = False
            if monitor is None:
                ep_lengths.append(0)
                ep_rewards.append(0)
            while not done:
                ob = torch.from_numpy(ob)[None].to(self.device)
                if self.recurrent:
                    outs = self.net(ob, mask=mask, state_in=state)
                    action = outs.action
                    state = outs.state_out
                else:
                    action = self.net(ob).action
                ob, r, done, _ = eval_env.step(action)
                if self.recurrent:
                    mask = torch.FloatTensor([1.]).to(self.device)
                if monitor is None:
                    ep_lengths[-1] += 1
                    ep_rewards[-1] += r
                else:
                    done = monitor.needs_reset
                    if done:
                        ep_lengths.append(monitor.episode_lengths[-1])
                        ep_rewards.append(monitor.episode_rewards[-1])

        outs = {
            'episode_lengths': ep_lengths,
            'episode_rewards': ep_rewards,
            'mean_length': np.mean(ep_lengths),
            'mean_reward': np.mean(ep_rewards),
        }

        os.makedirs(os.path.join(self.logdir, 'eval'), exist_ok=True)
        with open(os.path.join(self.logdir, f'eval/{self.t}.json'), 'w') as f:
            json.dump(outs, f)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
        logger.reset()






import unittest, shutil
from dl.util import atari_env, load_gin_configs
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

class TestPPO(unittest.TestCase):
    def test_feed_forward_ppo(self):
        env_fn = lambda rank: atari_env('Pong', rank=rank)
        ppo = PPO('logs', env_fn, maxt=1000, eval=True, eval_nepisodes=1, eval_period=1000)
        ppo.train()
        shutil.rmtree('logs')


if __name__=='__main__':
    load_gin_configs(['../configs/ppo.gin'])
    unittest.main()
