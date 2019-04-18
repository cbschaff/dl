
"""
PPO RL algorithm.
https://arxiv.org/abs/1707.06347
"""
from dl import Trainer
from dl.modules import Policy
from dl.util import RolloutStorage
from dl.util import logger, find_monitor, VecMonitor
from dl.eval import rl_evaluate, rl_record, rl_plot
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
                 policy=Policy,
                 batch_size=32,
                 steps_per_iter=128,
                 epochs_per_iter=10,
                 max_grad_norm=None,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 clip_param=0.2,
                 gamma=0.99,
                 lambda_=1.0,
                 norm_observations=True,
                 norm_advantages=True,
                 eval_nepisodes=100,
                 gpu=True,
                 **trainer_kwargs
    ):
        super().__init__(logdir, **trainer_kwargs)
        self.env = self._make_env(env_fn, nenv)
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

        self.net = policy(self.env.observation_space.shape, self.env.action_space,  norm_observations=norm_observations)
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
            self.recurrent = True
            self.rollout = RolloutStorage(steps_per_iter, self.nenv, device=self.device, other_keys=['logp'], recurrent_state_keys=self.recurrent_keys)
            self.init_state = []
            for i,state in enumerate(self.recurrent_state_size):
                self.init_state.append(torch.zeros(size=[self.nenv] + list(state.shape), device=self.device))

        self.t, self.t_start = 0,0
        self.losses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}
        self.meanlosses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}

        self._ob = torch.from_numpy(self.env.reset()).to(self.device)
        self._mask = torch.Tensor([0. for _ in range(self.nenv)]).to(self.device)
        self._state = self.init_state

    def _make_env(self, env_fn, nenv):
        def _env(rank):
            def _thunk():
                return env_fn(rank=rank)
            return _thunk
        if nenv > 1:
            env = SubprocVecEnv([_env(i) for i in range(nenv)])
        else:
            env = DummyVecEnv([_env(0)])
        tstart = max(self.ckptr.ckpts()) if len(self.ckptr.ckpts()) > 0 else 0
        return VecMonitor(env, max_history=100, tstart=tstart, tbX=True)

    def state_dict(self):
        return {
            'net': self.net.state_dict(),
            'opt': self.opt.state_dict(),
            't':   self.t,
        }

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
        self._mask = torch.Tensor([0.0 if done_ else 1.0 for done_ in done]).to(self.device)
        self._state = outs.state_out
        self.t += self.nenv

    def loss(self, batch):
        if self.recurrent:
            state = [batch[k] for k in self.recurrent_keys]
            outs = self.net(batch['ob'], mask=batch['mask'], state_in=state)
        else:
            outs = self.net(batch['ob'])

        # compute policy loss
        logp = outs.dist.log_prob(batch['ac'])
        assert logp.shape == batch['logp'].shape
        ratio = torch.exp(logp - batch['logp'])
        assert ratio.shape == batch['atarg'].shape
        ploss1 = ratio * batch['atarg']
        ploss2 = torch.clamp(ratio, 1.0-self.clip_param, 1.0+self.clip_param) * batch['atarg']
        pi_loss = -torch.min(ploss1, ploss2).mean()
        self.losses['pi'].append(pi_loss)

        # compute value loss
        criterion = torch.nn.MSELoss(reduction='none')
        vloss1 = 0.5 * criterion(outs.value, batch['vtarg'])
        vpred_clipped = batch['vpred'] + (outs.value - batch['vpred']).clamp(-self.clip_param, self.clip_param)
        vloss2 = 0.5 * criterion(vpred_clipped, batch['vtarg'])
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
            with torch.no_grad():
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
            logger.add_scalar('loss/total',   np.mean(self.meanlosses['tot']), self.t, time.time())
            logger.add_scalar('loss/policy',  np.mean(self.meanlosses['pi']), self.t, time.time())
            logger.add_scalar('loss/value',   np.mean(self.meanlosses['value']), self.t, time.time())
            logger.add_scalar('loss/entropy', np.mean(self.meanlosses['ent']), self.t, time.time())
        self.meanlosses = {'tot':[], 'pi':[], 'value':[], 'ent':[]}
        # Logging stats...
        logger.logkv('timesteps', self.t)
        logger.logkv('fps', int((self.t - self.t_start) / (time.monotonic() - self.time_start)))
        logger.logkv('time_elapsed', time.monotonic() - self.time_start)

        logger.logkv('mean episode length', np.mean(self.env.episode_lengths))
        logger.logkv('mean episode reward', np.mean(self.env.episode_rewards))
        vmax = torch.max(self.rollout.data['vpred']).cpu().numpy()
        vmean = torch.mean(self.rollout.data['vpred']).cpu().numpy()
        logger.add_scalar('alg/v_max', float(vmax), self.t, time.time())
        logger.add_scalar('alg/v_mean', float(vmean), self.t, time.time())
        logger.dumpkvs()

    def evaluate(self):
        self.net.train(False)
        # create new env to access true reward function and episode lenghts from the Monitor wrapper (if it exists)
        eval_env = self.env_fn(rank=self.nenv+1)

        os.makedirs(os.path.join(self.logdir, 'eval'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'eval', self.ckptr.format.format(self.t) + '.json')
        stats = rl_evaluate(eval_env, self.net, self.eval_nepisodes, outfile, self.device)
        logger.add_scalar('eval/mean_episode_reward', float(stats['mean_reward']), self.t, time.time())
        logger.add_scalar('eval/mean_episode_length', float(stats['mean_length']), self.t, time.time())

        os.makedirs(os.path.join(self.logdir, 'video'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'video', self.ckptr.format.format(self.t) + '.mp4')
        rl_record(eval_env, self.net, 5, outfile, self.device)

        if find_monitor(eval_env):
            rl_plot(os.path.join(self.logdir, 'logs'), eval_env.spec.id, self.t)
        self.net.train(True)

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
    load_gin_configs(['../configs/ppo.gin'], ['PPO.gpu=False'])
    unittest.main()
