"""Extends the Trainer class with environment utils."""
from dl import Trainer
from dl import logger
from dl.rl.envs import VecEpisodeLogger, ObsNormWrapper, VecObsNormWrapper
from dl.rl.util import rl_evaluate, rl_record, find_wrapper, is_vec_env
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gin
import os
import time


@gin.configurable(blacklist=['logdir'])
class RLTrainer(Trainer):
    """Extends Trainer with basic functionality for Reinforcement Learning.

    The resposibilities of this class are:
    - Handle environment creation
    - Observation normralization
    - Provide evaluation code
    - Log episode stats to tensorboard

    Subclasses will need to:
    - Handle environment stepping and data.
    - Create/handle models and their predictions.
    - Implement Trainer.step (update model).
    """

    def __init__(self,
                 logdir,
                 env_fn,
                 nenv=1,
                 norm_observations=False,
                 eval_num_episodes=1,
                 record_num_episodes=1,
                 **kwargs):
        """Init."""
        super().__init__(logdir, **kwargs)
        self.env_fn = env_fn
        self.nenv = nenv
        self.norm_observations = norm_observations
        self.eval_num_episodes = eval_num_episodes
        self.record_num_episodes = record_num_episodes
        self.env = self.make_training_env()

    def make_training_env(self):
        """Create a training environment."""
        def _env(rank):
            def _thunk():
                return self.env_fn(rank=rank)
            return _thunk
        if self.nenv > 1:
            env = SubprocVecEnv([_env(i) for i in range(self.nenv)])
        else:
            env = DummyVecEnv([_env(0)])
        env = self.add_obs_norm_wrapper(env)
        return VecEpisodeLogger(env)

    def make_eval_env(self):
        """Environment used for evaluation.

        Cannot be a VecEnv.
        """
        eval_env = self._make_eval_env()
        assert not is_vec_env(eval_env)
        return self.add_obs_norm_wrapper(eval_env)

    def _make_eval_env(self):
        """Create an environment.

        Overwrite in subclasses if needed.
        """
        return self.env_fn(rank=self.nenv + 1)

    def add_obs_norm_wrapper(self, env):
        """Wrap an env to normalize observations.

        If env is different from self.env, then the normalization used by
        self.env will be used.

        Otherwise normalization parameters will be used from the latest ckpt
        or will be computed from rollouts in the environemnt.
        """
        if is_vec_env(env):
            wrapper_class = VecObsNormWrapper
        else:
            wrapper_class = ObsNormWrapper
        assert find_wrapper(env, wrapper_class) is None, (
            "Environment already has an ObsNormWrapper.")
        is_training_env = not hasattr(self, 'env')
        if not self.norm_observations:
            env = wrapper_class(env, norm=False, log=is_training_env)
        elif not is_training_env:
            ob_wrapper = find_wrapper(self.env, VecObsNormWrapper)
            env = wrapper_class(env,
                                mean=ob_wrapper.mean,
                                std=ob_wrapper.std,
                                log=False)
        else:
            ckpts = self.ckptr.ckpts()
            if ckpts:
                # Load in normalization parameters here. This can result in
                # loading the ckpt twice, but guarantees the correct
                # normalization is used.
                state_dict = self.ckptr.load(max(ckpts))
                env = wrapper_class(env, find_norm_params=False)
                env.load_state_dict(state_dict['obs_norm'])
            else:
                env = wrapper_class(env, find_norm_params=True)
        return env

    def _save(self, state_dict):
        ob_wrapper = find_wrapper(self.env, VecObsNormWrapper)
        if ob_wrapper:
            state_dict['obs_norm'] = ob_wrapper.state_dict()
        super()._save(state_dict)

    def _load(self, state_dict):
        ob_wrapper = find_wrapper(self.env, VecObsNormWrapper)
        if ob_wrapper:
            ob_wrapper.load_state_dict(state_dict['obs_norm'])
        log_wrapper = find_wrapper(self.env, VecEpisodeLogger)
        if log_wrapper:
            log_wrapper.t = state_dict['t']
        super()._load(state_dict)

    def rl_evaluate(self, eval_actor):
        """Log episode stats."""
        if not hasattr(self, 'eval_env'):
            self.eval_env = self.make_eval_env()
        eval_actor.eval()
        os.makedirs(os.path.join(self.logdir, 'eval'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'eval',
                               self.ckptr.format.format(self.t) + '.json')
        stats = rl_evaluate(self.eval_env, eval_actor, self.eval_num_episodes,
                            outfile, self.device)
        logger.add_scalar('eval/mean_episode_reward', stats['mean_reward'],
                          self.t, time.time())
        logger.add_scalar('eval/mean_episode_length', stats['mean_length'],
                          self.t, time.time())
        eval_actor.train()

    def rl_record(self, eval_actor):
        """Record videos."""
        if not hasattr(self, 'eval_env'):
            self.eval_env = self.make_eval_env()
        eval_actor.eval()
        os.makedirs(os.path.join(self.logdir, 'video'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'video',
                               self.ckptr.format.format(self.t) + '.mp4')
        rl_record(self.eval_env, eval_actor, self.record_num_episodes, outfile,
                  self.device)
        eval_actor.train()

    def close(self):
        """Close environment."""
        self.env.close()
        super().close()


if __name__ == '__main__':
    import unittest
    import shutil
    import gym
    import torch
    import numpy as np
    from collections import namedtuple

    class TestRLTrainer(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            class T(RLTrainer):
                def step(self):
                    if self.t == 0:
                        self.env.reset()
                    self.t += self.nenv
                    self.env.step([self.env.action_space.sample()
                                   for _ in range(self.nenv)])

                def evaluate(self):
                    class Actor():
                        def __init__(self, env):
                            self.env = env

                        def __call__(self, ob):
                            ac = torch.from_numpy(
                                np.array(self.env.action_space.sample()))[None]
                            return namedtuple('test', ['action', 'state_out'])(
                                action=ac, state_out=None)

                        def train(self):
                            pass

                        def eval(self):
                            pass
                    actor = Actor(self.env)

                    self.rl_evaluate(actor)
                    self.rl_record(actor)

                def state_dict(self):
                    return {}

                def load_state_dict(self, state_dict):
                    pass

            def env_fn(rank=0):
                env = gym.make('PongNoFrameskip-v4')
                env.seed(rank)
                return env
            t = T('./.test', env_fn, nenv=2, eval=True, eval_period=5000,
                  maxt=10000)
            t.train()
            shutil.rmtree('./.test')

    unittest.main()
