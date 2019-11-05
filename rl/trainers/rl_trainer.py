"""Extends the Trainer class with environment utils."""
from dl import Trainer
from dl import logger
from dl.rl.envs import VecEpisodeLogger, VecObsNormWrapper
from dl.rl.util import rl_evaluate, rl_record, misc
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
    - Implement Trainer.step (update model) and Trainer.evaluate.
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
            env = SubprocVecEnv([_env(i) for i in range(self.nenv)],
                                context='fork')
        else:
            env = DummyVecEnv([_env(0)])
        env = VecObsNormWrapper(env, norm=self.norm_observations)
        return VecEpisodeLogger(env)

    def _save(self, state_dict):
        state_dict['env'] = misc.env_state_dict(self.env)
        super()._save(state_dict)

    def _load(self, state_dict):
        misc.env_load_state_dict(self.env, state_dict['env'])
        super()._load(state_dict)

    def rl_evaluate(self, eval_env, eval_actor):
        """Log episode stats."""
        eval_actor.eval()
        misc.set_env_to_eval_mode(eval_env)
        os.makedirs(os.path.join(self.logdir, 'eval'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'eval',
                               self.ckptr.format.format(self.t) + '.json')
        stats = rl_evaluate(eval_env, eval_actor, self.eval_num_episodes,
                            outfile, self.device)
        logger.add_scalar('eval/mean_episode_reward', stats['mean_reward'],
                          self.t, time.time())
        logger.add_scalar('eval/mean_episode_length', stats['mean_length'],
                          self.t, time.time())
        eval_actor.train()
        misc.set_env_to_train_mode(eval_env)

    def rl_record(self, eval_env, eval_actor):
        """Record videos."""
        eval_actor.eval()
        misc.set_env_to_eval_mode(eval_env)
        os.makedirs(os.path.join(self.logdir, 'video'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'video',
                               self.ckptr.format.format(self.t) + '.mp4')
        rl_record(eval_env, eval_actor, self.record_num_episodes, outfile,
                  self.device)
        eval_actor.train()
        misc.set_env_to_train_mode(eval_env)

    def close(self):
        """Close environment."""
        try:
            self.env.close()
        except Exception:
            pass
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
                            self.nenv = self.env.num_envs

                        def __call__(self, ob):
                            ac = torch.from_numpy(
                                np.array([self.env.action_space.sample()
                                          for _ in range(self.nenv)]))
                            return namedtuple('test', ['action', 'state_out'])(
                                action=ac, state_out=None)

                        def train(self):
                            pass

                        def eval(self):
                            pass
                    actor = Actor(self.env)

                    self.rl_evaluate(self.env, actor)
                    self.rl_record(self.env, actor)

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
