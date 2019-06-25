from dl import Trainer, logger
from dl.rl.envs import VecEpisodeLogger
from dl.rl.util import rl_evaluate, rl_record
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gin, os, time

@gin.configurable(blacklist=['logdir'])
class RLTrainer(Trainer):
    """
    Extends Trainer with basic functionality for Reinforcement Learning.
    The resposibilities of this class are:
        - Handle environment creation
        - Provide evaluation code
        - Log episode stats to tensorboard
    """
    def __init__(self,
                 logdir,
                 env_fn,
                 nenv=1,
                 eval_num_episodes=1,
                 record_num_episodes=1,
                 **kwargs
                ):
        super().__init__(logdir, **kwargs)
        self.env_fn = env_fn
        self.nenv = nenv
        self.eval_num_episodes = eval_num_episodes
        self.record_num_episodes = record_num_episodes
        self.env = self.make_env()

    def make_env(self):
        def _env(rank):
            def _thunk():
                return self.env_fn(rank=rank)
            return _thunk
        if self.nenv > 1:
            env = SubprocVecEnv([_env(i) for i in range(self.nenv)])
        else:
            env = DummyVecEnv([_env(0)])
        return VecEpisodeLogger(env)

    def _load(self, state_dict):
        self.env.t = state_dict['t']
        super()._load(state_dict)


    def rl_evaluate(self, eval_env, eval_actor):
        """
        Logs episode stats.
        """
        eval_actor.eval()
        os.makedirs(os.path.join(self.logdir, 'eval'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'eval', self.ckptr.format.format(self.t) + '.json')
        stats = rl_evaluate(eval_env, eval_actor, self.eval_num_episodes, outfile, self.device)
        logger.add_scalar('eval/mean_episode_reward', stats['mean_reward'], self.t, time.time())
        logger.add_scalar('eval/mean_episode_length', stats['mean_length'], self.t, time.time())
        eval_actor.train()

    def rl_record(self, eval_env, eval_actor):
        """
        Records videos.
        """
        eval_actor.eval()
        os.makedirs(os.path.join(self.logdir, 'video'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'video', self.ckptr.format.format(self.t) + '.mp4')
        rl_record(eval_env, eval_actor, self.record_num_episodes, outfile, self.device)
        eval_actor.train()

    def close(self):
        self.env.close()
        super().close()


if __name__ == '__main__':
    import unittest, shutil
    import gym, torch
    import numpy as np
    from collections import namedtuple

    class TestRLTrainer(unittest.TestCase):

        def test(self):
            class T(RLTrainer):
                def step(self):
                    self.t += 1

                def evaluate(self):
                    class Actor():
                        def __init__(self, env):
                            self.env = env
                        def __call__(self, ob):
                            ac = torch.from_numpy(np.array(self.env.action_space.sample()))[None]
                            return namedtuple('test','action')(action=ac)
                        def train(self):
                            pass
                        def eval(self):
                            pass
                    actor = Actor(self.env)

                    self.rl_evaluate(self.env, actor)
                    self.rl_record(self.env.venv.envs[0], actor)

                def state_dict(self):
                    return {}

            def env_fn(rank=0):
                env = gym.make('PongNoFrameskip-v4')
                env.seed(rank)
                return env
            t = T('./.test', env_fn, nenv=1, eval=True, eval_period=1, maxt=3)
            t.train()
            shutil.rmtree('./.test')

    unittest.main()
