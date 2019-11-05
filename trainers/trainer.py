"""Base class for training DL models."""
import gin
import os
import time
from dl import Checkpointer, logger, rng
import torch


@gin.configurable(blacklist=['logdir'])
class Trainer(object):
    """Base class for training ML models.

    It implements checkpointing, seeding, saving/loading random states,
    and logging of config files.

    Subclasses should implement, the step, evaluate, state_dict,
    and load_state_dict methods of this class.

    The train method uses as special instance variable, self.t, to keep track
    of the current timestep. The step method should increment this variable.
    Args:
        logdir (str):
            The base directory for the training run.
        seed (int):
            The initial seed of this experiment.
        eval (bool):
            Whether or not to evaluate the model throughout training.
        eval_period (int):
            The period with which the model is evaluated.
        save_period (int):
            The period with which the model is saved.
        maxt (int):
            The maximum number of timesteps to train the model.
        maxseconds (float):
            The maximum amount of time to train the model.
        gpu (bool):
            Where training will take place.

    """

    def __init__(self,
                 logdir,
                 seed=0,
                 eval=False,
                 eval_period=None,
                 save_period=None,
                 maxt=None,
                 maxseconds=None,
                 gpu=True):
        """Init."""
        self.logdir = logdir
        self.ckptr = Checkpointer(os.path.join(self.logdir, 'ckpts'))
        self.eval = eval
        self.eval_period = eval_period
        self.save_period = save_period
        self.maxt = maxt
        self.seed = seed
        self.maxseconds = maxseconds
        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available()
                                   else "cpu")
        logger.configure(os.path.join(logdir, 'tb'))

        self.t = 0
        rng.seed(seed)

    def step(self):
        """Step the optimization. Implement in subclasses."""
        raise NotImplementedError

    def evaluate(self):
        """Evaluate model. Implement in subclasses."""
        raise NotImplementedError

    def state_dict(self):
        """State to save. Implement in subclasses."""
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        """Load state. Implement in subclasses."""
        raise NotImplementedError

    def save(self):
        """Save model. Implement in subclasses."""
        self._save(self.state_dict())

    def _save(self, state_dict):
        assert '_rng' not in state_dict, (
                "'_rng' key is used to save randomstates. "
                "Please change your key.")
        assert 't' not in state_dict, (
               "'t' key is used by base Trainer class."
               "Please change your key.")
        state_dict['_rng'] = rng.get_state()
        state_dict['t'] = self.t
        self.ckptr.save(state_dict, self.t)

    def load(self, t=None):
        """Load model."""
        self._load(self.ckptr.load(t))

    def _load(self, state_dict):
        rng.set_state(state_dict['_rng'])
        self.t = state_dict['t']
        self.load_state_dict(state_dict)

    def train(self):
        """Train loop."""
        config = gin.operative_config_str()
        logger.log("=================== CONFIG ===================")
        logger.log(config)
        with open(os.path.join(self.logdir, 'config.gin'), 'w') as f:
            f.write(config)
        self.time_start = time.monotonic()
        if len(self.ckptr.ckpts()) > 0:
            self.load()
        if self.t == 0:
            cstr = config.replace('\n', '  \n')
            cstr = cstr.replace('#', '\\#')
            logger.add_text('config', cstr, 0, time.time())
        if self.maxt and self.t > self.maxt:
            return
        if self.save_period:
            last_save = (self.t // self.save_period) * self.save_period
        if self.eval_period:
            last_eval = (self.t // self.eval_period) * self.eval_period

        try:
            while True:
                if self.maxt and self.t >= self.maxt:
                    break
                if self.maxseconds and \
                   time.monotonic() - self.time_start >= self.maxseconds:
                    break
                self.step()
                if self.save_period and \
                   (self.t - last_save) >= self.save_period:
                    self.save()
                    last_save = self.t
                if self.eval and (self.t - last_eval) >= self.eval_period:
                    self.evaluate()
                    last_eval = self.t
        except KeyboardInterrupt:
            logger.log("Caught Ctrl-C. Saving model and exiting...")
        self.save()
        logger.flush()
        logger.close()
        self.close()

    def close(self):
        """Close trainer."""
        pass


if __name__ == '__main__':

    import unittest
    import shutil
    import numpy as np

    class T(Trainer):
        """Dummy trainer."""

        def step(self):
            """Step."""
            self.t += 1

        def evaluate(self):
            """Eval."""
            assert self.t % self.eval_period == 0

        def state_dict(self):
            """State dict."""
            if self.maxt is None or self.t < self.maxt:
                if self.maxseconds is None:
                    assert self.t % self.save_period == 0
            return {}

        def load_state_dict(self, state_dict):
            """Load."""
            pass

    class TestTrainer(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            trainer = T('logs', eval=True, eval_period=50, save_period=100,
                        maxt=1000)
            trainer.train()
            shutil.rmtree('logs')
            trainer = T('logs', eval=True, eval_period=50, save_period=100,
                        maxseconds=2)
            trainer.train()
            c = trainer.ckptr.ckpts()
            trainer2 = T('logs', eval=True, eval_period=50, save_period=100,
                         maxseconds=1)
            trainer.train()
            c2 = trainer2.ckptr.ckpts()
            for x in c:
                assert x in c2
            assert len(c2) > len(c)
            shutil.rmtree('logs')

        def test_rng(self):
            """Test rng."""
            trainer = T('logs', seed=0, save_period=1)
            trainer.save()
            r1 = np.random.rand(10)
            trainer.load()
            r2 = np.random.rand(10)
            assert np.allclose(r1, r2)
            shutil.rmtree('logs')

    unittest.main()
