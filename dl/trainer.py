import gin, os, time
from dl.util import Checkpointer, logger, rng

@gin.configurable(blacklist=['logdir'])
class Trainer(object):
    """
    Base class for training ML models. Subclasses should implement, the step,
    evaluate, state_dict, and load_state_dict methods of this class.
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
    """
    def __init__(self, logdir, seed=0, eval=False, eval_period=None, save_period=None, maxt=None, maxseconds=None):
        self.logdir = logdir
        self.ckptr = Checkpointer(os.path.join(self.logdir, 'ckpts'))
        self.eval = eval
        self.eval_period = eval_period
        self.save_period = save_period
        self.maxt = maxt
        self.seed = seed
        self.maxseconds = maxseconds
        logger.configure(logdir)

        self.t = 0
        rng.seed(seed)

    def step(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self):
        raise NotImplementedError

    def save(self):
        self.ckptr.save(self.state_dict(), self.t)

    def load(self, t=None):
        self.load_state_dict(self.ckptr.load(t))

    def train(self):
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
                if self.maxseconds and time.monotonic() - self.time_start >= self.maxseconds:
                    break
                self.step()
                if self.save_period and (self.t - last_save) >= self.save_period:
                    self.save()
                    last_save = self.t
                if self.eval and (self.t - last_eval) >= self.eval_period:
                    self.evaluate()
                    last_eval = self.t
        except KeyboardInterrupt:
            logger.log("Caught Ctrl-C. Saving model and exiting...")
        self.save()
        logger.export_scalars(self.ckptr.format.format(self.t) + '.json')
        self.close()

    def close(self):
        pass



import unittest,shutil

class TestTrainer(unittest.TestCase):
    def test(self):
        class T(Trainer):
            def step(self):
                self.t += 1
            def evaluate(self):
                assert self.t % self.eval_period == 0
            def state_dict(self):
                if self.maxt is None or self.t < self.maxt:
                    if self.maxseconds is None:
                        assert self.t % self.save_period == 0
                return {'t': self.t}
            def load_state_dict(self, state_dict):
                self.t = state_dict['t']

        trainer = T('logs', eval=True, eval_period=50, save_period=100, maxt=1000)
        trainer.train()
        shutil.rmtree('logs')
        trainer = T('logs', eval=True, eval_period=50, save_period=100, maxseconds=2)
        trainer.train()
        c = trainer.ckptr.ckpts()
        trainer2 = T('logs', eval=True, eval_period=50, save_period=100, maxseconds=1)
        trainer.train()
        c2 = trainer2.ckptr.ckpts()
        for x in c:
            assert x in c2
        assert len(c2) > len(c)
        shutil.rmtree('logs')


if __name__ == '__main__':
    unittest.main()
