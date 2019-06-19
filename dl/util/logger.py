"""
Global Tensorboard writer.
"""
from torch.utils.tensorboard import SummaryWriter
import torch
import os, time, json

def log(out):
    print(out)

WRITER = None

def configure(logdir, **kwargs):
    global WRITER, LOGDIR
    WRITER = TBWriter(logdir, **kwargs)

def get_summary_writer():
    return WRITER

def get_dir():
    return None if WRITER is None else WRITER.log_dir


class TBWriter(SummaryWriter):
    def __init__(self, logdir, *args, **kwargs):
        super().__init__(logdir, *args, **kwargs)
        self.last_flush = time.time()
        self.scalar_dict = {}

    def _unnumpy(self, x):
        """
        Numpy data types are not json serializable.
        """
        if hasattr(x, 'tolist'):
            return x.tolist()
        return x

    def _scalarize(self, x):
        """
        Turn into scalar
        """
        x = self._unnumpy(x)
        if isinstance(x, list):
            if len(x) == 1:
                return self._scalarize(x[0])
            else:
                assert False, "Tried to log something that isn't a scalar!"
        return x

    def flush(self, force=True):
        if time.time() - self.last_flush > 60 or force:
            for writer in self.all_writers.values():
                writer.flush()
            self.last_flush = time.time()

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        scalar_value = self._scalarize(scalar_value)
        global_step  = self._scalarize(global_step)
        walltime     = self._scalarize(walltime)
        super().add_scalar(tag, scalar_value, global_step, walltime)
        self.flush(force=False)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        for k,v in tag_scalar_dict.items():
            tag_scalar_dict[k] = self._scalarize(v)
        global_step  = self._scalarize(global_step)
        walltime     = self._scalarize(walltime)
        super().add_scalars(main_tag, tag_scalar_dict, global_step, walltime)
        self.flush(force=False)


    def _append_to_scalar_dict(self, tag, scalar_value, global_step, walltime):
        """
        Disable scalar dict. Data will be fetched from tb logs when needed.
        """
        pass


def add_scalar(*args, **kwargs):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_scalar(*args, **kwargs)

def add_scalars(*args, **kwargs):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_scalars(*args, **kwargs)

def add_histogram(*args, **kwargs):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_histogram(*args, **kwargs)
    WRITER.flush(force=True)

def add_image(*args, **kwargs):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_image(*args, **kwargs)
    WRITER.flush(force=True)

def add_figure(*args, **kwargs):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_figure(*args, **kwargs)
    WRITER.flush(force=True)

def add_video(*args, **kwargs):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_video(*args, **kwargs)
    WRITER.flush(force=True)

def add_audio(*args, **kwargs):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_audio(*args, **kwargs)
    WRITER.flush(force=True)

def add_text(*args, **kwargs):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_text(*args, **kwargs)
    WRITER.flush(force=True)

def add_graph(*args, **kwargs):
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.add_graph(*args, **kwargs)
    WRITER.flush(force=True)

def flush():
    assert WRITER is not None, "call configure to initialize SummaryWriter"
    WRITER.flush(force=True)

def close():
    global WRITER
    if WRITER is not None:
        WRITER.flush(force=True)
        WRITER.close()
        WRITER = None


######################
# Decorators
######################

class LogOutputs(object):
    def __init__(self, f, log_freq=1, name=None, global_step_fn=None):
        self.f = f
        self.name = f.__name__ if name is None else name
        self.log_freq = log_freq
        self.count = 0
        self.global_step_fn = global_step_fn

    def __call__(self, *args, **kwargs):
        out = self.f(*args, **kwargs)
        self.count += 1
        if self.count % self.log_freq == 0:
            self.log(out)
        return out


    def log(self, out):
        raise NotImplementedError

class PrintOutputs(LogOutputs):
    def log(self, out):
        print('==================================')
        print(f'Output of {self.name}:')
        print(out)
        print('==================================')

class TBLogOutputs(LogOutputs):
    def log(self, out):
        if WRITER is None:
            return
        if self.global_step_fn:
            t = self.global_step_fn()
        else:
            t = self.count
        _tblog(self, out, t)

    def _tblog(self, out, global_step):
        raise NotImplementedError

class TBLogScalar(TBLogOutputs):
    def _tblog(self, out, global_step):
        add_scalar(self.name, out, global_step, time.time())

class TBLogScalarDict(TBLogOutputs):
    def __init__(self, *args, group=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.group = group

    def _tblog(self, out, global_step):
        assert isinstance(out, dict)
        if self.group:
            add_scalars(self.name, out, t, time.time())
        else:
            for k in out:
                add_scalar(self.name + f'/{k}', out[k], t, time.time())

class TBLogText(TBLogOutputs):
    def _tblog(self, out, global_step):
        add_text(self.name, out, global_step, time.time())

class TBLogImage(TBLogOutputs):
    def _tblog(self, out, global_step):
        add_image(self.name, out, global_step, time.time())
