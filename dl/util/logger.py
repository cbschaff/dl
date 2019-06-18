"""
Change baselines logger to append to log files.
"""
from torch.utils.tensorboard import SummaryWriter
import torch
import os, time, json


WRITER = None

def configure(logdir, **kwargs):
    global WRITER, LOGDIR
    WRITER = TBWriter(logdir, **kwargs)

def get_summary_writer():
    return WRITER

class TBWriter(SummaryWriter):
    def __init__(self, logdir, *args, **kwargs):
        super().__init__(logdir, logdir, *args, **kwargs)
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
        # change interface so both add_scalar and add_scalars adds to the scalar dict.
        self._append_to_scalar_dict(tag, scalar_value, global_step, walltime)
        self.flush(force=False)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        super().add_scalars(main_tag, tag_scalar_dict, global_step, walltime)
        # edit scalar_dict
        fw_logdir = self._get_file_writer().get_logdir()
        for tag,value in tag_scalar_dict.items():
            fwtag = fw_logdir + "/" + main_tag + "/" + tag
            new_tag = main_tag + "/" + tag
            del self.scalar_dict[fwtag]
            self._append_to_scalar_dict(new_tag, value, global_step, walltime)


    def _append_to_scalar_dict(self, tag, scalar_value, global_step, walltime):
        if not tag in self.scalar_dict:
            self.scalar_dict[tag] = {'value': [], 'step': [], 'time': []}
        self.scalar_dict[tag]['value'].append(scalar_value)
        self.scalar_dict[tag]['step'].append(global_step)
        self.scalar_dict[tag]['time'].append(walltime)

    def export_scalars(self, fname, overwrite=False):
        os.makedirs(os.path.join(self.log_dir, 'scalar_data'), exist_ok=True)
        if fname[-4:] != 'json':
            fname += '.json'
        fname = os.path.join(self.log_dir, 'scalar_data', fname)
        i = 1
        while not overwrite and os.path.exists(fname):
            fname = fname.rsplit('.',1)[0] + f'_{i}.json'
            i += 1
        with open(fname, 'w') as f:
            json.dump(self.scalar_dict, f)

        self.flush(force=True)
        self.scalar_dict = {}


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

def export_scalars(fname, overwrite=False):
    WRITER.export_scalars(fname, overwrite=overwrite)


def log(out):
    print(out)

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
