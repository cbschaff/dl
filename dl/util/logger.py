"""
Change baselines logger to append to log files.
"""
from baselines.logger import *
from baselines.logger import configure as baselines_configure
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import torch
import os

def append_human_init(self, filename_or_file):
    if isinstance(filename_or_file, str):
        self.file = open(filename_or_file, 'at')
        self.own_file = True
    else:
        assert hasattr(filename_or_file, 'read'), 'expected file or str, got %s'%filename_or_file
        self.file = filename_or_file
        self.own_file = False

def append_json_init(self, filename):
    self.file = open(filename, 'at')

def append_csv_init(self, filename):
    self.file = open(filename, 'a+t')
    self.keys = []
    self.sep = ','

HumanOutputFormat.__init__ = append_human_init
JSONOutputFormat.__init__ = append_json_init
CSVOutputFormat.__init__ = append_csv_init



WRITER = None

def configure(logdir, format_strs=None, **kwargs):
    global WRITER
    WRITER = TBWriter(logdir, **kwargs)
    FLUSH = time.time()
    baselines_configure(logdir, format_strs)

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
        if not os.path.exists(fname) or overwrite:
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
