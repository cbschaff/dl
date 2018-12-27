"""
Change baselines Monitor to append to log files.
"""
from baselines import bench
from baselines.bench.monitor import ResultsWriter
from gym.core import Wrapper
import time, json, csv
import os.path as osp


class AppendResultsWriter(ResultsWriter):
    def __init__(self, filename=None, header='', extra_keys=()):
        self.extra_keys = extra_keys
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if osp.isdir(filename):
                    filename = osp.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.f = open(filename, "wt")
            if isinstance(header, dict):
                header = '# {} \n'.format(json.dumps(header))
            self.f.write(header)
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+tuple(extra_keys))
            self.logger.writeheader()
            self.f.flush()

            if osp.exists(filename):
                with open(filename, 'rt') as f:
                    info = json.loads(f.readline()[1:])
                    print(info)
                    self.tstart = info['t_start']
                self.f = open(filename, 'at')
                self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+tuple(extra_keys))
            else:
                self.f = open(filename, "wt")
                if isinstance(header, dict):
                    self.tstart = header['t_start']
                    header = '# {} \n'.format(json.dumps(header))
                else:
                    self.tstart = time.time()
                self.f.write(header)
                self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+tuple(extra_keys))
                self.logger.writeheader()
                self.f.flush()

class Monitor(bench.Monitor):
    """
    Changing baselines.bench.Monitor to append to an existing log file.
    This allows for easier starting and stopping of experiments.
    """
    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=(), info_keywords=()):
        Wrapper.__init__(self, env=env)
        self.results_writer = AppendResultsWriter(
            filename,
            header={"t_start": time.time(), 'env_id' : env.spec and env.spec.id},
            extra_keys=reset_keywords + info_keywords
        )
        self.tstart = self.results_writer.tstart
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()
