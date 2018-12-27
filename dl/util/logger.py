"""
Change baselines logger to append to log files.
"""
from baselines.logger import *

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
