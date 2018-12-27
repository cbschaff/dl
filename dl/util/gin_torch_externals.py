import gin
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import inspect

def load_gin_configs(gin_files, gin_bindings=[]):
    """Loads gin configuration files.
    Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
    """
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False)

optimizers = [obj for name,obj in inspect.getmembers(optim) if inspect.isclass(obj)]
for o in optimizers:
    gin.config.external_configurable(o, module='optim')

modules = [obj for name,obj in inspect.getmembers(nn) if inspect.isclass(obj)]
for m in modules:
    gin.config.external_configurable(m, module='nn')

funcs = [f for name,f in inspect.getmembers(F) if inspect.isfunction(f)]
for f in funcs:
    try:
        gin.config.external_configurable(f, module='F')
    except:
        pass
