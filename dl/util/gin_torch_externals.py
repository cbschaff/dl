import gin
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import inspect

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
