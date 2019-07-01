from dl.rl.modules import ActorCriticBase
from dl.rl.util import conv_out_shape
from dl.modules import Categorical, MaskedLSTM, TimeAndBatchUnflattener
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gin


@gin.configurable
class A3CCNN(ActorCriticBase):
    """
    Deep network from https://arxiv.org/abs/1602.01783
    """
    def build(self):
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        shape = self.observation_space.shape[1:]
        for c in [self.conv1, self.conv2]:
            shape = conv_out_shape(shape, c)
        self.nunits = 32 * np.prod(shape)
        self.fc = nn.Linear(self.nunits, 256)
        self.vf = nn.Linear(256, 1)
        self.dist = Categorical(256, self.action_space.n)
        nn.init.orthogonal_(self.vf.weight.data, gain=1.0)
        nn.init.constant_(self.vf.bias.data, 0)

    def forward(self, x):
        x = (x.float() / 128.) - 1.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc(x.view(-1, self.nunits)))
        return self.dist(x), self.vf(x)


@gin.configurable
class A3CRNN(ActorCriticBase):
    """
    Deep network from https://arxiv.org/abs/1602.01783
    """
    def build(self):
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        shape = self.observation_space.shape[1:]
        for c in [self.conv1, self.conv2]:
            shape = conv_out_shape(shape, c)
        self.nunits = 32 * np.prod(shape)
        self.fc = nn.Linear(self.nunits, 256)
        self.lstm = MaskedLSTM(256, 256, 1)
        self.tbf = TimeAndBatchUnflattener()
        self.vf = nn.Linear(256, 1)
        self.dist = Categorical(256, self.action_space.n)
        nn.init.orthogonal_(self.vf.weight.data, gain=1.0)
        nn.init.constant_(self.vf.bias.data, 0)

    def forward(self, x, state_in=None, mask=None):
        x = (x.float() / 128.) - 1.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc(x.view(-1, self.nunits)))
        if state_in is None:
            x, state_out = self.lstm(self.tbf(x))
        else:
            x, state_out = self.lstm(self.tbf(x, state_in[0]), state_in)
        x = self.tbf.flatten(x)
        return self.dist(x), self.vf(x)
