from dl.rl.modules import ActorCriticBase
from dl.rl.util import conv_out_shape
from dl.modules import Categorical, MaskedLSTM, TimeAndBatchUnflattener
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gin

@gin.configurable
class NatureDQN(ActorCriticBase):
    """
    Deep network from https://www.nature.com/articles/nature14236
    """
    def build(self):
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        shape = self.observation_space.shape[1:]
        for c in [self.conv1, self.conv2, self.conv3]:
            shape = conv_out_shape(shape, c)
        self.nunits = 64 * np.prod(shape)
        self.fc = nn.Linear(self.nunits, 512)
        self.vf = nn.Linear(512, 1)
        self.dist = Categorical(512, self.action_space.n)

    def forward(self, x):
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(-1, self.nunits)))
        return self.dist(x), self.vf(x)


@gin.configurable
class NatureDQNRNN(ActorCriticBase):
    """
    Deep network from https://www.nature.com/articles/nature14236
    """
    def build(self):
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        shape = self.observation_space.shape[1:]
        for c in [self.conv1, self.conv2, self.conv3]:
            shape = conv_out_shape(shape, c)
        self.nunits = 64 * np.prod(shape)
        self.lstm = MaskedLSTM(self.nunits, 512, 1)
        self.tbf = TimeAndBatchUnflattener()
        self.vf = nn.Linear(512, 1)
        self.dist = Categorical(512, self.action_space.n)

    def forward(self, x, state_in=None, mask=None):
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.nunits)
        if state_in is None:
            x, state_out = self.lstm(self.tbf(x))
        else:
            x, state_out = self.lstm(self.tbf(x, state_in[0]), state_in)
        x = self.tbf.flatten(x)
        return self.dist(x), self.vf(x)
