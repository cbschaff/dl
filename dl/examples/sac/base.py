from dl.rl.modules import PolicyBase, ContinuousQFunctionBase, ValueFunctionBase
from dl.modules import TanhDiagGaussian
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin

@gin.configurable
class FeedForwardPolicyBase(PolicyBase):
    def build(self):
        self.fc1 = nn.Linear(self.observation_space.shape[0], 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.dist = TanhDiagGaussian(32, self.action_space.shape[0])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.dist(x)

@gin.configurable
class AppendActionFeedForwardQFBase(ContinuousQFunctionBase):
    def build(self):
        nin = self.observation_space.shape[0] + self.action_space.shape[0]
        self.fc1 = nn.Linear(nin, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.qvalue = nn.Linear(32, 1)

    def forward(self, x, a):
        x = F.relu(self.fc1(torch.cat([x,a], dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.qvalue(x)

@gin.configurable
class FeedForwardVFBase(ValueFunctionBase):
    def build(self):
        self.fc1 = nn.Linear(self.observation_space.shape[0], 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.value = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value(x)