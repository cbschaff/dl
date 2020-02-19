"""Define networks for PPO2 experiments."""
from dl.rl.modules import PolicyBase, ValueFunctionBase, Policy, ValueFunction
from dl.rl.util import conv_out_shape
from dl.modules import Categorical
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import numpy as np
import gin


class FeedForwardPolicyBase(PolicyBase):
    """Policy network."""

    def build(self):
        """Build."""
        self.fc1 = nn.Linear(self.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, 128)
        self.dist = Categorical(128, self.action_space.n)

    def forward(self, x):
        """Forward."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.dist(x)


class VFNet(ValueFunctionBase):
    """Value Function."""

    def build(self):
        """Build."""
        self.fc1 = nn.Linear(self.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, 128)
        self.vf = nn.Linear(128, 1)

    def forward(self, x):
        """Forward."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.vf(x)


@gin.configurable
def policy_fn(env):
    """Create a3c conv net policy."""
    return Policy(FeedForwardPolicyBase(env.observation_space, env.action_space))


@gin.configurable
def value_fn(env):
    """Create value function network."""
    return ValueFunction(VFNet(env.observation_space, env.action_space))
