"""Defines networks for SAC experiments."""
from dl.rl.modules import PolicyBase, ContinuousQFunctionBase, ValueFunctionBase
from dl.rl.modules import QFunction, ValueFunction, UnnormActionPolicy
from dl.modules import TanhDiagGaussian
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin


class FeedForwardPolicyBase(PolicyBase):
    """Policy network."""

    def build(self):
        """Build."""
        self.fc1 = nn.Linear(self.observation_space.shape[0], 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.dist = TanhDiagGaussian(32, self.action_space.shape[0])

    def forward(self, x):
        """Forward."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.dist(x)


class AppendActionFeedForwardQFBase(ContinuousQFunctionBase):
    """Q network."""

    def build(self):
        """Build."""
        nin = self.observation_space.shape[0] + self.action_space.shape[0]
        self.fc1 = nn.Linear(nin, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.qvalue = nn.Linear(32, 1)

    def forward(self, x, a):
        """Forward."""
        x = F.relu(self.fc1(torch.cat([x, a], dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.qvalue(x)


class FeedForwardVFBase(ValueFunctionBase):
    """Value network."""

    def build(self):
        """Build."""
        self.fc1 = nn.Linear(self.observation_space.shape[0], 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.value = nn.Linear(32, 1)

    def forward(self, x):
        """Forward."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value(x)


@gin.configurable
def policy_fn(env):
    """Create a policy network."""
    return UnnormActionPolicy(FeedForwardPolicyBase(env.observation_space,
                                                    env.action_space))


@gin.configurable
def qf_fn(env):
    """Create a qfunction network."""
    return QFunction(AppendActionFeedForwardQFBase(env.observation_space,
                                                   env.action_space))


@gin.configurable
def vf_fn(env):
    """Create a value function network."""
    return ValueFunction(FeedForwardVFBase(env.observation_space,
                                           env.action_space))
