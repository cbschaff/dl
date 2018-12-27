import torch
import torch.nn as nn
import torch.distributions as D

"""
Modules which map from feature vectors to torch distributions.

Standardize the inferface among distributions.
"""

class CatDist(D.Categorical):
    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_prob(self, ac):
        return super().log_prob(ac.squeeze(-1))

class Normal(D.Normal):
    def mode(self):
        return self.mean

    def log_prob(self, ac):
        return super().log_prob(ac).sum(-1)


class Categorical(nn.Module):
    """
    Categorical distribution. Unnormalized logits are parameterized
    as a linear function of the features.
    """
    def __init__(self, nin, nout):
        """
        Args:
            nin  (int): dimensionality of the input
            nout (int): number of categories
        """
        super().__init__()

        self.linear = nn.Linear(nin, nout)
        nn.init.orthogonal_(self.linear.weight.data, gain=0.01)
        nn.init.constant_(self.linear.bias.data, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): vectors of length nin
        Returns:
            dist (torch.distributions.Categorical): Categorical distribution
        """
        x = self.linear(x)
        return CatDist(logits=x)


class DiagGaussian(nn.Module):
    """
    Diagonal Gaussian distribution. Mean is parameterized as a linear function
    of the features. logstd is torch.Parameter.
    """
    def __init__(self, nin, nout):
        """
        Args:
            nin  (int): dimensionality of the input
            nout (int): number of categories
        """
        super().__init__()

        self.linear = nn.Linear(nin, nout)
        self.logstd = nn.Parameter(torch.zeros(nout))
        nn.init.orthogonal_(self.linear.weight.data, gain=1.0)
        nn.init.constant_(self.linear.bias.data, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): vectors of length nin
        Returns:
            dist (torch.distributions.Normal): normal distribution
        """
        mean = self.linear(x)
        return Normal(mean, self.logstd.exp())



import unittest

class TestDistributions(unittest.TestCase):
    def test_categorical(self):
        cat = Categorical(10, 2)
        features = torch.ones(2,10)
        dist = cat(features)
        ac = dist.sample()
        assert ac.shape == (2, 1)
        assert torch.all(dist.mode()[0] == dist.mode()[1])
        assert dist.mode().shape == (2, 1)
        assert dist.log_prob(ac).shape == (2,)

    def test_normal(self):
        dg = DiagGaussian(10, 2)
        features = torch.ones(2,10)
        dist = dg(features)
        ac = dist.sample()
        assert ac.shape == (2, 2)
        assert torch.all(dist.mode()[0] == dist.mode()[1])
        assert dist.mode().shape == (2, 2)
        assert dist.log_prob(ac).shape == (2,)


if __name__ == '__main__':
    unittest.main()
