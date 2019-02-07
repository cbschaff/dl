import torch
import torch.nn as nn
import torch.distributions as D
import gin

"""
Modules which map from feature vectors to torch distributions.

Standardize the inferface among distributions.
"""

class CatDist(D.Categorical):
    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def sample(self, *args, **kwargs):
        return super().sample(*args, **kwargs).unsqueeze(-1)

    def log_prob(self, ac):
        return super().log_prob(ac.squeeze(-1))

class Normal(D.Normal):
    def mode(self):
        return self.mean

    def log_prob(self, ac):
        return super().log_prob(ac).sum(-1)

    def entropy(self):
        return super().entropy().sum(-1)


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


@gin.configurable
class DiagGaussian(nn.Module):
    """
    Diagonal Gaussian distribution. Mean is parameterized as a linear function
    of the features. logstd is torch.Parameter by default, but can also be a
    linear function of the features.
    """
    def __init__(self, nin, nout, constant_log_std=True):
        """
        Args:
            nin  (int): dimensionality of the input
            nout (int): number of categories
            constant_log_std (bool): If False, logstd will be a linear function of the features.
        """
        super().__init__()
        self.constant_log_std = constant_log_std

        self.fc_mean = nn.Linear(nin, nout)
        nn.init.orthogonal_(self.fc_mean.weight.data, gain=1.0)
        nn.init.constant_(self.fc_mean.bias.data, 0)
        if constant_log_std:
            self.logstd = nn.Parameter(torch.zeros(nout))
        else:
            self.fc_logstd = nn.Linear(nin, nout)
            nn.init.orthogonal_(self.fc_logstd.weight.data, gain=0.01)
            nn.init.constant_(self.fc_logstd.bias.data, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): vectors of length nin
        Returns:
            dist (torch.distributions.Normal): normal distribution
        """
        mean = self.fc_mean(x)
        if self.constant_log_std:
            return Normal(mean, self.logstd.exp())
        else:
            logstd = self.fc_logstd(x)
            return Normal(mean, logstd.exp())



import unittest

class TestDistributions(unittest.TestCase):
    def test_categorical(self):
        cat = Categorical(10, 2)
        features = torch.ones(2,10)
        dist = cat(features)
        ac = dist.sample()
        assert ac.requires_grad == False
        assert ac.shape == (2, 1)
        assert torch.all(dist.mode()[0] == dist.mode()[1])
        assert dist.mode().shape == (2, 1)
        assert dist.log_prob(ac).shape == (2,)
        assert dist.entropy().shape == (2,)

    def test_normal(self):
        dg = DiagGaussian(10, 2)
        features = torch.ones(2,10)
        dist = dg(features)
        ac = dist.sample()
        assert ac.requires_grad == False
        assert ac.shape == (2, 2)
        assert torch.all(dist.mode()[0] == dist.mode()[1])
        assert dist.mode().shape == (2, 2)
        assert dist.log_prob(ac).shape == (2,)
        assert dist.entropy().shape == (2,)
        ent = dist.entropy()
        features = torch.zeros(2,10)
        dist = dg(features)
        assert torch.allclose(dist.entropy(), ent)

        ac = dist.rsample()
        assert ac.requires_grad == True

        dg = DiagGaussian(10, 2, constant_log_std=False)
        features = torch.ones(2,10)
        dist = dg(features)
        ac = dist.sample()
        assert ac.requires_grad == False
        assert ac.shape == (2, 2)
        assert torch.all(dist.mode()[0] == dist.mode()[1])
        assert dist.mode().shape == (2, 2)
        assert dist.log_prob(ac).shape == (2,)
        assert dist.entropy().shape == (2,)
        ent = dist.entropy()
        features = torch.zeros(2,10)
        dist = dg(features)
        assert not torch.allclose(dist.entropy(), ent)




if __name__ == '__main__':
    unittest.main()
