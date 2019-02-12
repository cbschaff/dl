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

class TanhNormal(D.Distribution):
    """
    Modified from https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/distributions.py"
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal = D.Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def mode(self):
        return torch.tanh(self.normal.mean)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        logps = self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )
        return logps.sum(-1)

    def sample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = self.normal.rsample()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def entropy(self):
        # TODO: implement this.
        return 0.



"""
Modules
"""

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
    def __init__(self, nin, nout, constant_log_std=True, log_std_min=-20, log_std_max=2):
        """
        Args:
            nin  (int): dimensionality of the input
            nout (int): number of categories
            constant_log_std (bool): If False, logstd will be a linear function of the features.
        """
        super().__init__()
        self.constant_log_std = constant_log_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc_mean = nn.Linear(nin, nout)
        nn.init.orthogonal_(self.fc_mean.weight.data, gain=1.0)
        nn.init.constant_(self.fc_mean.bias.data, 0)
        if constant_log_std:
            self.logstd = nn.Parameter(torch.zeros(nout))
        else:
            self.fc_logstd = nn.Linear(nin, nout)
            nn.init.orthogonal_(self.fc_logstd.weight.data, gain=0.01)
            nn.init.constant_(self.fc_logstd.bias.data, 0)

    def forward(self, x, return_logstd=False):
        """
        Args:
            x (torch.Tensor): vectors of length nin
        Returns:
            dist (torch.distributions.Normal): normal distribution
        """
        mean = self.fc_mean(x)
        if self.constant_log_std:
            logstd = torch.clamp(self.logstd, self.log_std_min, self.log_std_max)
        else:
            logstd = torch.clamp(self.fc_logstd(x), self.log_std_min, self.log_std_max)
        if return_logstd:
            return Normal(mean, logstd.exp()), logstd
        else:
            return Normal(mean, logstd.exp())


@gin.configurable
class TanhDiagGaussian(DiagGaussian):
    """
    Module for TanhNormal distribution for constrained continuous action spaces.
    This is esspecially useful for off-policy rl algorithms in which the action
    is an input to a QFunction parameterized by a neural network.
    """
    def forward(self, x, return_logstd=False):
        """
        Args:
            x (torch.Tensor): vectors of length nin
        Returns:
            dist (torch.distributions.Normal): normal distribution
        """
        mean = self.fc_mean(x)
        if self.constant_log_std:
            logstd = torch.clamp(self.logstd, self.log_std_min, self.log_std_max)
        else:
            logstd = torch.clamp(self.fc_logstd(x), self.log_std_min, self.log_std_max)
        if return_logstd:
            return TanhNormal(mean, logstd.exp()), logstd
        else:
            return TanhNormal(mean, logstd.exp())



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

        dist, logstd = dg(features, return_logstd=True)
        assert torch.allclose(logstd, torch.zeros([1]))

    def test_tanhnormal(self):
        dg = TanhDiagGaussian(10, 2)
        features = torch.ones(2,10)
        dist = dg(features)
        ac = dist.sample()
        assert ac.requires_grad == False
        assert ac.shape == (2, 2)
        assert torch.all(dist.mode()[0] == dist.mode()[1])
        assert dist.mode().shape == (2, 2)
        assert dist.log_prob(ac).shape == (2,)
        features = torch.zeros(2,10)
        dist = dg(features)

        ac = dist.rsample()
        assert ac.requires_grad == True

        dg = TanhDiagGaussian(10, 2, constant_log_std=False)
        features = torch.ones(2,10)
        dist = dg(features)
        ac = dist.sample()
        assert ac.requires_grad == False
        assert ac.shape == (2, 2)
        assert torch.all(dist.mode()[0] == dist.mode()[1])
        assert dist.mode().shape == (2, 2)
        assert dist.log_prob(ac).shape == (2,)
        features = torch.zeros(2,10)
        dist = dg(features)

        dist, logstd = dg(features, return_logstd=True)
        assert torch.allclose(logstd, torch.zeros([1]))

        ac, pac = dist.sample(return_pretanh_value=True)
        logp = dist.log_prob(ac, pac)
        logp2 = dist.log_prob(ac)
        assert torch.allclose(logp, logp2)
        assert logp.shape == (2,)



if __name__ == '__main__':
    unittest.main()
