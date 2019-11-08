"""Standardize distribution interface and make convenience modules."""
import torch
import torch.nn as nn
import torch.distributions as D
import gin


class CatDist(D.Categorical):
    """Categorical disctribution."""

    def mode(self):
        """Mode."""
        return self.probs.argmax(dim=-1)


class Normal(D.Normal):
    """Normal Distribution."""

    def mode(self):
        """mode."""
        return self.mean

    def log_prob(self, ac):
        """Log prob."""
        return super().log_prob(ac).sum(-1)

    def entropy(self):
        """Entropy."""
        return super().entropy().sum(-1)


class TanhNormal(D.Distribution):
    """Squash normal samples with tanh.

    Modified from
    https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/distributions.py"
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """Init."""
        self.normal = D.Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def mode(self):
        """Mode."""
        return torch.tanh(self.normal.mean)

    def log_prob(self, value, pre_tanh_value=None):
        """Log prob."""
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        logps = self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )
        return logps.sum(-1)

    def sample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """Sample.

        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample(sample_shape).detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """Sample with reparameterization trick."""
        z = self.normal.rsample(sample_shape)
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def entropy(self):
        """Entropy."""
        # TODO: implement this.
        return torch.zeros([self.normal.mean.shape[0]],
                           device=self.normal.mean.device)


class DeltaDist(D.Distribution):
    """Delta distribution."""

    def __init__(self, x):
        """Init."""
        self._x = x
        self.batch_size = self.mean.shape[0]

    @property
    def mean(self):
        """Mean."""
        return self._x

    @property
    def stddev(self):
        """Std."""
        return torch.zeros_like(self._x)

    @property
    def variance(self):
        """Variance."""
        return torch.zeros_like(self._x)

    def mode(self):
        """Mode."""
        return self._x

    def log_prob(self, value):
        """Log probability."""
        zeros = torch.zeros([self.batch_size], device=self._x.device)
        if torch.allclose(self._x, value):
            return zeros
        else:
            return torch.log(zeros)

    def sample(self, sample_shape=torch.Size()):
        """Sample."""
        return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Reprameterized sample."""
        shape = sample_shape + torch.Size([1 for _ in self._x.shape])
        return self._x.repeat(shape)


"""
Modules
"""


class Categorical(nn.Module):
    """Categorical distribution.

    Unnormalized logits are parameterized as a linear function of the features.
    """

    def __init__(self, nin, nout):
        """Init.

        Args:
            nin  (int): dimensionality of the input
            nout (int): number of categories

        """
        super().__init__()

        self.linear = nn.Linear(nin, nout)
        nn.init.orthogonal_(self.linear.weight.data, gain=0.01)
        nn.init.constant_(self.linear.bias.data, 0)

    def forward(self, x):
        """Forward.

        Args:
            x (torch.Tensor): vectors of length nin
        Returns:
            dist (torch.distributions.Categorical): Categorical distribution

        """
        x = self.linear(x)
        return CatDist(logits=x)


class Delta(nn.Module):
    """Delta distribution."""

    def __init__(self, nin, nout):
        """Init.

        Args:
            nin  (int): dimensionality of the input
            nout (int): number of categories

        """
        super().__init__()

        self.linear = nn.Linear(nin, nout)
        nn.init.orthogonal_(self.linear.weight.data, gain=0.01)
        nn.init.constant_(self.linear.bias.data, 0)

    def forward(self, x):
        """Forward.

        Args:
            x (torch.Tensor): vectors of length nin
        Returns:
            dist (torch.distributions.Categorical): Categorical distribution

        """
        return DeltaDist(self.linear(x))


class TanhDelta(Delta):
    """Same as Delta, but the input to DeltaDist is passed through a tanh."""

    def forward(self, x):
        """Forward.

        Args:
            x (torch.Tensor): vectors of length nin
        Returns:
            dist (torch.distributions.Categorical): Categorical distribution

        """
        return DeltaDist(torch.tanh(self.linear(x)))


@gin.configurable
class DiagGaussian(nn.Module):
    """Diagonal Gaussian distribution.

    Mean is parameterized as a linear function of the features.
    logstd is torch.Parameter by default, but can also be a linear function
    of the features.
    """

    def __init__(self, nin, nout, constant_log_std=True, log_std_min=-20,
                 log_std_max=2):
        """Init.

        Args:
            nin  (int): dimensionality of the input
            nout (int): number of categories
            constant_log_std (bool): If False, logstd will be a linear function
                                     of the features.

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
        """Forward.

        Args:
            x (torch.Tensor): vectors of length nin
        Returns:
            dist (torch.distributions.Normal): normal distribution

        """
        mean = self.fc_mean(x)
        if self.constant_log_std:
            logstd = torch.clamp(self.logstd, self.log_std_min,
                                 self.log_std_max)
        else:
            logstd = torch.clamp(self.fc_logstd(x), self.log_std_min,
                                 self.log_std_max)
        if return_logstd:
            return Normal(mean, logstd.exp()), logstd
        else:
            return Normal(mean, logstd.exp())


@gin.configurable
class TanhDiagGaussian(DiagGaussian):
    """Tanh gaussian module.

    Module for TanhNormal distribution for constrained continuous action spaces.
    This is esspecially useful for off-policy rl algorithms in which the action
    is an input to a QFunction parameterized by a neural network.
    """

    def forward(self, x, return_logstd=False):
        """Forward.

        Args:
            x (torch.Tensor): vectors of length nin
        Returns:
            dist (torch.distributions.Normal): normal distribution

        """
        mean = self.fc_mean(x)
        if self.constant_log_std:
            logstd = torch.clamp(self.logstd, self.log_std_min,
                                 self.log_std_max)
        else:
            logstd = torch.clamp(self.fc_logstd(x), self.log_std_min,
                                 self.log_std_max)
        if return_logstd:
            return TanhNormal(mean, logstd.exp()), logstd
        else:
            return TanhNormal(mean, logstd.exp())


if __name__ == '__main__':
    import unittest

    class TestDistributions(unittest.TestCase):
        """Test."""

        def test_categorical(self):
            """Test categorical."""
            cat = Categorical(10, 2)
            features = torch.ones(2, 10)
            dist = cat(features)
            ac = dist.sample()
            assert not ac.requires_grad
            assert ac.shape == (2,)
            assert torch.all(dist.mode()[0] == dist.mode()[1])
            assert dist.mode().shape == (2,)
            assert dist.log_prob(ac).shape == (2,)
            assert dist.entropy().shape == (2,)

        def test_delta(self):
            """Test delta."""
            d = Delta(10, 2)
            dtanh = TanhDelta(10, 2)
            dtanh.load_state_dict(d.state_dict())
            features = torch.ones(2, 10)
            dist = d(features)
            ac = dist.sample()
            assert not ac.requires_grad
            assert ac.shape == (2, 2)
            assert torch.all(dist.mode()[0] == dist.mode()[1])
            assert dist.mode().shape == (2, 2)
            assert dist.log_prob(ac).shape == (2,)

            assert torch.allclose(dist.log_prob(ac), torch.Tensor([0.]))
            assert torch.allclose(dist.log_prob(ac + 1),
                                  torch.Tensor([float('-inf')]))
            actanh = dtanh(features).mode()
            assert torch.allclose(torch.tanh(ac), actanh)

        def test_normal(self):
            """Test normal."""
            dg = DiagGaussian(10, 2)
            features = torch.ones(2, 10)
            dist = dg(features)
            ac = dist.sample()
            assert not ac.requires_grad
            assert ac.shape == (2, 2)
            assert torch.all(dist.mode()[0] == dist.mode()[1])
            assert dist.mode().shape == (2, 2)
            assert dist.log_prob(ac).shape == (2,)
            assert dist.entropy().shape == (2,)
            ent = dist.entropy()
            features = torch.zeros(2, 10)
            dist = dg(features)
            assert torch.allclose(dist.entropy(), ent)

            ac = dist.rsample()
            assert ac.requires_grad

            dg = DiagGaussian(10, 2, constant_log_std=False)
            features = torch.ones(2, 10)
            dist = dg(features)
            ac = dist.sample()
            assert not ac.requires_grad
            assert ac.shape == (2, 2)
            assert torch.all(dist.mode()[0] == dist.mode()[1])
            assert dist.mode().shape == (2, 2)
            assert dist.log_prob(ac).shape == (2,)
            assert dist.entropy().shape == (2,)
            ent = dist.entropy()
            features = torch.zeros(2, 10)
            dist = dg(features)
            assert not torch.allclose(dist.entropy(), ent)

            dist, logstd = dg(features, return_logstd=True)
            assert torch.allclose(logstd, torch.zeros([1]))

        def test_tanhnormal(self):
            """Test tanh normal."""
            dg = TanhDiagGaussian(10, 2)
            features = torch.ones(2, 10)
            dist = dg(features)
            ac = dist.sample()
            assert not ac.requires_grad
            assert ac.shape == (2, 2)
            assert torch.all(dist.mode()[0] == dist.mode()[1])
            assert dist.mode().shape == (2, 2)
            assert dist.log_prob(ac).shape == (2,)
            features = torch.zeros(2, 10)
            dist = dg(features)

            ac = dist.rsample()
            assert ac.requires_grad

            dg = TanhDiagGaussian(10, 2, constant_log_std=False)
            features = torch.ones(2, 10)
            dist = dg(features)
            ac = dist.sample()
            assert not ac.requires_grad
            assert ac.shape == (2, 2)
            assert torch.all(dist.mode()[0] == dist.mode()[1])
            assert dist.mode().shape == (2, 2)
            assert dist.log_prob(ac).shape == (2,)
            features = torch.zeros(2, 10)
            dist = dg(features)

            dist, logstd = dg(features, return_logstd=True)
            assert torch.allclose(logstd, torch.zeros([1]))

            ac, pac = dist.sample(return_pretanh_value=True)
            logp = dist.log_prob(ac, pac)
            logp2 = dist.log_prob(ac)
            assert torch.allclose(logp, logp2)
            assert logp.shape == (2,)

    unittest.main()
