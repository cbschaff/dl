import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
from dl.modules import *
from collections import namedtuple
import numpy as np


"""
This file defines torch modules for a QFunction and Policy.
The constructors for these modules generally take as input:
    obs_shape (tuple):
            the shape of the observation space of the environment
    action_space (gym.Space):
            the action space of the enironment
    base (nn.Module):
            the base module whose output is the features used by the
            QFunction or Policy. These modules place minor assumptions
            on the base:

            - base.__init__ takes one argument:
                Args:
                    in_shape (int):
                        the shape of the observation space

            For a non-recurrent base:
                - base.forward has the following interface:
                    Args:
                        X (torch.Tensor):
                            The input to base. Usually the current observation.
                    Returns:
                        out (torch.Tensor):
                            A 2-d Tensor containing the output of the module
                            (i.e the output of the penultimate hidden layer
                            of a DQN)

            For a recurrent base:
                - base.forward has the following interface:
                    Args:
                        X (torch.Tensor):
                            The input to base. Usually the current observation.
                        mask  (torch.Tensor):
                            The "done mask" to reset the hidden state when an
                            episode ends.
                        state_in (list):
                            The hidden state of the base.
                    Returns:
                        out (torch.Tensor):
                            A 2-d Tensor containing the output of the module
                            (Usually the output of the penultimate hidden layer)
                        state_out (list):
                            A list of the temporal state of the module.
                            Algorithms will pass the returned state as input
                            to the forward method.

            If the base is not specified, a standard MLP or CNN will
            be used.
"""


@gin.configurable(whitelist=['base'])
class QFunction(nn.Module):
    def __init__(self, obs_shape, action_space, base=None):
        """
        Args:
            See above. base is assumed to be not recurrent.
        """
        super().__init__()
        if base:
            self.base = base(obs_shape)
        else:
            self.base = get_default_base(obs_shape)
        self.action_space = action_space
        assert self.action_space.__class__.__name__ == 'Discrete'
        self.qvals = None
        self.outputs = namedtuple('Outputs', ['action', 'max_q', 'qvals'])

    def _init_qvals(self, in_shape):
        self.qvals = nn.Linear(in_shape, self.action_space.n)
        nn.init.orthogonal_(self.qvals.weight.data, gain=1.0)
        nn.init.constant_(self.qvals.bias.data, 0)
        self.qvals.to(next(self.base.parameters()).device)

    def forward(self, x):
        """
        Computes Q-value.
        Args:
            Same as self.base.forward (see above)
        Returns:
            out (namedtuple):
                out.action: The action corresponding to the argmax of the Q-values
                out.max_q:  The max of the Q-values
                out.qvals:  The Q-value for each action
        """
        x = self.base(x)
        if self.qvals is None:
            assert len(x.shape) == 2, 'Output of base should be a feature vector for each input.'
            self._init_qvals(x.shape[-1])
        qvals = self.qvals(x)
        max_q, action = qvals.max(dim=-1)
        return self.outputs(action=action, max_q=max_q, qvals=qvals)



@gin.configurable(whitelist=['base', 'critic_base'])
class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, critic_base=None):
        """
        Args:
            obs_shape (tuple):         See above
            action_space (gym.Space):  See above
            base (nn.Module):          See above
            critic_base (nn.Module, optional):
                    The base network for the value function.
                    If not specified, critic_base will be the same as base.
                    If specified, critic_base is assumed to be not recurrent.
        """
        super().__init__()
        if base:
            self.base = base(obs_shape)
        else:
            self.base = get_default_base(obs_shape)
        if critic_base:
            self.critic_base = critic_base(obs_shape)
        else:
            self.critic_base = None
        self.action_space = action_space
        self.dist = None
        self.vf = None
        self.outputs = namedtuple('Outputs', ['action', 'value', 'logp', 'dist', 'state_out'])

    def _init_dist(self, in_shape):
        if self.action_space.__class__.__name__ == 'Discrete':
            self.dist = Categorical(in_shape, self.action_space.n)
        elif self.action_space.__class__.__name__ == 'Box':
            self.dist = DiagGaussian(in_shape, self.action_space.n)
        else:
            assert False, "Uknown action space {self.action_space.__class__.__name__}"
        # Assume all parameters are on the same device
        self.dist.to(next(self.base.parameters()).device)

    def _init_vf(self, in_shape):
        self.vf = nn.Linear(in_shape, 1)
        if self.critic_base:
            self.vf.to(next(self.critic_base.parameters()).device)
        else:
            self.vf.to(next(self.base.parameters()).device)

    def _run_bases(self, x, mask, state_in):
        if state_in is None:
            state_out = None
            out = self.base(x)
        else:
            out, state_out = self.base(x, mask=mask, state_in=state_in)
        if self.critic_base:
            vf_out = self.critic_base(x)
        else:
            vf_out = out
        return out, vf_out, state_out

    def forward(self, X, mask=None, state_in=None, deterministic=False):
        """
        Computes the action of the policy and the value of the input.
        Args:
            deterministic (bool): True  => return mode of action dist,
                                  False => sample from action dist.
            Other args are the same as self.base.forward (see above)
        Returns:
            out (namedtuple):
                out.action: The sampled action, or the mode if deterministic=True
                out.value:  The value of the current observation
                out.logp:   The log probability of out.action
                out.dist:   The action distribution
                out.state_out:  The temporal state of base (See above for details)
        """
        out, vf_out, state_out = self._run_bases(X, mask, state_in)

        if self.dist is None:
            assert len(out.shape) == 2, 'Output of base should be a feature vector for each input.'
            self._init_dist(out.shape[-1])

        if self.vf is None:
            assert len(out.shape) == 2, 'Output of base should be a feature vector for each input.'
            self._init_vf(out.shape[-1])

        dist = self.dist(out)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        value = self.vf(vf_out).squeeze(-1)

        return self.outputs(value=value, action=action, logp=dist.log_prob(action), dist=dist, state_out=state_out)


from dl.util import conv_out_shape

class NatureDQN(nn.Module):
    """
    Deep network from https://www.nature.com/articles/nature14236
    """
    def __init__(self, img_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(img_shape[0], 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        shape = img_shape[1:]
        for c in [self.conv1, self.conv2, self.conv3]:
            shape = conv_out_shape(shape, c)
        self.nunits = 64 * np.prod(shape)
        self.fc = nn.Linear(self.nunits, 512)

    def forward(self, x):
        x = x.float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return F.relu(self.fc(x.view(-1, self.nunits)))


def get_default_base(obs_shape):
    if len(obs_shape) == 1:
        return FeedForward(obs_shape[0], units=[64,64,64], activation_fn=F.tanh, activate_last=True)
    if len(obs_shape) == 3:
        return NatureDQN(obs_shape)
    assert False, f"No default network for inputs of {len(obs_shape)} dimensions"



import unittest
from dl.util import atari_env

class TestRLModules(unittest.TestCase):
    def testQFunction(self):
        env = atari_env('Pong')
        net = QFunction(env.observation_space.shape, env.action_space)
        ob = env.reset()
        for _ in range(10):
            outs = net(torch.from_numpy(ob[None]))
            assert outs.action.shape == (1,)
            assert outs.max_q.shape == (1,)
            assert outs.qvals.shape == (1,env.action_space.n)
            ob, r, done, _ = env.step(outs.action[0])


    def testPolicy(self):
        env = atari_env('Pong')
        net = Policy(env.observation_space.shape, env.action_space)
        ob = env.reset()
        for _ in range(10):
            outs = net(torch.from_numpy(ob[None]))
            assert outs.action.shape == (1,1)
            assert outs.value.shape == (1,)
            assert outs.state_out is None
            ob, r, done, _ = env.step(outs.action[0])


if __name__=='__main__':
    unittest.main()
