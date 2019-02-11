import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
from dl.modules import *
from collections import namedtuple
import numpy as np


"""
This file defines torch modules for ValueFunction, QFunction, and Policy.
The constructors for these modules generally take as input:
    obs_shape (tuple):
            the shape of the observation space of the environment
    action_space (gym.Space):
            the action space of the enironment. QFunction and Policy only.
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
                - base has a method named recurrent_state_size, which returns
                  a list of shapes for each temporal state of the model.
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
class ValueFunction(nn.Module):
    def __init__(self, obs_shape, base=None):
        """
        Args:
            See above. base is assumed to be not recurrent.
        """
        super().__init__()
        if base:
            self.base = base(obs_shape)
        else:
            self.base = get_default_base(obs_shape)
        with torch.no_grad():
            in_shape = self.base(torch.zeros(obs_shape)[None]).shape[-1]

        self.vf = nn.Linear(in_shape, 1)
        nn.init.orthogonal_(self.vf.weight.data, gain=1.0)
        nn.init.constant_(self.vf.bias.data, 0)

        self.outputs = namedtuple('Outputs', ['value'])

    def forward(self, x):
        """
        Computes Q-value.
        Args:
            Same as self.base.forward (see above)
        Returns:
            out (namedtuple):
                out.action: The action corresponding to the argmax of the Q-values
                out.maxq:  The max of the Q-values
                out.qvals:  The Q-value for each action
        """
        x = self.base(x)
        value = self.vf(x).squeeze(-1)
        return self.outputs(value=value)


@gin.configurable(whitelist=['base'])
class QFunction(nn.Module):
    def __init__(self, obs_shape, action_space, base=None):
        """
        Args:
            See above. base is assumed to be not recurrent.

        When using a continuous action space, standard qfunction parameterizations
        take two inputs, state and action. In this case, the interface of
        'base.forward' is assumed to take both arguements.
        """
        super().__init__()
        self.action_space = action_space
        self.discrete = action_space.__class__.__name__ == 'Discrete'
        base_args = [obs_shape] if self.discrete else [obs_shape, action_space.shape]
        if base:
            self.base = base(*base_args)
        else:
            self.base = get_default_base(*base_args)
        if self.discrete:
            with torch.no_grad():
                in_shape = self.base(torch.zeros(obs_shape)[None]).shape[-1]
            self.qvals = nn.Linear(in_shape, self.action_space.n)
        else:
            with torch.no_grad():
                ac = torch.from_numpy(np.array([self.action_space.sample()]))
                in_shape = self.base(torch.zeros(obs_shape)[None], ac).shape[-1]
            self.qvals = nn.Linear(in_shape, 1)
        nn.init.orthogonal_(self.qvals.weight.data, gain=1.0)
        nn.init.constant_(self.qvals.bias.data, 0)

        self.outputs = namedtuple('Outputs', ['action', 'value', 'max_a', 'max_q', 'qvals'])

    def forward(self, x, action=None):
        """
        Computes Q-value.
        Args:
            Same as self.base.forward (see above)
        Returns:
            out (namedtuple):
                out.action: If an action is specified, out.action is the same, otherwise it is the argmax of the Q-values
                out.value: The q value of (x, out.action)
                out.max_a:  The argmax of the Q-values   (only available for discrete action spaces)
                out.max_q:  The max of the Q-values      (only available for discrete action spaces)
                out.qvals:  The Q-value for each action  (only available for discrete action spaces)
        """
        if action is None:
            assert self.discrete, "You must provide an action for a continuous action space"
            x = self.base(x)
            qvals = self.qvals(x)
            maxq, ac = qvals.max(dim=-1)
            return self.outputs(action=ac, value=maxq, max_a=ac, max_q=maxq, qvals=qvals)
        elif self.discrete:
            x = self.base(x)
            qvals = self.qvals(x)
            maxq, maxa = qvals.max(dim=-1)
            if len(action.shape) == 1:
                inds = action.long().unsqueeze(1)
            else:
                inds = action.long()
            value = qvals.gather(1, inds).squeeze(1)
            return self.outputs(action=action, value=value, max_a=maxa, max_q=maxq, qvals=qvals)
        else:
            x = self.base(x, action)
            value = self.qvals(x).squeeze(1)
            return self.outputs(action=action, value=value, max_a=None, max_q=None, qvals=None)





@gin.configurable(whitelist=['base', 'critic_base'])
class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, critic=True, critic_base=None, norm_observations=False, dist=None):
        """
        Args:
            obs_shape (tuple):         See above
            action_space (gym.Space):  See above
            base (nn.Module):          See above
            critic (bool, optional):
                    If False, no critic will be used.
            critic_base (nn.Module, optional):
                    The base network for the value function.
                    If not specified, critic_base will be the same as base.
                    If specified, critic_base is assumed to be not recurrent.
            running_ob_norm (bool):
                    If True, normalize observations passed to forward.
            dist (nn.Module):
                    If specified, overrides the default distribution.
        """
        super().__init__()
        if base:
            self.base = base(obs_shape)
        else:
            self.base = get_default_base(obs_shape)
        if critic and critic_base:
            self.critic_base = critic_base(obs_shape)
        else:
            self.critic_base = None
        self.critic = critic
        self.action_space = action_space
        with torch.no_grad():
            in_shape = self.base(torch.zeros(obs_shape)[None]).shape[-1]

        # init distribution
        if self.action_space.__class__.__name__ == 'Discrete':
            args = [in_shape, self.action_space.n]
            defualt_dist = Categorical
        elif self.action_space.__class__.__name__ == 'Box':
            args = [in_shape, np.prod(self.action_space.shape).item()]
            defualt_dist = DiagGaussian
        else:
            assert False, f"Uknown action space {self.action_space.__class__.__name__}"
        self.dist = dist(*args) if dist else defualt_dist(*args)

        # init value function haed
        if critic:
            if critic_base:
                with torch.no_grad():
                    in_shape = self.critic_base(torch.zeros(obs_shape)[None]).shape[-1]
            self.vf = nn.Linear(in_shape, 1)


        if norm_observations:
            self.running_norm = RunningObNorm(obs_shape)
        else:
            self.running_norm = None

        self.outputs = namedtuple('Outputs', ['action', 'value', 'logp', 'logstd', 'dist', 'state_out'])

    def _run_bases(self, x, mask, state_in):
        if state_in is None:
            state_out = None
            out = self.base(x)
        else:
            out, state_out = self.base(x, mask=mask, state_in=state_in)
        if self.critic and self.critic_base:
            vf_out = self.critic_base(x)
        elif self.critic:
            vf_out = out
        else:
            vf_out = None
        return out, vf_out, state_out

    def forward(self, X, mask=None, state_in=None, deterministic=False, reparameterization_trick=False):
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
                out.logstd: The log std deviation of out.dist
                out.dist:   The action distribution
                out.state_out:  The temporal state of base (See above for details)
        """
        if self.running_norm:
            X = self.running_norm(X)
        out, vf_out, state_out = self._run_bases(X, mask, state_in)

        if isinstance(self.dist, DiagGaussian):
            dist, logstd = self.dist(out, return_logstd=True)
        else:
            dist = self.dist(out)
            logstd = None
        if deterministic:
            action = dist.mode()
        elif reparameterization_trick:
            try:
                action = dist.rsample()
            except:
                assert False, f"{dist.__class__.__name__} distribution does not have a reparameterization trick."
        else:
            action = dist.sample()

        if self.critic:
            value = self.vf(vf_out).squeeze(-1)
        else:
            value = None

        return self.outputs(value=value, action=action, logp=dist.log_prob(action), logstd=logstd, dist=dist, state_out=state_out)

    def recurrent_state_size(self):
        if not hasattr(self.base, 'recurrent_state_size'):
            return None
        else:
            return self.base.recurrent_state_size()


from dl.util import conv_out_shape

@gin.configurable
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



@gin.configurable
class FeedForwardBase(nn.Module):
    def __init__(self, ob_shape, *args, **kwargs):
        super().__init__()
        self.net = FeedForwardNet(ob_shape[0], *args, **kwargs)

    def forward(self, x):
        return self.net(x.float())


@gin.configurable
class AppendActionFeedForwardBase(nn.Module):
    def __init__(self, ob_shape, ac_shape, *args, **kwargs):
        super().__init__()
        self.net = FeedForwardNet(ob_shape[0] + ac_shape[0], *args, **kwargs)

    def forward(self, x, a):
        return self.net(torch.cat([x.float(), a.float()], -1))


def get_default_base(obs_shape, ac_shape=None):
    if ac_shape:
        assert len(obs_shape) == 1, "Default base for continuous action spaces requires one dimensional observations."
        return AppendActionFeedForwardBase(obs_shape, ac_shape, units=[64,64], activation_fn=torch.tanh, activate_last=True)
    if len(obs_shape) == 1:
        return FeedForwardBase(obs_shape, units=[64,64], activation_fn=torch.tanh, activate_last=True)
    if len(obs_shape) == 3:
        return NatureDQN(obs_shape)
    assert False, f"No default network for inputs of {len(obs_shape)} dimensions"



import unittest
from dl.util import atari_env
import gym

class TestRLModules(unittest.TestCase):
    def testValueFunction(self):
        env = atari_env('Pong')
        net = ValueFunction(env.observation_space.shape)
        ob = env.reset()
        for _ in range(10):
            outs = net(torch.from_numpy(ob[None]))
            assert outs.value.shape == (1,)
            ob, r, done, _ = env.step(env.action_space.sample())

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

        out1 = net(torch.from_numpy(ob[None]))
        outs = net(torch.from_numpy(ob[None]), (out1.action + 1) % env.action_space.n)
        assert outs.action.shape == (1,)
        assert outs.max_q.shape == (1,)
        assert outs.qvals.shape == (1,env.action_space.n)
        assert outs.action != out1.action
        assert outs.value != out1.value

        class SABase(nn.Module):
            def __init__(self, ob_shape, ac_shape):
                super().__init__()

            def forward(self, x, a):
                return x

        env = gym.make('MountainCarContinuous-v0')
        net = QFunction(env.observation_space.shape, env.action_space, base=SABase)
        ob = env.reset()
        for _ in range(10):
            ac = torch.from_numpy(np.array([env.action_space.sample()])).float()
            outs = net(torch.from_numpy(ob[None]).float(), ac)
            assert outs.action.shape == (1,*env.action_space.shape)
            assert outs.value.shape == (1,)
            assert outs.max_q == None
            assert outs.qvals == None
            ob, r, done, _ = env.step(outs.action[0])


    def testPolicy(self):
        env = atari_env('Pong')
        net = Policy(env.observation_space.shape, env.action_space, norm_observations=True)
        ob = env.reset()
        for _ in range(10):
            outs = net(torch.from_numpy(ob[None]))
            assert outs.action.shape == (1,1)
            assert outs.value.shape == (1,)
            assert outs.state_out is None
            ob, r, done, _ = env.step(outs.action[0])
        state = net.state_dict()
        assert 'running_norm.mean' in state
        assert 'running_norm.var' in state
        assert 'running_norm.count' in state

        env = gym.make('MountainCarContinuous-v0')
        net = Policy(env.observation_space.shape, env.action_space)
        ob = env.reset()
        for _ in range(10):
            outs = net(torch.from_numpy(ob[None]).float())
            assert outs.action.shape == (1,*env.action_space.shape)
            assert outs.value.shape == (1,)
            assert outs.state_out is None
            ob, r, done, _ = env.step(outs.action[0])

        net = Policy(env.observation_space.shape, env.action_space, dist=TanhDiagGaussian)
        ob = env.reset()
        for _ in range(10):
            outs = net(torch.from_numpy(ob[None]).float())
            assert outs.action.shape == (1,*env.action_space.shape)
            assert outs.value.shape == (1,)
            assert outs.state_out is None
            assert outs.dist.__class__.__name__ == 'TanhNormal'
            assert torch.abs(outs.action) < 1
            ob, r, done, _ = env.step(outs.action[0])



if __name__=='__main__':
    unittest.main()
