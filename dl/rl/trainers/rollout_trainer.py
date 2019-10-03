"""Extends RLTrainer with functionality for on policy RL algorithms."""
from dl.rl.trainers import RLTrainer
from dl.rl.util import RolloutStorage
from dl import nest
import gin
import torch


@gin.configurable(blacklist=['logdir'])
class RolloutTrainer(RLTrainer):
    """Extends RLTrainer with functionality for on policy RL algorithms.

    The resposibilities of this class are:
        - Handle storage of rollout data
        - Handle computing rollouts
        - Handle batching and iterating over rollout data

    Subclasses will need to:
        - Create/handle models and their predictions.
        - Implement Trainer.step (update model).
    """

    def __init__(self,
                 logdir,
                 env_fn,
                 rollout_length=128,
                 batch_size=32,
                 gamma=0.99,
                 lambda_=0.95,
                 norm_advantages=False,
                 **kwargs):
        """Init."""
        super().__init__(logdir, env_fn, **kwargs)
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambda_ = lambda_
        self.norm_advantages = norm_advantages

        self._ob = torch.from_numpy(self.env.reset()).to(self.device)

    def init_rollout_storage(self, state, other_keys=[]):
        """Initialize rollout storage."""
        if state is None:
            self.storage = RolloutStorage(self.rollout_length,
                                          self.nenv,
                                          device=self.device,
                                          other_keys=other_keys)
            self.init_state = None
            self.recurrent = False
        else:
            self.state_nest = nest.get_structure(state)
            states = nest.flatten(state)
            self.recurrent_keys = [f'state{i}' for i in range(len(states))]
            self.recurrent = True
            self.storage = RolloutStorage(
                self.rollout_length, self.nenv, device=self.device,
                other_keys=other_keys, recurrent_state_keys=self.recurrent_keys)
            self.init_state = []
            for state in states:
                self.init_state.append(torch.zeros(size=state.shape,
                                                   device=self.device,
                                                   dtype=state.dtype))

        self._other_keys = other_keys
        self._ob = torch.from_numpy(self.env.reset()).to(self.device)
        self._mask = torch.Tensor(
            [0. for _ in range(self.nenv)]).to(self.device)
        self._state = self.init_state

    def rollout_step(self):
        """Compute one environment step."""
        with torch.no_grad():
            if self.recurrent:
                state = nest.pack_sequence_as(self._state, self.state_nest)
                outs, keys = self.act(self._ob, state_in=state, mask=self._mask)
            else:
                outs, keys = self.act(self._ob, state_in=None, mask=None)
        ob, r, done, _ = self.env.step(outs.action.cpu().numpy())
        data = {}
        data['ob'] = self._ob
        data['ac'] = outs.action
        data['r'] = torch.from_numpy(r).float()
        data['mask'] = self._mask
        data['vpred'] = outs.value
        for key in self._other_keys:
            data[key] = keys[key]
        if self.recurrent:
            for i, name in enumerate(self.recurrent_keys):
                data[name] = self._state[i]
        self.storage.insert(data)
        self._ob = torch.from_numpy(ob).to(self.device)
        self._mask = torch.Tensor(
                [0.0 if done_ else 1.0 for done_ in done]).to(self.device)
        if self.recurrent:
            self._state = nest.flatten(outs.state_out)
        self.t += self.nenv

    def rollout(self):
        """Compute entire rollout and advantage targets."""
        for _ in range(self.rollout_length):
            self.rollout_step()
        with torch.no_grad():
            if self.recurrent:
                state = nest.pack_sequence_as(self._state, self.state_nest)
                outs, _ = self.act(self._ob, mask=self._mask, state_in=state)
            else:
                outs, _ = self.act(self._ob, mask=None, state_in=None)
            self.storage.compute_targets(outs.value,
                                         self._mask,
                                         self.gamma,
                                         use_gae=True,
                                         lambda_=self.lambda_,
                                         norm_advantages=self.norm_advantages)

    def _recurrent_generator(self):
        for batch in self.storage.recurrent_generator(self.batch_size):
            state = [batch[k] for k in self.recurrent_keys]
            new_batch = {'state': nest.pack_sequence_as(state,
                                                        self.state_nest)}
            for k in batch:
                if k not in self.recurrent_keys:
                    new_batch[k] = batch[k]
            yield new_batch

    def _feed_forward_generator(self):
        for batch in self.storage.feed_forward_generator(self.batch_size):
            batch['state'] = None
            batch['mask'] = None
            yield batch

    def rollout_sampler(self):
        """Create sampler to iterate over rollout data."""
        if self.recurrent:
            return self._recurrent_generator()
        else:
            return self._feed_forward_generator()

    def act(self, ob, state_in=None, mask=None):
        """Run the model to produce an action.

        Overwrite this method in subclasses.

        Returns:
            out: namedtuple output of Policy or QFunction
            data: dict containing addition data to store in RolloutStorage.
                  The keys should match 'other_keys' given to
                  init_rollout_storage().

        """
        raise NotImplementedError


if __name__ == '__main__':
    import unittest
    import shutil
    from dl.rl.modules import Policy, ActorCriticBase
    from dl.rl.envs import make_env
    from dl.modules import FeedForwardNet, Categorical, DiagGaussian
    from dl.modules import MaskedLSTM, TimeAndBatchUnflattener

    class FeedForwardBase(ActorCriticBase):
        """Test feed forward network."""

        def build(self):
            """Build network."""
            inshape = self.observation_space.shape[0]
            self.net = FeedForwardNet(inshape, [32, 32], activate_last=True)
            if hasattr(self.action_space, 'n'):
                self.dist = Categorical(32, self.action_space.n)
            else:
                self.dist = DiagGaussian(32, self.action_space.shape[0])
            self.vf = torch.nn.Linear(32, 1)

        def forward(self, ob):
            """Forward."""
            x = self.net(ob.float())
            return self.dist(x), self.vf(x)

    class RNNBase(ActorCriticBase):
        """Test recurrent network."""

        def build(self):
            """Build network."""
            inshape = self.observation_space.shape[0]
            self.net = FeedForwardNet(inshape, [32, 32], activate_last=True)
            if hasattr(self.action_space, 'n'):
                self.dist = Categorical(32, self.action_space.n)
            else:
                self.dist = DiagGaussian(32, self.action_space.shape[0])
            self.lstm = MaskedLSTM(32, 32, 1)
            self.tbf = TimeAndBatchUnflattener()
            self.vf = torch.nn.Linear(32, 1)

        def forward(self, ob, state_in=None, mask=None):
            """Forward."""
            x = self.net(ob.float())
            if state_in is None:
                x, state_out = self.lstm(self.tbf(x))
            else:
                x = self.tbf(x, state_in['lstm'][0])
                mask = self.tbf(mask, state_in['lstm'][0])
                x, state_out = self.lstm(x, state_in['lstm'], mask)
            x = self.tbf.flatten(x)
            state_out = {'lstm': state_out, '1': torch.zeros_like(state_out[0])}
            return self.dist(x), self.vf(x), state_out

    class T(RolloutTrainer):
        """Test trainer."""

        def __init__(self, *args, base=None, **kwargs):
            """Init."""
            super().__init__(*args, **kwargs)
            self.pi = Policy(base(self.env.observation_space,
                                  self.env.action_space))

            self.init_rollout_storage(self.pi(self._ob).state_out, ['key1'])

        def act(self, ob, state_in, mask):
            """Act."""
            outs = self.pi(ob, state_in, mask)
            other_keys = {'key1': torch.zeros_like(ob)}
            return outs, other_keys

        def step(self):
            """Step."""
            self.rollout()
            count = 0
            for batch in self.rollout_sampler():
                assert 'key1' in batch
                count += 1
                assert 'state' in batch
                assert 'mask' in batch
                self.act(batch['ob'], batch['state'], batch['mask'])
            if self.recurrent:
                assert count == self.nenv // self.batch_size
            else:
                assert count == (
                    (self.nenv*self.rollout_length) // self.batch_size)

        def state_dict(self):
            """State dict."""
            return {}

    def env_discrete(rank):
        """Create discrete env."""
        return make_env('CartPole-v1', rank=rank)

    def env_continuous(rank):
        """Create continuous env."""
        return make_env('LunarLanderContinuous-v2', rank=rank)

    class TestOnPolicyTrainer(unittest.TestCase):
        """Test case."""

        def test_feed_forward(self):
            """Test feed forward network."""
            t = T('./test', env_discrete, nenv=2, base=FeedForwardBase,
                  maxt=1000)
            t.train()
            shutil.rmtree('./test')
            t = T('./test', env_continuous, nenv=2, base=FeedForwardBase,
                  maxt=1000)
            t.train()
            shutil.rmtree('./test')

        def test_recurrent(self):
            """Test recurrent network."""
            t = T('./test', env_discrete, nenv=2, batch_size=1, base=RNNBase,
                  maxt=1000)
            t.train()
            shutil.rmtree('./test')
            t = T('./test', env_continuous, nenv=2, batch_size=1, base=RNNBase,
                  maxt=1000)
            t.train()
            shutil.rmtree('./test')

    unittest.main()
