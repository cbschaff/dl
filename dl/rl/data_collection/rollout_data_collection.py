"""Code for storing and iterating over rollout data."""
from dl.rl.data_collection import RolloutStorage
from dl.rl.util import ensure_vec_env
from dl import nest
import torch


class RolloutDataManager(object):
    """Collects data from environments and stores it in a RolloutStorage.

    The resposibilities of this class are:
        - Handle storage of rollout data
        - Handle computing rollouts
        - Handle batching and iterating over rollout data

    act_fn:
        A callable which takes in the observation, recurrent state and
        mask and returns:
            - a dictionary with the data to store in the rollout. 'action'
              and 'value' must be in the dict. Recurrent states must
              be nested under the 'state' key. All values except
              data['state'] must be pytorch Tensors.
    """

    def __init__(self,
                 env,
                 act_fn,
                 device,
                 rollout_length=128,
                 batch_size=32,
                 gamma=0.99,
                 lambda_=0.95,
                 norm_advantages=False):
        """Init."""
        self.env = ensure_vec_env(env)
        self.nenv = self.env.num_envs
        self.act = act_fn
        self.device = device
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambda_ = lambda_
        self.norm_advantages = norm_advantages

        self.storage = None

    def init_rollout_storage(self):
        """Initialize rollout storage."""
        self._ob = torch.from_numpy(self.env.reset()).to(self.device)
        data = self.act(self._ob)
        if 'action' not in data:
            raise ValueError('the key "action" must be in the dict returned '
                             'act_fn')
        if 'value' not in data:
            raise ValueError('the key "value" must be in the dict returned '
                             'act_fn')
        state = None
        if 'state' in data:
            state = data['state']

        if state is None:
            self.storage = RolloutStorage(self.rollout_length,
                                          self.nenv,
                                          device=self.device,
                                          recurrent=False)
            self.init_state = None
            self.recurrent = False
        else:
            self.recurrent = True
            self.storage = RolloutStorage(self.rollout_length,
                                          self.nenv,
                                          device=self.device,
                                          recurrent=True)

            def _init_state(s):
                return torch.zeros(size=s.shape, device=self.device,
                                   dtype=s.dtype)

            self.init_state = nest.map_structure(_init_state, state)

        self._ob = torch.from_numpy(self.env.reset()).to(self.device)
        self._mask = torch.Tensor(
            [0. for _ in range(self.nenv)]).to(self.device)
        self._state = self.init_state

    def rollout_step(self):
        """Compute one environment step."""
        if not self.storage:
            self.init_rollout_storage()
        with torch.no_grad():
            if self.recurrent:
                outs = self.act(self._ob, state_in=self._state, mask=self._mask)
            else:
                outs = self.act(self._ob, state_in=None, mask=None)
        ob, r, done, _ = self.env.step(outs['action'].cpu().numpy())
        data = {}
        data['obs'] = self._ob
        data['action'] = outs['action']
        data['reward'] = torch.from_numpy(r).float()
        data['mask'] = self._mask
        data['vpred'] = outs['value']
        for key in outs:
            if key != 'action':
                data[key] = outs[key]
        self.storage.insert(data)
        self._ob = torch.from_numpy(ob).to(self.device)
        self._mask = torch.Tensor(
                [0.0 if done_ else 1.0 for done_ in done]).to(self.device)
        if self.recurrent:
            self._state = outs['state']

    def rollout(self):
        """Compute entire rollout and advantage targets."""
        for _ in range(self.rollout_length):
            self.rollout_step()
        with torch.no_grad():
            if self.recurrent:
                outs = self.act(self._ob, state_in=self._state, mask=self._mask)
            else:
                outs = self.act(self._ob, state_in=None, mask=None)
            self.storage.compute_targets(outs['value'],
                                         self._mask,
                                         self.gamma,
                                         use_gae=True,
                                         lambda_=self.lambda_,
                                         norm_advantages=self.norm_advantages)

    def sampler(self):
        """Create sampler to iterate over rollout data."""
        return self.storage.sampler(self.batch_size)

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
    from dl.rl.trainers import RLTrainer
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

    class RolloutActor(object):
        """actor."""

        def __init__(self, pi):
            """init."""
            self.pi = pi

        def __call__(self, ob, state_in=None, mask=None):
            """act."""
            outs = self.pi(ob, state_in, mask)
            data = {'value': outs.value,
                    'action': outs.action,
                    'key1': torch.zeros_like(ob)}
            if outs.state_out:
                data['state'] = outs.state_out
            return data

    class T(RLTrainer):
        """Test trainer."""

        def __init__(self, *args, base=None, batch_size=32, **kwargs):
            """Init."""
            super().__init__(*args, **kwargs)
            self.pi = Policy(base(self.env.observation_space,
                                  self.env.action_space))
            self.data_manager = RolloutDataManager(self.env,
                                                   RolloutActor(self.pi),
                                                   self.device,
                                                   batch_size=batch_size)

        def step(self):
            """Step."""
            self.data_manager.rollout()
            count = 0
            for batch in self.data_manager.sampler():
                assert 'key1' in batch
                count += 1
                assert 'mask' in batch
                if self.data_manager.recurrent:
                    assert 'state' in batch
                    self.data_manager.act(batch['obs'], batch['state'],
                                          batch['mask'])
                else:
                    self.data_manager.act(batch['obs'])
            if self.data_manager.recurrent:
                assert count == self.nenv // self.data_manager.batch_size
            else:
                assert count == (
                    (self.nenv * self.data_manager.rollout_length)
                    // self.data_manager.batch_size)
            self.t += self.nenv * self.data_manager.rollout_length

        def state_dict(self):
            """State dict."""
            return {}

    def env_discrete(rank):
        """Create discrete env."""
        return make_env('CartPole-v1', rank=rank)

    def env_continuous(rank):
        """Create continuous env."""
        return make_env('LunarLanderContinuous-v2', rank=rank)

    class TestRolloutDataCollection(unittest.TestCase):
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
