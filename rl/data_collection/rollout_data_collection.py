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
                 batch_size=32,
                 gamma=0.99,
                 lambda_=0.95,
                 norm_advantages=False):
        """Init."""
        self.env = ensure_vec_env(env)
        self.nenv = self.env.num_envs
        self.act = act_fn
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambda_ = lambda_
        self.norm_advantages = norm_advantages

        self.storage = RolloutStorage(self.nenv, device=self.device)
        self._initialized = False

    def init_rollout_storage(self):
        """Initialize rollout storage."""
        def _to_torch(o):
            return torch.from_numpy(o).to(self.device)
        self._ob = nest.map_structure(_to_torch, self.env.reset())
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
            self.init_state = None
            self.recurrent = False
        else:
            self.recurrent = True

            def _init_state(s):
                return torch.zeros(size=s.shape, device=self.device,
                                   dtype=s.dtype)

            self.init_state = nest.map_structure(_init_state, state)

        self._initialized = True

    def _reset(self):
        if not self._initialized:
            self.init_rollout_storage()

        def _to_torch(o):
            return torch.from_numpy(o).to(self.device)
        self._ob = nest.map_structure(_to_torch, self.env.reset())
        self._state = self.init_state
        self.storage.reset()

    def rollout_step(self):
        """Compute one environment step."""
        with torch.no_grad():
            if self.recurrent:
                outs = self.act(self._ob, state_in=self._state)
            else:
                outs = self.act(self._ob, state_in=None)
        ob, r, done, _ = self.env.step(outs['action'].cpu().numpy())
        data = {}
        data['obs'] = self._ob
        data['action'] = outs['action']
        data['reward'] = torch.from_numpy(r).float()
        data['done'] = torch.from_numpy(done)
        data['vpred'] = outs['value']
        for key in outs:
            if key not in ['action', 'value', 'state']:
                data[key] = outs[key]
        self.storage.insert(data)

        def _to_torch(o):
            return torch.from_numpy(o).to(self.device)

        self._ob = nest.map_structure(_to_torch, ob)
        if self.recurrent:
            self._state = outs['state']

    def rollout(self):
        """Compute entire rollout and advantage targets."""
        self._reset()
        while not self.storage.rollout_complete:
            self.rollout_step()
        self.storage.compute_targets(self.gamma, self.lambda_,
                                     norm_advantages=self.norm_advantages)
        return self.storage.rollout_length()

    def sampler(self):
        """Create sampler to iterate over rollout data."""
        return self.storage.sampler(self.batch_size, self.recurrent)


if __name__ == '__main__':
    import unittest
    from dl.rl.modules import Policy, ActorCriticBase
    from dl.rl.envs import make_env
    from dl.modules import FeedForwardNet, Categorical, DiagGaussian
    from gym.spaces import Tuple
    from baselines.common.vec_env import VecEnvWrapper
    from torch.nn.utils.rnn import PackedSequence
    import numpy as np

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
            if isinstance(ob, (list, tuple)):
                ob = ob[0]
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
            self.lstm = torch.nn.LSTM(32, 32, 1)
            self.vf = torch.nn.Linear(32, 1)

        def forward(self, ob, state_in=None):
            """Forward."""
            if isinstance(ob, PackedSequence):
                x = self.net(ob.data.float())
                x = PackedSequence(x, batch_sizes=ob.batch_sizes,
                                   sorted_indices=ob.sorted_indices,
                                   unsorted_indices=ob.unsorted_indices)
            else:
                x = self.net(ob.float()).unsqueeze(0)
            if state_in is None:
                x, state_out = self.lstm(x)
            else:
                x, state_out = self.lstm(x, state_in['lstm'])
            if isinstance(x, PackedSequence):
                print(x)
                x = x.data
            else:
                x = x.squeeze(0)
            state_out = {'lstm': state_out, '1': torch.zeros_like(state_out[0])}
            return self.dist(x), self.vf(x), state_out

    class RolloutActor(object):
        """actor."""

        def __init__(self, pi):
            """init."""
            self.pi = pi

        def __call__(self, ob, state_in=None):
            """act."""
            outs = self.pi(ob, state_in)
            data = {'value': outs.value,
                    'action': outs.action}
            if outs.state_out:
                data['state'] = outs.state_out
            if isinstance(ob, (list, tuple)):
                data['key1'] = torch.zeros_like(ob[0])
            else:
                data['key1'] = torch.zeros_like(ob)
            return data

    class NestedVecObWrapper(VecEnvWrapper):
        """Nest observations."""

        def __init__(self, venv):
            """Init."""
            super().__init__(venv)
            self.observation_space = Tuple([self.observation_space,
                                            self.observation_space])

        def reset(self):
            """Reset."""
            ob = self.venv.reset()
            return (ob, ob)

        def step_wait(self):
            """Step."""
            ob, r, done, info = self.venv.step_wait()
            return (ob, ob), r, done, info

    def test(env, base, batch_size, nested):
        pi = Policy(base(env.observation_space,
                         env.action_space))
        if nested:
            env = NestedVecObWrapper(env)
        nenv = env.num_envs
        data_manager = RolloutDataManager(env, RolloutActor(pi), 'cpu',
                                          batch_size=batch_size)
        for _ in range(3):
            data_manager.rollout()
            count = 0
            for batch in data_manager.sampler():
                assert 'key1' in batch
                count += 1
                assert 'done' in batch
                data_manager.act(batch['obs'])
            if data_manager.recurrent:
                assert count == np.ceil(nenv / data_manager.batch_size)
            else:
                n = data_manager.storage.get_rollout()['reward'].data.shape[0]
                print(n, data_manager.batch_size, count)
                assert count == np.ceil(n / data_manager.batch_size)

    def env_discrete(nenv):
        """Create discrete env."""
        return make_env('CartPole-v1', nenv=nenv)

    def env_continuous(nenv):
        """Create continuous env."""
        return make_env('LunarLanderContinuous-v2', nenv=nenv)

    class TestRolloutDataCollection(unittest.TestCase):
        """Test case."""

        def test_feed_forward(self):
            """Test feed forward network."""
            test(env_discrete(2), FeedForwardBase, 32, False)

        def test_recurrent(self):
            """Test recurrent network."""
            test(env_discrete(2), RNNBase, 2, False)

        def test_feed_forward_nested_ob(self):
            """Test feed forward network."""
            test(env_discrete(2), FeedForwardBase, 32, False)

        def test_recurrent_nested_ob(self):
            """Test recurrent network."""
            test(env_discrete(2), RNNBase, 2, False)

    unittest.main()
