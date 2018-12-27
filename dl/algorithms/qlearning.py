from dl.modules import QFunction
from dl.util import ReplayBuffer, PrioritizedReplayBuffer, Checkpointer
from baselines.common.schedules import LinearSchedule
import gin


@gin.configurable(blacklist=['logdir', 'env'])
class QLearning(object):
    def __init__(self,
                 logdir,
                 env,
                 optimizer,
                 gamma=0.99,
                 batch_size=32,
                 update_freq=4,
                 frame_stack=4,
                 huber_loss=True,
                 learning_starts=50000,
                 exploration_timesteps=1000000,
                 final_eps=0.1,
                 target_update_freq=10000,
                 double_dqn=False,
                 buffer=ReplayBuffer,
                 buffer_size=1000000,
                 prioritized_replay=False,
                 replay_alpha=0.6,
                 replay_beta=0.4,
                 t_beta_max=int(1e7),
                 gpu=True,
                 ):

        self.logdir = logdir
        self.env = env
        self.gamma = gamma
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay
        self.buffer = buffer(buffer_size, frame_stack)
        if prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(self.buffer, alpha=replay_alpha)
            self.beta_schedule = LinearSchedule(t_beta_max, 1.0, replay_beta)
        self.eps_schedule = LinearSchedule(exploration_timesteps, final_eps, 1.0)

        s = env.observation_space.shape
        ob_shape = (s[0] * self.frame_stack, *s[1:])
        self.net = QFunction(ob_shape, env.action_space)
        self.target_net = QFunction(ob_shape, env.action_space)
        device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        self.net.to(device)
        self.target_net.to(device)
        self.opt = optimizer(self.net.parameters())
        self.ckptr = Checkpointer(self.logdir)
        self.t = 0

    def save(self):
        state_dict = {
            'net': self.net.state_dict(),
            'opt': self.opt.state_dict(),
            't':   self.t,
            'buffer': self.buffer.state_dict()
        }
        self.ckptr.save(state_dict, self.t)

    def load(self, t=None):
        state_dict = self.ckptr.load(t)
        self.net.load_state_dict(state_dict['net'])
        sel.target_net.load_state_dict(state_dict['net'])
        self.opt.load_state_dict(state_dict['opt'])
        self.buffer.load_state_dict(state_dict['buffer'])
        self.t = state_dict['t']
