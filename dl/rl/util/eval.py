"""Evaluation for RL Environments."""
import os
import json
import tempfile
import time
import numpy as np
from imageio import imwrite
import subprocess as sp
from dl import logger
from dl import nest
from dl.rl.util import ensure_vec_env
import torch


class Actor(object):
    """Wrap actor to convert actions to numpy."""

    def __init__(self, net, device):
        """Init."""
        self.net = net
        self.state = None
        self.device = device

    def __call__(self, ob, dones=None):
        """__call__."""
        def _to_torch(o):
            return torch.from_numpy(o).to(self.device)
        ob = nest.map_structure(_to_torch, ob)
        if dones is None:
            batch_size = nest.flatten(ob)[0].shape[0]
            mask = torch.zeros([batch_size]).to(self.device).float()
        else:
            mask = torch.from_numpy(1. - dones).to(self.device).float()
        if self.state is None:
            out = self.net(ob)
        else:
            out = self.net(ob, self.state, mask)
        if hasattr(out, 'state_out'):
            self.state = out.state_out
        return out.action.cpu().numpy()


def rl_evaluate(env, actor, nepisodes, outfile=None, device='cpu'):
    """Compute episode stats for an environment and actor.

    If the environment has an EpisodeInfo Wrapper, rl_record will use that
    to determine episode termination.
    Args:
        env: A Gym environment
        actor: A torch.nn.Module whose input is an observation and output has a
               '.action' attribute.
        nepisodes: The number of episodes to run.
        outfile: Where to save results (if provided)
        device: The device which contains the actor.
    Returns:
        A dict of episode stats

    """
    env = ensure_vec_env(env)
    ep_lengths = []
    ep_rewards = []
    obs = env.reset()
    lengths = np.zeros(env.num_envs, dtype=np.int32)
    rewards = np.zeros(env.num_envs, dtype=np.float32)
    dones = None
    actor = Actor(actor, device)
    while len(ep_lengths) < nepisodes:
        obs, rs, dones, infos = env.step(actor(obs, dones))
        rewards += rs
        lengths += 1
        for i, done in enumerate(dones):
            if 'episode_info' in infos[i]:
                if infos[i]['episode_info']['done']:
                    ep_lengths.append(infos[i]['episode_info']['length'])
                    ep_rewards.append(infos[i]['episode_info']['reward'])
                    lengths[i] = 0
                    rewards[i] = 0.
                dones[i] = infos[i]['episode_info']['done']
            elif done:
                ep_lengths.append(int(lengths[i]))
                ep_rewards.append(float(rewards[i]))
                lengths[i] = 0
                rewards[i] = 0.

    outs = {
        'episode_lengths': ep_lengths,
        'episode_rewards': ep_rewards,
        'mean_length': np.mean(ep_lengths),
        'mean_reward': np.mean(ep_rewards),
    }
    if outfile:
        with open(outfile, 'w') as f:
            json.dump(outs, f)
    return outs


def rl_record(env, actor, nepisodes, outfile, device='cpu', fps=30):
    """Compute episode stats for an environment and actor.

    If the environment has an EpisodeInfo Wrapper, rl_record will use that to
    determine episode termination.
    Args:
        env: A Gym environment
        actor: A callable whose input is an observation and output has a
               '.action' attribute.
        nepisodes: The number of episodes to run.
        outfile: Where to save the video.
        device: The device which contains the actor.
        fps: The frame rate of the video.
    Returns:
        A dict of episode stats

    """
    env = ensure_vec_env(env)
    tmpdir = os.path.join(tempfile.gettempdir(),
                          'video_' + str(time.monotonic()))
    os.makedirs(tmpdir)
    id = 0
    actor = Actor(actor, device)
    episodes = 0
    obs = env.reset()
    dones = None
    ims = None

    def write_ims(ims, id):
        for im in ims:
            imwrite(os.path.join(tmpdir, '{:05d}.png'.format(id)), im)
            id += 1
        return id

    while episodes < nepisodes:
        try:
            rgbs = env.get_images()
        except Exception:
            logger.log("Error while rendering.")
            return
        if not ims:
            ims = [[] for rgb in rgbs]
        for i, rgb in enumerate(rgbs):
            ims[i].append(rgb)
        _, _, dones, infos = env.step(actor(obs, dones))
        for i, done in enumerate(dones):
            if 'episode_info' in infos[i]:
                if infos[i]['episode_info']['done']:
                    id = write_ims(ims[i], id)
                    ims[i] = []
                    episodes += 1
                dones[i] = infos[i]['episode_info']['done']
            elif done:
                id = write_ims(ims[i], id)
                ims[i] = []
                episodes += 1

    sp.call(['ffmpeg', '-r', str(fps), '-f', 'image2', '-i',
             os.path.join(tmpdir, '%05d.png'), '-vcodec', 'libx264',
             '-pix_fmt', 'yuv420p', os.path.join(tmpdir, 'out.mp4')])
    sp.call(['mv', os.path.join(tmpdir, 'out.mp4'), outfile])
    sp.call(['rm', '-rf', tmpdir])


if __name__ == '__main__':
    import unittest
    import gym
    from dl.rl.envs import EpisodeInfo
    from collections import namedtuple

    class Test(unittest.TestCase):
        """Test."""

        def test_eval(self):
            """Test."""
            env = EpisodeInfo(gym.make('CartPole-v1'))

            def actor(ob):
                ac = torch.from_numpy(np.array(env.action_space.sample()))[None]
                return namedtuple('test', ['action', 'state_out'])(
                    action=ac, state_out=None)

            stats = rl_evaluate(env, actor, 10, outfile='./out.json')
            assert len(stats['episode_lengths']) >= 10
            assert len(stats['episode_rewards']) >= 10
            assert len(stats['episode_rewards']) == len(
                    stats['episode_lengths'])
            assert np.mean(stats['episode_lengths']) == stats['mean_length']
            assert np.mean(stats['episode_rewards']) == stats['mean_reward']
            env.close()
            os.remove('./out.json')

        def test_record(self):
            """Test record."""
            env = EpisodeInfo(gym.make('CartPole-v1'))

            def actor(ob):
                ac = torch.from_numpy(np.array(env.action_space.sample()))[None]
                return namedtuple('test', ['action', 'state_out'])(
                    action=ac, state_out=None)

            rl_record(env, actor, 10, './video.mp4')
            os.remove('./video.mp4')
            env.close()

    unittest.main()
