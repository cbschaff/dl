import os, json, tempfile, time
import numpy as np
from imageio import imwrite
import shutil
import subprocess as sp
from dl.util import find_monitor, logger
import torch
from dl.eval.rl_plot import plot_results
import matplotlib.pyplot as plt


def rl_evaluate(env, actor, nepisodes, outfile, device='cpu'):
    ep_lengths = []
    ep_rewards = []
    monitor = find_monitor(env)
    for i in range(nepisodes):
        ob = env.reset()
        done = False
        ep_lengths.append(0)
        ep_rewards.append(0)
        while not done:
            ob = torch.from_numpy(ob).to(device)[None]
            action = actor(ob).action.cpu().numpy()
            ob, r, done, _ = env.step(action)
            if monitor is None:
                ep_lengths[-1] += 1
                ep_rewards[-1] += r
            else:
                done = monitor.needs_reset
                if done:
                    ep_lengths[-1] = monitor.episode_lengths[-1]
                    ep_rewards[-1] = monitor.episode_rewards[-1]

    outs = {
        'episode_lengths': ep_lengths,
        'episode_rewards': ep_rewards,
        'mean_length': np.mean(ep_lengths),
        'mean_reward': np.mean(ep_rewards),
    }

    with open(outfile, 'w') as f:
        json.dump(outs, f)

def rl_record(env, actor, nepisodes, outfile, device='cpu', fps=30):
    tmpdir = os.path.join(tempfile.gettempdir(), 'video_' + str(time.monotonic()))
    os.makedirs(tmpdir)
    monitor = find_monitor(env)
    id = 0
    for i in range(nepisodes):
        ob = env.reset()
        done = False
        while not done:
            try:
                rgb = env.render('rgb_array')
            except:
                logger.log("Error while rendering.")
                return
            imwrite(os.path.join(tmpdir, '{:05d}.png'.format(id)), rgb)
            id += 1

            ob = torch.from_numpy(ob).to(device)[None]
            action = actor(ob).action.cpu().numpy()
            ob, r, done, _ = env.step(action)
            if monitor:
                done = monitor.needs_reset

    sp.call(['ffmpeg', '-r', str(fps), '-f', 'image2', '-i', os.path.join(tmpdir, '%05d.png'), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(tmpdir, 'out.mp4')])
    sp.call(['mv', os.path.join(tmpdir, 'out.mp4'), outfile])
    sp.call(['rm', '-rf', tmpdir])


def rl_plot(logdir, title, t):
    logdir = os.path.abspath(logdir)
    try:
        plot_results([logdir], t, 'timesteps', title)
        plt.savefig(os.path.join(logdir, 'plot_steps.pdf'))
        plot_results([logdir], t, 'episodes', title)
        plt.savefig(os.path.join(logdir, 'plot_episodes.pdf'))
        plot_results([logdir], t, 'walltime_hrs', title)
        plt.savefig(os.path.join(logdir, 'plot_time.pdf'))
    except:
        pass
