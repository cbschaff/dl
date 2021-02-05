# dl
Deep Reinforcement Learning library based on [PyTorch](https://pytorch.org/), [OpenAI Gym](https://gym.openai.com/), and
[gin config](https://github.com/google/gin-config), and [Tensorboard](https://www.tensorflow.org/tensorboard)
(for visualization and logging).
The library is designed for research and fast iteration.
It is highly modular to speed up the implementation of new algorithms and shorten iteration cycles.

Some key abstractions include:

* Flexible data collection and storage for on-policy (rollouts) and off-policy (replay buffer) methods.
* Code for evaluating and recording trained agents, as well as checkpointing and logging experiments.
* Abstract interface for algorithms with a training loop suitable for both RL and supervised learning.


### Implemented algorithms

* Two versions of [PPO](https://arxiv.org/abs/1707.06347): One with a clipped objective and one with an adaptive KL penalty.
* [DQN](https://www.nature.com/articles/nature14236) with [Double Q Learning](https://arxiv.org/abs/1509.06461) and
[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
* [DDPG](https://arxiv.org/abs/1509.02971)
* [TD3](https://arxiv.org/abs/1802.09477)
* [SAC](https://arxiv.org/abs/1801.01290)
* [Alpha Zero](https://science.sciencemag.org/content/362/6419/1140.full?ijkey=XGd77kI6W4rSc&keytype=ref&siteid=sci)

Examples of how to launch experiments can be found [here](https://github.com/cbschaff/dl/tree/master/examples).

### Installation

The code can be installed using docker or using pip.

Pip:
1. In the top level directory, run ```pip install -e .```

Docker:
1. Install [docker](https://docs.docker.com/get-docker/).
1. Install [x-docker](https://github.com/afdaniele/x-docker), a wrapper around docker for running GUI applications inside a container.
3. In the top level directory, build the docker image by running:
    ```./build_docker.sh```
4. Launch the docker container by running:
    ```./launch_docker.sh```
    This will start a container and mount the code at /root/pkgs/dl.
  

### Running Examples

1. From inside the container, run:
    ```cd /root/pkgs/dl/examples/ppo```
    (You can replace ppo with another exapmle algorithm)
2. Run:
    ```./train.sh```
    This will create a log directory and start training with the default environment and hyperparameters.
    Pressing Ctrl-C will interrupt trianing and save the current model.
3. In another terminal, run:
    ```tensorboard --logdir /path/to/log/directory```
