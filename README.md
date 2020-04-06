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
