"""Gin configurable for training models."""
import gin


@gin.configurable(whitelist=['Trainer'])
def train(logdir, Trainer):
    """Train."""
    Trainer(logdir).train()
