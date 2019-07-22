import gin

@gin.configurable(whitelist=['Trainer'])
def train(logdir, Trainer):
    Trainer(logdir).train()
