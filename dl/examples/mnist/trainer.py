"""Defines trainer and network for MNIST."""
import dl
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
import time
import numpy as np


@gin.configurable
class MNISTTrainer(dl.SingleModelTrainer):
    """Trainer for mnist."""

    def before_epoch(self):
        """Log stuff."""
        dl.logger.log(f"Starting epoch {self.t}...")

    def forward(self, batch):
        """Forward."""
        return self.model(batch[0])

    def loss(self, batch, out):
        """Loss."""
        return F.nll_loss(out, batch[1])

    def metrics(self, batch, out):
        """Metrics."""
        _, yhat = torch.max(out, dim=1)
        return {'accuracy': (yhat == batch[1]).float().mean()}

    def visualization(self):
        """Viz."""
        confusion_matrix = np.zeros((10, 10))
        for batch in self.dval:
            out = self.forward(self.batch_to_device(batch))
            _, yhat = torch.max(out, dim=1)
            yhat = yhat.cpu().numpy()
            target = batch[1]
            for i, y in enumerate(yhat):
                confusion_matrix[target[i], y] += 1
        s = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = 255 * (1. - confusion_matrix / s)
        confusion_matrix = confusion_matrix.astype(np.uint8)
        dl.logger.add_image('confusion', confusion_matrix[None], self.t,
                            time.time())


@gin.configurable
class MNISTNet(nn.Module):
    """MNIST convolutional network."""

    def __init__(self):
        """Init."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        """Forward."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
