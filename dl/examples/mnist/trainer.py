import dl
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin, time
import numpy as np


@gin.configurable
class MNISTTrainer(dl.Trainer):
    def loss(self, batch):
        data, target = batch
        out = self.model(data)
        return F.nll_loss(out, target)

    def predict(self, batch):
        data, target = batch
        out = self.model(data)
        _,yhat = torch.max(out, dim=1)
        return yhat

    def metrics(self, batch):
        yhat = self.predict(batch)
        return {'accuracy': (yhat == batch[1]).float().mean()}

    def visualization(self, dval):
        confusion_matrix = np.zeros((10,10))
        for batch in self.dval:
            batch = self._batch_to_device(batch)
            yhat = self.predict(batch).cpu().numpy()
            target = batch[1]
            for i,y in enumerate(yhat):
                confusion_matrix[target[i], y] += 1
        s = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = 255 * (1. - confusion_matrix / s)
        confusion_matrix = confusion_matrix.astype(np.uint8)
        dl.logger.add_image('confusion', confusion_matrix[None], self.t, time.time())


@gin.configurable
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
