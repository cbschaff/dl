"""Extends dataset trainer for the case with a single model."""
import torch
from dl.trainers import DatasetTrainer
from dl import logger
import gin
import time
import numpy as np


@gin.configurable(blacklist=['logdir'])
class SingleModelTrainer(DatasetTrainer):
    """This class extends DatasetTrainer to the case with a single model/opt.

    It provides functionality for updating and evaluating
    the model, as well as logs loss and metrics to TensorBoard.
    Subclasses of this class are expected to provide code for:
        1) Running the model
        2) Computing the loss
        3) Computing metrics
        4) Doing any additional visualization
    """

    def __init__(self,
                 logdir,
                 model,
                 opt,
                 dataset_train,
                 dataset_val=None,
                 **kwargs):
        """Init."""
        super().__init__(logdir, dataset_train, dataset_val, **kwargs)
        self.model = model.to(self.device)
        self.opt = opt(model.parameters())

    def _handle_loss(self, loss):
            if isinstance(loss, torch.Tensor):
                logger.add_scalar(
                    'train_loss/total',
                    loss.detach().cpu().numpy(), self.nsamples, time.time())
                return loss
            else:
                assert isinstance(loss, dict), (
                       "The return value of loss() must be a Tensor or a dict")
                for k in loss:
                    logger.add_scalar(
                        f'train_loss/{k}',
                        loss[k].detach().cpu().numpy(), self.nsamples,
                        time.time())
                return loss['total']

    def state_dict(self):
        """State dict."""
        return {
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.model.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['opt'])

    def update(self, batch):
        """Update model."""
        self.model.train()
        self.opt.zero_grad()
        out = self.forward(batch)
        loss = self._handle_loss(self.loss(batch, out))
        loss.backward()
        self.opt.step()

    def evaluate(self):
        """Evaluate model."""
        self.model.eval()
        if self.dval is None:
            # Only run vis script if no validation set is provided
            with torch.no_grad():
                self.visualization()
            return

        def _append_to_dict(d, x):
            for k in x:
                if k not in d:
                    d[k] = []
                d[k].append(x[k].cpu().numpy())

        losses = {'total': []}
        metrics = {}
        with torch.no_grad():
            for batch in self.dval:
                batch = self.batch_to_device(batch)
                out = self.forward(batch)
                loss = self.loss(batch, out)
                if isinstance(loss, torch.Tensor):
                    losses['total'].append(loss.cpu().numpy())
                else:
                    _append_to_dict(losses, loss)
                _append_to_dict(metrics, self.metrics(batch, out))
            self.visualization()

        for k in losses:
            avg = np.mean(losses[k])
            logger.add_scalar(f'val_loss/{k}', avg, self.t, time.time())
        for k in metrics:
            avg = np.mean(metrics[k])
            logger.add_scalar(f'val_metrics/{k}', avg, self.t, time.time())

    def forward(self, batch):
        """Compute model outputs for a given minibatch.

        Args:
            batch: a minibatch from the training dataset
        Returns:
            model output.

        """
        raise NotImplementedError

    def loss(self, batch, model_out):
        """Compute the loss for a given minibatch.

        Args:
            batch: a minibatch from the training dataset
            model_out: the return value of Trainer.forward
        Returns:
            A scalar tensor for the loss
            OR
            A dict of scalar tensors containing the key "total" which will be
            used for backpropegation. Other keys will be logged to tensorboard.

        """
        raise NotImplementedError

    def metrics(self, batch, model_out):
        """Compute scalar metrics for a given minibatch. (i.e. accuracy)
        Args:
            batch: a minibatch from the validation dataset
            model_out: the return value of Trainer.forward
        Returns:
            A dict of scalar tensors.

        """
        return {}

    def visualization(self):
        """Visualize model.

        This function is for arbitrary visualization. It is called during each
        evaluation.
        Args:
            dval: DataLoader for the validation dataset.
        Returns:
            Return value is unused. Any logging must be done in this function.

        """
        pass


if __name__ == '__main__':
    import unittest
    import shutil
    from torch.utils.data import Dataset

    class TestTrainer(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            class T(SingleModelTrainer):
                def forward(self, batch):
                    return self.model(batch['x'])

                def loss(self, batch, out):
                    assert out.shape == (4, 10)
                    assert batch['y'].shape == (4,)
                    l1 = torch.nn.CrossEntropyLoss()(out, batch['y'])
                    return {'total': 3*l1, 'l1': l1, 'l2': 2*l1}

                def metrics(self, batch, out):
                    _, yhat = out.max(dim=1)
                    acc = (yhat == batch['y'])
                    return {'accuracy': acc}

                def visualization(self):
                    img = torch.from_numpy(
                            255 * np.random.rand(3, 10, 10)).byte()
                    logger.add_image('viz1', img, self.t, time.time())

            class D(Dataset):
                def __len__(self):
                    return 100

                def __getitem__(self, ind):
                    x = torch.from_numpy(np.array([ind])).float()
                    y = torch.from_numpy(np.array(ind)).long() // 10
                    assert x.shape == (1,) and y.shape == ()
                    return {'x': x, 'y': y}

            from dl.modules import FeedForwardNet
            model = FeedForwardNet(1, [32, 32, 10])
            dtrain = D()
            dval = D()

            trainer = T('test',
                        model,
                        torch.optim.Adam,
                        dtrain,
                        dval,
                        batch_size=4,
                        eval=True,
                        eval_period=1,
                        maxt=10)
            trainer.train()

            shutil.rmtree('test')

    unittest.main()
