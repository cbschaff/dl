import torch
from dl import BaseTrainer
from dl.util import StatefulSampler
from torch.utils.data import DataLoader
from dl import logger
import gin, time
import numpy as np


@gin.configurable(blacklist=['logdir'])
class Trainer(BaseTrainer):
    def __init__(self,
                 logdir,
                 model,
                 opt,
                 dataset_train,
                 dataset_val=None,
                 batch_size=1,
                 shuffle=True,
                 num_workers=4,
                 gpu=True,
                 **trainer_kwargs
                ):
        super().__init__(logdir, **trainer_kwargs)
        self.model = model
        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.opt = opt(self.model.parameters())
        self._dsize = len(dataset_train)
        self.batch_size = batch_size
        self._sampler = StatefulSampler(dataset_train, shuffle=shuffle)
        self.dtrain = DataLoader(dataset_train, sampler=self._sampler, batch_size=batch_size, num_workers=num_workers)
        if dataset_val is None:
            self.dval = None
        else:
            self.dval = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        self._diter = None
        self._nsamples = 0

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'sampler': self._sampler.state_dict(self._diter),
            't': self.t,
            'nsamples': self._nsamples,
        }

    def load_state_dict(self, state_dict):
        if self._diter is not None:
            self._diter.__del__()
            self._diter = None
        self.model.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['opt'])
        self._sampler.load_state_dict(state_dict['sampler'])
        self.t = state_dict['t']
        self._nsamples = state_dict['nsamples']

    def step(self):
        self.model.train()
        if self._diter is None:
            self.before_epoch()
            self._diter = self.dtrain.__iter__()
        try:
            batch = self._diter.__next__()
        except StopIteration:
            self.t += 1
            self._diter = None
            return
        self.opt.zero_grad()
        batch = self.batch_to_device(batch)
        loss = self._handle_loss(self.loss(batch, self.forward(batch)))
        loss.backward()
        self.opt.step()
        self._nsamples += min(self._dsize - (self._nsamples % self._dsize), self.batch_size)

    def _handle_loss(self, loss):
        if isinstance(loss, torch.Tensor):
            logger.add_scalar('train_loss/total', loss.detach().cpu().numpy(), self._nsamples, time.time())
            return loss
        else:
            assert isinstance(loss, dict), "The return value of loss() must be a Tensor or a dict"
            for k in loss:
                logger.add_scalar(f'train_loss/{k}', loss[k].detach().cpu().numpy(), self._nsamples, time.time())
            return loss['total']

    def batch_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self.batch_to_device(v) for k,v in batch.items()}
        else:
            return [self.batch_to_device(b) for b in batch]

    def evaluate(self):
        if self.dval is None:
            return

        def _append_to_dict(d, x):
            for k in x:
                if k not in d:
                    d[k] = []
                d[k].append(x[k].cpu().numpy())

        self.model.eval()
        losses = {'total':[]}
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
            self.visualization(self.dval)

        for k in losses:
            avg = np.mean(losses[k])
            logger.add_scalar(f'val_loss/{k}', avg, self.t, time.time())
        for k in metrics:
            avg = np.mean(metrics[k])
            logger.add_scalar(f'val_metrics/{k}', avg, self.t, time.time())

    def close(self):
        if self._diter is not None:
            self._diter.__del__()
            self._diter = None


    def before_epoch(self):
        """
        Called before the start of each epoch.
        """
        pass

    def forward(self, batch):
        """
        Computes model outputs for a given minibatch.
        Args:
            batch: a minibatch from the training dataset
        Returns:
            model outputs
        """
        raise NotImplementedError

    def loss(self, batch, model_out):
        """
        Computes the loss for a given minibatch.
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
        """
        Computes scalar metrics for a given minibatch. (i.e. accuracy)
        Args:
            batch: a minibatch from the validation dataset
            model_out: the return value of Trainer.forward
        Returns:
            A dict of scalar tensors.
        """
        return {}

    def visualization(self, dval):
        """
        A place for aribtrary visualization. This will be called
        during each evaluation.
        Args:
            dval: DataLoader for the validation dataset
        Returns:
            Return value is unused. Any logging must be done in this function
        """
        pass



if __name__ == '__main__':

    import unittest, shutil
    from torch.utils.data import Dataset
    import numpy as np

    class TestTrainer(unittest.TestCase):
        def test(self):
            class T(Trainer):
                def forward(self, batch):
                    return self.model(batch['x'])

                def loss(self, batch, out):
                    _l = torch.nn.CrossEntropyLoss()
                    assert out.shape == (4,10)
                    assert batch['y'].shape == (4,)
                    l1 = _l(out, batch['y'])
                    return {'total': 3*l1, 'l1': l1, 'l2': 2*l1}

                def metrics(self, batch, out):
                    _, yhat = out.max(dim=1)
                    acc = (yhat == batch['y'])
                    return {'accuracy': acc}

                def visualization(self, dval):
                    img = torch.from_numpy(255 * np.random.rand(3,10,10)).byte()
                    logger.add_image('viz1', img, self.t, time.time())


            class D(Dataset):
                def __len__(self):
                    return 100
                def __getitem__(self, ind):
                    x = torch.from_numpy(np.array([ind])).float()
                    y = torch.from_numpy(np.array(ind)).long() // 10
                    assert x.shape == (1,) and y.shape == ()
                    return {'x':x, 'y':y}

            from dl.modules import FeedForwardNet
            model = FeedForwardNet(1, [32,32,10])
            dtrain = D()
            dval = D()

            trainer = T('test', model, torch.optim.Adam, dtrain, dval, 4, eval=True, eval_period=1, maxt=10)
            trainer.train()

            shutil.rmtree('test')

    unittest.main()
