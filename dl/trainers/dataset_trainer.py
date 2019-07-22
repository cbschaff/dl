import torch, gin
from dl.trainers import Trainer
from dl.util import StatefulSampler
from torch.utils.data import DataLoader

@gin.configurable(blacklist=['logdir'])
class DatasetTrainer(Trainer):
    """
    This class extends Trainer with dataset and dataloading functionality.
    It handles minibatching across an epoch, saving and loading epoch progress,
    and moving minibatches to a specified device.
    """
    def __init__(self,
                 logdir,
                 dataset_train,
                 dataset_val=None,
                 batch_size=1,
                 shuffle=True,
                 num_workers=1,
                 **trainer_kwargs
                ):
        super().__init__(logdir, **trainer_kwargs)
        self._dsize = len(dataset_train)
        self.batch_size = batch_size
        self._sampler = StatefulSampler(dataset_train, shuffle=shuffle)
        self.dtrain = DataLoader(dataset_train, sampler=self._sampler, batch_size=batch_size, num_workers=num_workers)
        if dataset_val is None:
            self.dval = None
        else:
            self.dval = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        self._diter = None
        self.nsamples = 0

    def _save(self, state_dict):
        assert '_sampler' not in state_dict, "'_sampler' key is used to save epoch progress. Please change your key."
        assert '_nsamples' not in state_dict, "'_nsamples' key is used to save epoch progress. Please change your key."
        state_dict['_sampler'] = self._sampler.state_dict(self._diter)
        state_dict['_nsamples'] = self.nsamples
        super()._save(state_dict)


    def _load(self, state_dict):
        if self._diter is not None:
            self._diter.__del__()
            self._diter = None
        self._sampler.load_state_dict(state_dict['_sampler'])
        self.nsamples = state_dict['_nsamples']
        super()._load(state_dict)


    def step(self):
        if self._diter is None:
            self.before_epoch()
            self._diter = self.dtrain.__iter__()
        try:
            batch = self._diter.__next__()
        except StopIteration:
            self.t += 1
            self._diter = None
            return
        self.update(self.batch_to_device(batch))
        self.nsamples += min(self._dsize - (self.nsamples % self._dsize), self.batch_size)


    def batch_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self.batch_to_device(v) for k,v in batch.items()}
        else:
            return [self.batch_to_device(b) for b in batch]


    def close(self):
        if self._diter is not None:
            self._diter.__del__()
            self._diter = None


    def state_dict(self):
        """
        Implement in Subclasses
        """
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        """
        Implement in Subclasses
        """
        raise NotImplementedError

    def before_epoch(self):
        """
        Called before the start of each epoch.
        """
        pass

    def update(self, batch):
        """
        Update a model with a minibatch from the training set.
        (i.e. compute loss, gradients, and do parameter updates)
        Args:
            batch: A minibatch from the training set. If using a gpu,
                   the batch will be moved to the gpu before
                   being passed to this function.
        """
        raise NotImplementedError




if __name__ == '__main__':

    import unittest, shutil
    from torch.utils.data import Dataset
    import numpy as np

    class TestDatasetTrainer(unittest.TestCase):
        def test(self):
            class T(DatasetTrainer):
                def update(self, batch):
                    assert self.nsamples % self._dsize == batch['x']

                def evaluate(self):
                    assert self.t == self.nsamples / self._dsize

                def state_dict(self):
                    return {}

                def load_state_dict(self, state_dict):
                    pass


            class D(Dataset):
                def __len__(self):
                    return 100
                def __getitem__(self, ind):
                    x = torch.from_numpy(np.array([ind])).float()
                    y = torch.from_numpy(np.array(ind)).long() // 10
                    assert x.shape == (1,) and y.shape == ()
                    return {'x':x, 'y':y}

            dtrain = D()
            dval = D()

            trainer = T('test', dtrain, dval, 1, shuffle=False, eval=True, eval_period=1, maxt=10)
            trainer.train()
            trainer = T('test', dtrain, dval, 1, shuffle=False, eval=True, eval_period=1, maxt=20)
            trainer.train()

            shutil.rmtree('test')

    unittest.main()
