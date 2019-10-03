"""Miscellaneous modules."""
import torch.nn as nn


class TimeAndBatchUnflattener(nn.Module):
    """Flatten and unflatten time and batch dimensions."""

    def __init__(self, batch_first=False):
        """Init."""
        super().__init__()
        self.batch_first = batch_first

    def forward(self, x, hx=None):
        """Forward."""
        return self.unflatten(x, hx)

    def unflatten(self, x, hx=None):
        """Unflatten."""
        nt = x.shape[0]
        if hx is None:
            n, t = nt, 1
        else:
            n = hx.shape[-2]
            t = nt // n
        if self.batch_first:
            return x.view(n, t, *x.shape[1:])
        else:
            return x.view(t, n, *x.shape[1:])

    def flatten(self, x):
        """Flatten."""
        return x.view(-1, *x.shape[2:])


if __name__ == '__main__':
    import torch
    import unittest

    class TestMisc(unittest.TestCase):
        """Test."""

        def test_tbf(self):
            """Test."""
            x = torch.zeros(10, 5, 3)
            hx = torch.zeros(2, 1)
            tbf = TimeAndBatchUnflattener(batch_first=False)
            assert tbf(x).shape == (1, 10, 5, 3)
            assert tbf(x, hx).shape == (5, 2, 5, 3)
            assert tbf.flatten(tbf(x, hx)).shape == x.shape
            assert torch.allclose(tbf.flatten(tbf(x, hx)), x)

            tbf = TimeAndBatchUnflattener(batch_first=True)
            assert tbf(x).shape == (10, 1, 5, 3)
            assert tbf(x, hx).shape == (2, 5, 5, 3)
            assert tbf.flatten(tbf(x, hx)).shape == x.shape
            assert torch.allclose(tbf.flatten(tbf(x, hx)), x)

    unittest.main()
