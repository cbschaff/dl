"""Add masking of hidden states to RNN Modules.

Implementation loosely based on
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/model.py
"""
import torch
import torch.nn as nn


class MaskedRecurrentModule(nn.Module):
    """Recurrent Module with masked hidden states."""

    def _get_masked_timesteps(self, mask):
        # Let's figure out which steps in the sequence have a zero for any agent
        # We will always assume t=0 has a zero in it as that makes the logic
        # cleaner.
        has_zeros = (mask[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        T = mask.shape[0]
        has_zeros = [0] + has_zeros + [T]
        return has_zeros

    def forward(self, x, hx=None, mask=None):
        """Forward."""
        # Check inputs
        # x is a (T, N, -1) tensor
        assert len(x.shape) == 3
        T, N = x.shape[0], x.shape[1]

        # If no hidden state is provided:
        if hx is None:
            assert mask is None
            return self._module(self, x)

        expects_tensor = isinstance(hx, torch.Tensor)
        # assume batch dim is the second to last.
        for hxi in hx:
            assert hxi.shape[-2] == N

        assert mask is not None, 'A mask must be provided with the state.'
        # m is a (T,N) tensor
        assert mask.shape == (T, N)
        # If multiple timesteps, group steps by mask for speed.
        masked_t = self._get_masked_timesteps(mask)
        # Make the batch dimension in mask line up with hx.
        mask = mask.unsqueeze(-1)
        outputs = []
        for i in range(len(masked_t) - 1):
            # We can now process steps that don't have any zeros in masks
            # together! This is much faster
            start, end = masked_t[i], masked_t[i+1]
            if expects_tensor:
                out, hx = self._module(self, x[start:end], hx * mask[start])
            else:
                masked_hx = tuple([hxi * mask[start] for hxi in hx])
                out, hx = self._module(self, x[start:end], masked_hx)
            outputs.append(out)
        out = torch.cat(outputs, dim=0)
        assert out.shape[:2] == (T, N)
        return out, hx


class MaskedRNN(MaskedRecurrentModule, nn.RNN):
    """Masked vanilla rnn."""

    def __init__(self, *args, **kwargs):
        """Init."""
        nn.RNN.__init__(self, *args, **kwargs)
        self._module = nn.RNN.forward
        assert not self.batch_first, (
            "Time dimension must be first for masked recurrent modules.")


class MaskedLSTM(MaskedRecurrentModule, nn.LSTM):
    """Masked LSTM."""

    def __init__(self, *args, **kwargs):
        """Init."""
        nn.LSTM.__init__(self, *args, **kwargs)
        self._module = nn.LSTM.forward
        assert not self.batch_first, (
            "Time dimension must be first for masked recurrent modules.")


class MaskedGRU(MaskedRecurrentModule, nn.GRU):
    """Masked GRU."""

    def __init__(self, *args, **kwargs):
        """Init."""
        nn.GRU.__init__(self, *args, **kwargs)
        self._module = nn.GRU.forward
        assert not self.batch_first, (
            "Time dimension must be first for masked recurrent modules.")


if __name__ == '__main__':
    import unittest

    class TestMaskedRecurrentModules(unittest.TestCase):
        """Test."""

        def test_rnn(self):
            """Test RNN."""
            rnn = nn.RNN(5, 10, 2)
            mrnn = MaskedRNN(5, 10, 2)
            mrnn.load_state_dict(rnn.state_dict())
            x = torch.ones([5, 3, 5])
            x1, hx1 = rnn(x)
            x2, hx2 = mrnn(x)
            assert torch.allclose(x1, x2)
            assert torch.allclose(hx1, hx2)

            mask = torch.Tensor([[0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1],
                                 [1, 1, 0]])
            x1, hx1 = rnn(x, hx1)
            x2, hx2 = mrnn(x, hx2, mask)
            for i in range(5):
                assert not torch.allclose(x1[i, 0], x2[i, 0])

            assert torch.allclose(x1[0, 1], x2[0, 1])
            assert torch.allclose(x1[1, 1], x2[1, 1])
            assert not torch.allclose(x1[2, 1], x2[2, 1])
            assert not torch.allclose(x1[3, 1], x2[3, 1])
            assert not torch.allclose(x1[4, 1], x2[4, 1])

            assert torch.allclose(x1[0, 2], x2[0, 2])
            assert torch.allclose(x1[1, 2], x2[1, 2])
            assert torch.allclose(x1[2, 2], x2[2, 2])
            assert torch.allclose(x1[3, 2], x2[3, 2])
            assert not torch.allclose(x1[4, 2], x2[4, 2])

            for i in range(3):
                assert not torch.allclose(hx1[:, i, :], hx2[:, i, :])

        def test_gru(self):
            """Test GRU."""
            rnn = nn.GRU(5, 10, 2)
            mrnn = MaskedGRU(5, 10, 2)
            mrnn.load_state_dict(rnn.state_dict())
            x = torch.ones([5, 3, 5])
            x1, hx1 = rnn(x)
            x2, hx2 = mrnn(x)
            assert torch.allclose(x1, x2)
            assert torch.allclose(hx1, hx2)

            mask = torch.Tensor([[0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1],
                                 [1, 1, 0]])
            x1, hx1 = rnn(x, hx1)
            x2, hx2 = mrnn(x, hx2, mask)
            for i in range(5):
                assert not torch.allclose(x1[i, 0], x2[i, 0])

            assert torch.allclose(x1[0, 1], x2[0, 1])
            assert torch.allclose(x1[1, 1], x2[1, 1])
            assert not torch.allclose(x1[2, 1], x2[2, 1])
            assert not torch.allclose(x1[3, 1], x2[3, 1])
            assert not torch.allclose(x1[4, 1], x2[4, 1])

            assert torch.allclose(x1[0, 2], x2[0, 2])
            assert torch.allclose(x1[1, 2], x2[1, 2])
            assert torch.allclose(x1[2, 2], x2[2, 2])
            assert torch.allclose(x1[3, 2], x2[3, 2])
            assert not torch.allclose(x1[4, 2], x2[4, 2])

            for i in range(3):
                assert not torch.allclose(hx1[:, i, :], hx2[:, i, :])

        def test_lstm(self):
            """Test LSTM."""
            rnn = nn.LSTM(5, 10, 2)
            mrnn = MaskedLSTM(5, 10, 2)
            mrnn.load_state_dict(rnn.state_dict())
            x = torch.ones([5, 3, 5])
            x1, hx1 = rnn(x)
            x2, hx2 = mrnn(x)
            assert torch.allclose(x1, x2)
            assert torch.allclose(hx1[0], hx2[0])
            assert torch.allclose(hx1[1], hx2[1])

            mask = torch.Tensor([[0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1],
                                 [1, 1, 0]])
            x1, hx1 = rnn(x, hx1)
            x2, hx2 = mrnn(x, hx2, mask)
            for i in range(5):
                assert not torch.allclose(x1[i, 0], x2[i, 0])

            assert torch.allclose(x1[0, 1], x2[0, 1])
            assert torch.allclose(x1[1, 1], x2[1, 1])
            assert not torch.allclose(x1[2, 1], x2[2, 1])
            assert not torch.allclose(x1[3, 1], x2[3, 1])
            assert not torch.allclose(x1[4, 1], x2[4, 1])

            assert torch.allclose(x1[0, 2], x2[0, 2])
            assert torch.allclose(x1[1, 2], x2[1, 2])
            assert torch.allclose(x1[2, 2], x2[2, 2])
            assert torch.allclose(x1[3, 2], x2[3, 2])
            assert not torch.allclose(x1[4, 2], x2[4, 2])

            for i in range(3):
                assert not torch.allclose(hx1[0][:, i, :], hx2[0][:, i, :])
                assert not torch.allclose(hx1[1][:, i, :], hx2[1][:, i, :])

    unittest.main()
