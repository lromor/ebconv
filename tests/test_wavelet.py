"""Test wavelet module."""
import torch

from ebconv.wavelet import ost2d


def test_ost2d():
    """Test the output shape of ost2d."""
    input_ = torch.empty(5, 3, 64, 64)

    # Perform orientation score transform at 20 different angles
    # We expect as output a shape of #images, ost2d angle resolution, H, W
    values = ost2d(input_, 20, 10, 500, 3)
    assert values.shape == (5, 3, 20, 64, 64)
