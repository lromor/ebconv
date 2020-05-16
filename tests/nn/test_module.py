"""Test the ready-to-use torch module."""

import pytest

import torch

from ebconv.nn import CBSConv


@pytest.mark.parametrize('k', [0, 1, 2, 3])
def test_cbconv_module(k):
    """Check basic functionality of the module.

    By default torch modules run with 32-bit floating point weights.
    """
    input_ = torch.rand(1, 3, 256, 256)

    if k < 1:
        with pytest.raises(ValueError):
            module = CBSConv(
                3, 1, (20, 20), 'random', 8, k, padding=2, bias=True)
        return
    module = CBSConv(3, 1, (20, 20), 'random', 8, k, padding=2, bias=True)

    for param in module.parameters():
        assert param.dtype == torch.float

    y = module(input_)
    assert y.dtype == torch.float
    assert y.requires_grad
