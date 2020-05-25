"""Test the ready-to-use torch module."""

import pytest

import torch

from ebconv.nn import CBSConv


@pytest.mark.parametrize('adaptive_centers', [True, False])
@pytest.mark.parametrize('adaptive_scalings', [True, False])
@pytest.mark.parametrize('k', [-1, 0, 1, 2, 3])
@pytest.mark.parametrize('nc', [None, (20)])
@pytest.mark.parametrize('layout', ['grid', 'random'])
def test_cbconv_module(layout, nc, k, adaptive_centers, adaptive_scalings):
    """Check basic functionality of the module."""
    input_channels = 3
    output_channels = 4

    input_ = torch.rand(1, input_channels, 256, 256)

    def create_module():
        return CBSConv(
            input_channels, output_channels, (20, 20),
            layout=layout, nc=nc, k=k,
            adaptive_centers=adaptive_centers,
            adaptive_scalings=adaptive_scalings)

    if k < 0:
        with pytest.raises(ValueError):
            module = create_module()
        return

    if layout == 'grid' and nc is not None:
        with pytest.warns(UserWarning):
            module = create_module()
    elif layout == 'random' and nc is None:
        with pytest.raises(ValueError):
            module = create_module()
        return
    else:
        module = create_module()

    # Check by default they are all of the same type (float 32)
    for param in module.parameters():
        assert param.dtype == torch.float

    assert module.centers.requires_grad == adaptive_centers
    assert module.scalings.requires_grad == adaptive_scalings
    assert module.weights.requires_grad

    if k == 0 and (adaptive_centers or adaptive_scalings):
        with pytest.warns(UserWarning):
            y = module(input_)
    else:
        y = module(input_)

    assert y.dtype == torch.float
    assert y.requires_grad
