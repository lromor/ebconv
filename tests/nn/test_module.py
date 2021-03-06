"""Test the ready-to-use torch module."""

import pytest

import torch

from ebconv.nn import CBSConv
from ebconv.nn import CBSConv2d


@pytest.mark.parametrize('adaptive_centers', [True, False])
@pytest.mark.parametrize('adaptive_scalings', [True, False])
@pytest.mark.parametrize('k', [-1, 0, 1, 2, 3])
@pytest.mark.parametrize('nc', [None, 20])
@pytest.mark.parametrize('init_region', [None, 20, (1, 2), (1, 2, 3), "te", [3, 2]])
@pytest.mark.parametrize('layout', ['grid', 'random'])
def test_cbconv_module(layout, init_region, nc, k, adaptive_centers, adaptive_scalings):
    """Check basic functionality of the module."""
    input_channels = 3
    output_channels = 4
    kernel_size = (20, 20)
    input_ = torch.rand(1, input_channels, 256, 256)

    def create_module():
        return CBSConv(
            input_channels, output_channels, kernel_size,
            layout=layout, nc=nc, k=k,
            adaptive_centers=adaptive_centers,
            init_region=init_region,
            adaptive_scalings=adaptive_scalings)

    if k < 0:
        with pytest.raises(ValueError):
            module = create_module()
        return

    if layout == 'random' and not isinstance(nc, int):
        with pytest.raises(TypeError):
            module = create_module()
        return

    if init_region is not None:
        if not isinstance(init_region, tuple):
            with pytest.raises(TypeError):
                module = create_module()
        elif len(init_region) != len(kernel_size):
            with pytest.raises(ValueError):
                module = create_module()
        return

    if layout == 'grid' and nc is not None:
        with pytest.warns(UserWarning):
            module = create_module()
        return
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


@pytest.mark.parametrize('dilation', [1, 2])
@pytest.mark.parametrize('padding', [1, 3])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('kernel_size', [3, 7])
@pytest.mark.parametrize('separable', [True, False])
def test_cbsconv_vs_standard(separable, kernel_size, stride, bias, padding, dilation):
    """Test that cbsconv can emulate standard conv."""
    batch = 2
    input_channels = 4
    output_channels = 8
    input_size = (32, 48)

    input_ = torch.rand(batch, input_channels, *input_size)
    standard_layer = torch.nn.Conv2d(
        input_channels, output_channels, kernel_size=kernel_size,
        stride=stride, bias=bias, padding=padding, dilation=dilation)

    cbsconv_layer = CBSConv2d(
        input_channels, output_channels, kernel_size=kernel_size,
        stride=stride, bias=bias, k=0, padding=padding,
        dilation=dilation, layout='grid', adaptive_centers=False,
        adaptive_scalings=False, separable=separable)

    loss = torch.nn.MSELoss()

    weights = torch.empty_like(standard_layer.weight)
    weights.requires_grad = True
    torch.nn.init.kaiming_normal_(weights, mode='fan_out', nonlinearity='relu')


    with torch.no_grad():
        standard_layer.weight.data = weights
        cbsconv_layer.weights.data = weights.reshape(
            *cbsconv_layer.weights.data.shape)
        cbsconv_layer.bias = standard_layer.bias

    assert torch.allclose(
        standard_layer.weight.view(-1), cbsconv_layer.weights.view(-1))
    result_standard = standard_layer(input_)
    result_cbsconv = cbsconv_layer(input_)

    target = torch.zeros_like(result_standard)
    ls_out = loss(result_standard, target)
    lc_out = loss(result_cbsconv, target)

    ls_out.backward()
    lc_out.backward()

    assert torch.allclose(result_standard, result_cbsconv, atol=1e-5)
    assert torch.allclose(
        standard_layer.weight.grad.view(-1),
        cbsconv_layer.weights.grad.view(-1),
        atol=1e-5)
