"""Tests of the ebconv.utils module."""

import pytest

import torch

from ebconv.utils import convolution_output_shape


@pytest.mark.parametrize('dilation', [(1, 2, 3), 2])
@pytest.mark.parametrize('padding', [(1, 2, 3), 2])
@pytest.mark.parametrize('stride', [(1, 2, 3), 2])
@pytest.mark.parametrize('ishape,wshape', [
    ((10, 3, 20, 30, 40),
     (3, 1, 5, 4, 7)),
    ((10, 3, 10, 20, 30),
     (3, 3, 5, 4, 10))
])
def test_3d_conv_output_shape(ishape, wshape, stride,
                              padding, dilation):
    """Check that the output shape is consistent with pytorch."""
    input_ = torch.empty(ishape)
    groups = ishape[1] // wshape[1]
    weights = torch.empty(wshape)
    out_shape = torch.nn.functional.conv3d(
        input_, weights, stride=stride, padding=padding,
        dilation=dilation, groups=groups).shape
    assert out_shape == convolution_output_shape(
        ishape, wshape, stride, padding, dilation)
