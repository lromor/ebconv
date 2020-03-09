"""Tests of the ebconv.utils module."""

from ebconv.utils import conv_output_shape
from ebconv.utils import round_modf, tensordot

import numpy as np

import pytest

import torch


@pytest.mark.parametrize('x,expected', [
    (0, (0.0, 0)),
    (5.000, (0.0, 5)),
    (6.0001, (0.0001, 6)),
    (7.9999, (-0.0001, 8)),
    (8.5, (0.5, 8)),
    (-5.000, (-0.0, -5)),
    (-6.0001, (-0.0001, -6)),
    (-7.9999, (0.0001, -8)),
    (-8.5, (-0.5, -8))
])
def test_round_modf(x, expected):
    """Test round modf."""
    df, di = round_modf(x)
    r_df, r_di = expected
    assert isinstance(di, int)
    assert df + di == x
    assert np.isclose(df, r_df)
    assert di == r_di


def test_tendordot():
    """Value of tensor product of the callables.

    We should have the same values as manually sampling the functions
    separately and then performing np.tensordot().
    """
    list_fn = [np.cos, np.sin]
    x = np.linspace(-np.pi, np.pi) * 2 + 3
    y = np.linspace(-np.pi, np.pi)

    xx, yy = np.meshgrid(x, y)
    fn = tensordot(list_fn)
    zf = fn(xx, yy)

    zx = np.cos(x)
    zy = np.sin(y)
    zn = np.tensordot(zx, zy, axes=0)
    assert np.allclose(zf, zn)


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
    assert out_shape == conv_output_shape(ishape, wshape, stride,
                                          padding, dilation)
