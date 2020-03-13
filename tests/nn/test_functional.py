
from ebconv.kernel import CardinalBSplineKernel
from ebconv.nn.functional import cbsconv
from ebconv.nn.functional import crop
from ebconv.nn.functional import translate

import numpy as np

import pytest

import torch


@pytest.mark.parametrize('k', [1, 2, 3])
@pytest.mark.parametrize('s', [0.5, 1.3])
@pytest.mark.parametrize('c', [
    (0.0,), (-6.9,), (-0.3,), (1.3,), (2.9,),
    (-5, 4.6, 0.9, 10.3),
    (10.4, 3.7, -5.4),
])
@pytest.mark.parametrize('i_size', [(1, 1, 50)])
def test_cbsconv1d(i_size, c, s, k):
    """Test the 1d numerical solution of bspline conv.

    Perform a basis check for a simple 1d signal
    with different shifts and sizes with a specified
    set of centers.
    """
    kb = CardinalBSplineKernel.create(c=c, s=s, k=k)

    # Sample the kernel
    kernel_size = kb.centered_region().round()
    cb = np.array((-kernel_size / 2, kernel_size / 2))
    x = np.arange(cb[0], cb[1]) + 0.5

    bases = kb(x)

    w = np.ones_like(c)
    kw = np.tensordot(w, bases, axes=1)

    # Standard convolution
    input_ = torch.rand(i_size)
    w_ = torch.Tensor(kw)[None, None, :]
    torch_conv = torch.nn.functional.conv1d(input_, w_)

    # Cardinal B-spline convolution.
    bw_ = torch.Tensor(w)[None, None, :]
    cbs_conv = cbsconv(input_, (kernel_size,), bw_, kb.c, kb.s, kb.k)
    assert np.allclose(torch_conv, cbs_conv)


@pytest.mark.skip(reason='Missing 2d test impl.')
@pytest.mark.parametrize('k', [1, 2, 3])
@pytest.mark.parametrize('s', [0.5, 1.3])
@pytest.mark.parametrize('c', [
    (0.0, 0.0), (-6.9, 3.2), (-0.3, 0.9), (1.3, -3), (2.9, 0),
    [(-5, 4.6), (0.9, 10.3), (-0.3, 0.9)],
])
@pytest.mark.parametrize('i_size', [(1, 1, 50, 50)])
def test_cbsconv2d(i_size, c, s, k):
    """Test the 2d numerical solution of bspline conv.

    Perform a basis check for a simple 1d signal
    with different shifts and sizes with a specified
    set of centers.
    """
    kb = CardinalBSplineKernel.create(c=c, s=s, k=k)

    # Sample the kernel
    kernel_size = kb.centered_region().round()
    cb = np.array((-kernel_size / 2, kernel_size / 2))
    x = np.arange(cb[0], cb[1]) + 0.5

    bases = kb(x)

    w = np.ones_like(c)
    kw = np.tensordot(w, bases, axes=1)

    # Standard convolution
    input_ = torch.rand(i_size)
    w_ = torch.Tensor(kw)[None, None, :]
    torch_conv = torch.nn.functional.conv1d(input_, w_)

    # Cardinal B-spline convolution.
    bw_ = torch.Tensor(w)[None, None, :]
    cbs_conv = cbsconv(input_, (kernel_size,), bw_, kb.c, kb.s, kb.k)
    assert np.allclose(torch_conv, cbs_conv)


def test_translate_simple():
    """Check the resulting op using 2d tensor respects the specification."""
    # Simple 2d tensor to test the shift and crop
    tt_input = torch.tensor((
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 2.0, 3.0, 0.0),
        (0.0, 4.0, 5.0, 6.0, 0.0),
        (0.0, 7.0, 8.0, 9.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ))[None, None, :]

    assert translate(tt_input, (1, -2), mode='constant', value=0) \
        .equal(torch.tensor((
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (2.0, 3.0, 0.0, 0.0, 0.0),
            (5.0, 6.0, 0.0, 0.0, 0.0),
            (8.0, 9.0, 0.0, 0.0, 0.0),
        ))[None, None, :])

    # There's no numpy symmetric, the closest is "reflect"
    # which discards the border in the reflection.
    assert translate(tt_input, (1, -2), mode='reflect') \
        .equal(torch.tensor((
            (2.0, 3.0, 0.0, 3.0, 2.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (2.0, 3.0, 0.0, 3.0, 2.0),
            (5.0, 6.0, 0.0, 6.0, 5.0),
            (8.0, 9.0, 0.0, 9.0, 8.0),
        ))[None, None, :])


def test_crop_simple():
    """Check that cropping along the two axes gives the expected result."""
    tt_input = torch.tensor((
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 2.0, 3.0, 0.0),
        (0.0, 4.0, 5.0, 6.0, 0.0),
        (0.0, 7.0, 8.0, 9.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ))[None, None, :]

    assert crop(tt_input, (1, 2, 2, 1)) \
        .equal(torch.tensor((
            (2.0, 3.0),
            (5.0, 6.0),
        ))[None, None, :])
