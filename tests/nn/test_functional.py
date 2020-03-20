
from ebconv.kernel import CardinalBSplineKernel
from ebconv.nn.functional import cbsconv
from ebconv.nn.functional import crop
from ebconv.nn.functional import translate
from ebconv.utils import sampling_domain

import numpy as np

import pytest

import torch


@pytest.mark.parametrize('batch', [10])
@pytest.mark.parametrize('k', [1, 2, 3])
@pytest.mark.parametrize('s', [0.5, 1.3])
@pytest.mark.parametrize('c', [
    (0.0,), (-6.9,), (-0.3,), (1.3,), (2.9,),
    (-5, 4.6, 0.9, 10.3),
    (10.4, 3.7, -5.4),
])
def test_cbsconv1d(c, s, k, batch):
    """Test the 1d numerical solution of bspline conv.

    Perform a basis check for a simple 1d signal
    with different shifts and sizes with a specified
    set of centers.
    """
    kb = CardinalBSplineKernel.create(c=c, s=s, k=k)

    # First we test the behavior using of the exact size
    # of the smallest centered region.

    # Sample the smallest centered region containing all the non-zero
    # values of the kernel.
    kernel_size = int(kb.centered_region().round())
    x = sampling_domain(kernel_size)
    bases = kb(x)
    w = np.random.rand(*np.array(c).shape)
    kw = np.tensordot(w, bases, axes=1)
    bw_ = torch.Tensor(w)[None, None, :]

    # kernel size > input_size
    with pytest.raises(RuntimeError):
        input_ = torch.rand((batch, 1, kernel_size - 1))
        cbs_conv = cbsconv(input_, (kernel_size,), bw_, kb.c, kb.s, kb.k)

    # kernel size == input size
    input_ = torch.rand(batch, 1, kernel_size)
    w_ = torch.Tensor(kw)[None, None, :]
    torch_conv = torch.nn.functional.conv1d(input_, w_)
    cbs_conv = cbsconv(input_, (kernel_size,), bw_, kb.c, kb.s, kb.k)
    assert np.allclose(torch_conv, cbs_conv)

    # kernel_size * 2 == input_size
    input_ = torch.rand(batch, 1, kernel_size * 2)
    torch_conv = torch.nn.functional.conv1d(input_, w_)
    cbs_conv = cbsconv(input_, (kernel_size,), bw_, kb.c, kb.s, kb.k)
    assert np.allclose(torch_conv, cbs_conv)

    # Then we test custom sizes of the kernel.
    # First we start with a smaller version
    kernel_size_small = int(round(kernel_size / 2)) + 1
    x_small = sampling_domain(kernel_size_small)
    bases = kb(x_small)
    kw = np.tensordot(w, bases, axes=1)
    w_ = torch.Tensor(kw)[None, None, :]
    bw_ = torch.Tensor(w)[None, None, :]
    torch_conv = torch.nn.functional.conv1d(input_, w_)
    cbs_conv = cbsconv(input_, (kernel_size_small,), bw_, kb.c, kb.s, kb.k)
    assert np.allclose(torch_conv, cbs_conv)

    # then with a bigger version.
    kernel_size_big = int(round(kernel_size * 2))
    x_big = sampling_domain(kernel_size_big)
    bases = kb(x_big)
    kw = np.tensordot(w, bases, axes=1)
    w_ = torch.Tensor(kw)[None, None, :]
    bw_ = torch.Tensor(w)[None, None, :]
    torch_conv = torch.nn.functional.conv1d(input_, w_)
    cbs_conv = cbsconv(input_, (kernel_size_big,), bw_, kb.c, kb.s, kb.k)
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
