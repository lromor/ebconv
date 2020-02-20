"""Test the nn.functional module."""

from ebconv.kernel import CardinalBSplineKernel, create_random_centers
from ebconv.nn.functional import cbsconv
from ebconv.nn.functional import crop
from ebconv.nn.functional import translate

import numpy as np

import pytest

import torch


# @pytest.mark.parametrize('k', [4])
# @pytest.mark.parametrize('s', [48])
# @pytest.mark.parametrize('n', [16])
# @pytest.mark.parametrize('k_interval', [(100, -100, -100, 100)])
# @pytest.mark.parametrize('input_size', [(1, 1, 800, 600)])
# def test_bconv2d(input_size, k_interval, n, s, k):
#     """Test the numerical solution of bspline conv."""
#     # Create a sample 2d function as input
#     interval = np.array((100, -100, -100, 100)).reshape(2, 2)
#     c = create_random_centers(interval, n)
#     kb = CardinalBSplineKernel.create(c=c, s=s, k=k)

#     tt = torch.rand(*input_size)

#     # Init random weights
#     tw = torch.rand((3, 1, n))
#     conv = cbsconv(tt, (100, 100), tw, torch.Tensor(kb.c),
#                    torch.Tensor(kb.s), torch.Tensor(kb.k))


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
