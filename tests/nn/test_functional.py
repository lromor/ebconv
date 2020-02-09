"""Test the nn.functional module."""

from ebconv.nn.functional import crop
from ebconv.nn.functional import translate

import torch


def test_bconv2d():
    """Test the numerical solution of bspline conv."""
    # Create a sample 2d function as input
    # f = torch.Tensor(np.random.rand(100, 100))
    pass


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
