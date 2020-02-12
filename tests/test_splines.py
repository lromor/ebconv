"""Test of the UniforBSPlineConv layer."""

import ebconv
from ebconv.splines import BSplineBasis

import numpy as np

import pytest


TEST_SPLINE_ORDERS = [3, 5, 11]


@pytest.mark.parametrize('n', TEST_SPLINE_ORDERS)
def test_uniform_knots(n):
    """Check that the number of knots satisfies the relations |k| = |n| + 1."""
    k = ebconv.splines.uniform_knots(n)
    assert np.mean(k) == 0
    assert len(k) == n + 1

    spacing = np.ediff1d(k)
    assert (spacing == np.ones_like(spacing)).all()


@pytest.mark.parametrize('epsilon', [1e-2])
@pytest.mark.parametrize('s', [0.1, 1, 10])
@pytest.mark.parametrize('n', TEST_SPLINE_ORDERS)
def test_bspline(n, s, epsilon):
    """Match values with scipy."""
    bspline_prev = BSplineBasis.create_cardinal(n, s)
    bspline_next = BSplineBasis.create_cardinal(n + 1, s)

    k_next = bspline_next.get_knots()
    # Create the domain.
    lb, ub = k_next[0], k_next[-1]
    x, step = np.linspace(
        lb, ub, int(np.ceil((ub - lb) * 2 / (s * epsilon))), retstep=True)

    # Check that we can build n + 1 from n using convolution.
    yprev = bspline_prev(x)
    square = ebconv.splines.square_signal(x / s)
    yconv = np.convolve(yprev, square, mode='same') * step / s
    yrec = bspline_next(x)
    assert np.allclose(yconv, yrec, atol=epsilon)


@pytest.mark.parametrize('shape', [(5,), (100,), (3, 3), (5, 10, 15)])
@pytest.mark.parametrize('n', TEST_SPLINE_ORDERS)
def test_bspline_sample(n, shape):
    """Sampling should return a symmetric array of non-zero values."""
    b = BSplineBasis.create_cardinal(n)
    y = b.sample(shape)

    # Check the size
    assert y.squeeze().shape == shape

    # Check symmetry
    assert np.allclose(y, np.flip(y))

    # Only the support should be sampled
    assert (y != 0).all()

    # Test sshift
    shift = np.zeros(len(shape))
    with pytest.raises(ValueError):
        b.sample(shape, sshift=shift + 0.5)

    shift_l = shift + 0.45
    shift_r = shift - 0.45
    yl = b.sample(shape, sshift=shift_l)
    yr = b.sample(shape, sshift=shift_r)

    # Check symmetry
    assert np.allclose(yl, np.flip(yr))
