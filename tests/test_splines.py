"""Test of the bsplines module."""

import ebconv
from ebconv.splines import BSpline

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
@pytest.mark.parametrize('k', TEST_SPLINE_ORDERS)
@pytest.mark.parametrize('s', [0.1, 1, 10])
@pytest.mark.parametrize('c', [-5.0, -0.3, 0.3, 7])
def test_cardinal_bspline(c, s, k, epsilon):
    """Match values with scipy."""
    bspline_prev = BSpline.create_cardinal(c, s, k)
    bspline_next = BSpline.create_cardinal(c, s, k + 1)
    k_next = bspline_next.get_knots()

    # Create the domain.
    lb, ub = k_next[0], k_next[-1]
    x, step = np.linspace(
        lb, ub, int(np.ceil((ub - lb) * 2 / (s * epsilon))), retstep=True)

    # Check that we can build n + 1 from n using convolution.
    yprev = bspline_prev(x)
    square = ebconv.splines.square_signal((x - c) / s)
    yconv = np.convolve(yprev, square, mode='same') * step / s
    yrec = bspline_next(x)
    assert np.allclose(yconv, yrec, atol=epsilon)
