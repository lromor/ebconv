"""Test of the bsplines module."""

import numpy as np

import pytest

from ebconv.splines import BSplineElement
from ebconv.splines import uniform_knots
from ebconv.splines import square_signal


TEST_SPLINE_ORDERS = [3, 5, 11]


@pytest.mark.parametrize('k', TEST_SPLINE_ORDERS)
def test_uniform_knots(k):
    """Check that the number of knots satisfies the relations |k| = |n| + 2."""
    knots = uniform_knots(k)
    assert np.mean(knots) == 0
    assert len(knots) == k + 2

    spacing = np.ediff1d(knots)
    assert (spacing == np.ones_like(spacing)).all()


@pytest.mark.parametrize('epsilon', [1e-2])
@pytest.mark.parametrize('k', TEST_SPLINE_ORDERS)
@pytest.mark.parametrize('s', [0.1, 1, 10])
@pytest.mark.parametrize('c', [-5.0, -0.3, 0.3, 7])
def test_cardinal_bspline1d(c, s, k, epsilon):
    """Match values with scipy."""
    bspline_prev = BSplineElement.create_cardinal(c, s, k)
    bspline_next = BSplineElement.create_cardinal(c, s, k + 1)
    k_next = bspline_next.knots()[0]

    # Create the domain.
    lower, upper = k_next[0], k_next[-1]
    x, step = np.linspace(
        lower, upper, int(np.ceil((upper - lower) * 2 / (s * epsilon))), retstep=True)

    # Check that we can build n + 1 from n using convolution.
    yprev = bspline_prev(x)
    square = square_signal((x - c) / s)
    yconv = np.convolve(yprev, square, mode='same') * step / s
    yrec = bspline_next(x)
    assert np.allclose(yconv, yrec, atol=epsilon)
