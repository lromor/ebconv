"""Test of the UniforBSPlineConv layer."""

import bconv
from bconv.splines import BSplineBasisFunction

import numpy as np

import pytest


@pytest.mark.parametrize('n', [1, 2, 3])
def test_uniform_knots(n):
    """Check that the number of knots satisfies the relations |k| = |n| + 1."""
    print(n)
    k = bconv.splines.uniform_knots(n)
    assert np.mean(k) == 0
    assert len(k) == n + 1

    spacing = np.ediff1d(k)
    assert (spacing == np.ones_like(spacing)).all()


@pytest.mark.parametrize('epsilon', [1e-2])
@pytest.mark.parametrize('s', [0.1, 1, 10])
@pytest.mark.parametrize('n', list(range(1, 10)))
def test_bspline(n, s, epsilon):
    """Match values with scipy."""
    bspline_prev = BSplineBasisFunction.create_cardinal(n, s)
    bspline_next = BSplineBasisFunction.create_cardinal(n + 1, s)

    k_next = bspline_next.get_knots()
    # Create the domain.
    lb, ub = k_next[0], k_next[-1]
    x, step = np.linspace(
        lb, ub, int(np.ceil((ub - lb) * 2 / (s * epsilon))), retstep=True)

    # Check that we can build n + 1 from n using convolution.
    yprev = bspline_prev(x)
    square = bconv.splines.square_signal(x / s)
    yconv = np.convolve(yprev, square, mode='same') * step / s
    yrec = bspline_next(x)
    assert np.allclose(yconv, yrec, atol=epsilon)
