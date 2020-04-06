"""Tests of the ebconv.operator module."""

import numpy as np

from ebconv.operator import tensordot


def test_tendordot():
    """Value of tensor product of the callables.

    We should have the same values as manually sampling the functions
    separately and then performing np.tensordot().
    """
    list_fn = [np.cos, np.sin]
    x = np.linspace(-np.pi, np.pi) * 2 + 3
    y = np.linspace(-np.pi, np.pi)

    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    prod_function = tensordot(*list_fn)
    fn_out = prod_function(x_grid, y_grid)

    z_x = np.cos(x)
    z_y = np.sin(y)
    z_n = np.tensordot(z_x, z_y, axes=0)
    assert np.allclose(fn_out, z_n)
