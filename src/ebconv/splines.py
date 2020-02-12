"""Spline utility classes.

This module contains spline utility functions and
classes.
"""

from typing import Iterable as Iterable
from typing import TypeVar, Union

import numpy as np

from scipy.interpolate import BSpline


def uniform_knots(n: int) -> np.ndarray:
    """Generate uniform and centered knots.

    Args:
        n: Order of the bspline.

    Returns:
        Array of knots positions.

    Raises:
        RuntimeError: if n < 1

    """
    if n < 1:
        raise RuntimeError('Knots order n must be >= 1.')
    knots = np.arange(0, n + 1) - n / 2
    return knots


def square_signal(x, width=1):
    """Return a centered square signal."""
    return np.heaviside(x + width / 2, 1) * np.heaviside(-x + width / 2, 1)


_TBSplineBasis = TypeVar('TBSplineBasis', bound='BSplineBasis')


class BSplineBasis():
    """Implementation of a Univariate BSpline function."""

    def __init__(self, knots: np.ndarray) -> None:
        """Initialize a cardinal bspline.

        Args:
            n: Order of the bspline
            s: Distance between knots.

        Returns:
           Array of knots positions.

        """
        self._knots = knots
        self._spacing = np.ediff1d(knots)
        self._is_cardinal = (self._spacing == self._spacing).all()
        self._b = BSpline.basis_element(self._knots, extrapolate=False)

    def get_knots(self) -> np.ndarray:
        """Return the knots of the basis."""
        return self._knots

    def is_cardinal(self) -> bool:
        """Return true if knots are uniformly spaced."""
        return self._is_cardinal

    def get_order(self) -> int:
        """Return the spline order."""
        return len(self.knots) - 1

    def get_polynomial_order(self) -> int:
        """Return polynimial order of the basis."""
        return self.get_order() - 1

    def get_spacing(self) -> np.ndarray:
        """Return the spacing between knots."""
        return self._spacing

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Sample the basis function."""
        return np.nan_to_num(self._b(x))

    def get_support_interval(self):
        """Return the non zero interval of the function."""
        return self._knots[0], self._knots[-1]

    def sample(self, shape: Union[Iterable[int], int],
               sshift: Union[Iterable[float], float] = 0.0):
        """Return the basis support using width samples.

        sshift is a list of float values between (-0.5, 0.5) to tweak
        a real centered position of the bsplines.

        Args:
            sshift: Float value

        Returns:
            Sampled kernel.

        """
        if isinstance(shape, int):
            shape = (shape,)

        if isinstance(sshift, float):
            sshift = (sshift,)
            if len(shape) > 1:
                sshift = sshift * len(shape)
        assert np.array(shape).shape == np.array(sshift).shape

        if not (np.abs(sshift) < 0.5).all():
            raise ValueError('sshift should be between (-0.5, 0.5)')

        lb, ub = self.get_support_interval()
        out = np.array((1,))

        for s, shift in zip(shape, sshift):
            x, step = np.linspace(lb, ub, s, endpoint=False, retstep=True)
            xout = self(x + step / 2 - shift)
            out = np.tensordot(out, xout, axes=0)
        return out

    @classmethod
    def create_cardinal(cls, n: int, s: int = 1.0) -> _TBSplineBasis:
        """Return a cardinal bspline instance.

        Args:
            n: Order of the spline.
            s: Scaling.

        Returns:
           Array of samples.

        """
        return cls(uniform_knots(n) * s)
