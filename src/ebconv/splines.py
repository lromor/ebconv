"""Spline utility classes.

This module contains spline utility functions and
classes.
"""

from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from scipy.interpolate import BSpline as _BSpline


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


_TBSplineBase = TypeVar('TBSplineBase', bound='BSplineBase')


class BSplineBase(ABC):
    """Implementation of a Univariate BSpline function."""

    def __init__(self, knots: np.ndarray) -> None:
        """Initialize a cardinal bspline.

        Args:
            n: Order of the bspline
            s: Distance between knots.

        """
        self._knots = knots
        self._spacing = np.ediff1d(knots)
        self._is_cardinal = np.isclose(self._spacing[0], self._spacing).all()

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

    def get_support_interval(self):
        """Return the non zero interval of the function."""
        return self._knots[0], self._knots[-1]

    @classmethod
    def create_cardinal(cls,
                        center: float = 0.0,
                        scaling: float = 1.0,
                        order: float = 3) -> _TBSplineBase:
        """Return a cardinal bspline instance.

        Args:
            center: Center of the cardinal spline.
            scaling: Distance between the knots.
            n: Order of the spline.
        Returns:
           BSplineBase child instance with uniform knots.

        """
        bspline = cls(uniform_knots(order) * scaling + center)
        assert bspline.is_cardinal()
        return bspline

    @abstractmethod
    def _sample(self, x):
        """Backend to sample the bspline."""
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Sample the basis function."""
        return self._sample(x)


class ScipyBSpline(BSplineBase):
    """Scipy implementation of a b-spline."""

    def __init__(self, knots: np.ndarray):
        """Init scipy based bspline implementation."""
        super().__init__(knots)
        self._b = _BSpline.basis_element(self._knots, extrapolate=False)

    def _sample(self, x):
        return np.nan_to_num(self._b(x))


# Default implementation
BSpline = ScipyBSpline
