"""Spline utility classes.

This module contains spline utility functions and
classes.
"""

from typing import Iterable, List, Tuple, TypeVar, Union

from ebconv.utils import tensordot

import numpy as np

from scipy.interpolate import BSpline as _BSpline


def uniform_knots(n: int) -> np.ndarray:
    """Generate uniform and centered knots.

    Args:
        n: Order of the bspline.

    Returns:
        Array of knots positions.

    Raises:
        RuntimeError: if n < 2

    """
    if n < 0:
        raise RuntimeError('Knots order n must be >= 0.')
    knots = np.arange(0, n + 2) - (n + 1) / 2
    return knots


def square_signal(x: np.ndarray, width=1) -> np.ndarray:
    """Return a centered square signal."""
    return np.heaviside(x + width / 2, 1) * np.heaviside(-x + width / 2, 1)


class UnivariateBSplineElement():
    """Define a univariate b-spline element."""

    def __init__(self, knots: np.ndarray) -> None:
        """Initialize a uniform b-spline element using knots.

        Args:
            knots: Knots that completely define the basis element.

        """
        self._b = _BSpline.basis_element(knots, extrapolate=False)
        self.k = self._b.k
        self.c = self._b.c
        self.t = self._b.t

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Sample the bspline element in the domain."""
        return np.nan_to_num(self._b(x))


_TBSplineElementBase = TypeVar('TBSplineElementBase',
                               bound='BSplineElementBase')


class BSplineElement():
    """Implementation of a Univariate BSpline function."""

    def __init__(self, knots: Iterable[np.ndarray]) -> None:
        """Initialize a bspline element.

        Args:
            knots: List of array of knots.

        """
        self._univariate_splines = [UnivariateBSplineElement(k) for k in knots]
        self._b = tensordot(self._univariate_splines)

    def dimensionality(self) -> int:
        """Return the number of dimensions of the basis element."""
        return len(self._univariate_splines)

    def knots(self) -> List[np.ndarray]:
        """Return a list of knots for every dimension."""
        return [b.t[b.k:-b.k if b.k != 0 else None]
                for b in self._univariate_splines]

    def is_cardinal(self) -> bool:
        """Return true if knots are uniformly spaced."""
        spacing = tuple(map(np.ediff1d, self.knots()))
        return np.array(
            tuple(np.isclose(sp[0], sp).all() for sp in spacing)).all()

    def get_order(self) -> List[int]:
        """Return the spline order for every dimension."""
        return [b.k for b in self._univariate_splines]

    def support_bounds(self) -> np.ndarray:
        """Return the non zero interval of the function."""
        return np.array(tuple((k[0], k[-1]) for k in self.knots())).squeeze()

    @classmethod
    def create_cardinal(
            cls, center: Union[Tuple[float, ...], float] = 0.0,
            scaling: Union[Tuple[float, ...], float] = 1.0,
            order: Union[Tuple[int, ...], int] = 3) -> _TBSplineElementBase:
        """Return a cardinal bspline instance.

        Args:
            center: Center of the cardinal spline.
            scaling: Distance between the knots.
            n: Order of the spline, 3=cubic.
        Returns:
           BSplineElementBase child instance with uniform knots.

        """
        if not isinstance(center, Iterable):
            center = (center,)
        if not isinstance(scaling, Iterable):
            scaling = (scaling,)
        if not isinstance(order, Iterable):
            order = (order,)

        knots = [uniform_knots(o) * s + c
                 for o, s, c in zip(order, scaling, center)]
        bspline = cls(knots)
        assert bspline.is_cardinal()
        return bspline

    def _sample(self, *args, **kwargs):
        """Backend to sample the bspline."""
        return self._b(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Sample the basis function."""
        return self._sample(*args, **kwargs)
