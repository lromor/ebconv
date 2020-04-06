"""Spline utility classes.

This module contains spline utility functions and
classes.
"""

from typing import Iterable, List, Tuple, TypeVar, Union

import numpy as np

from scipy.interpolate import BSpline as _BSpline

from ebconv.operator import tensordot


def uniform_knots(k: int) -> np.ndarray:
    """Generate uniform and centered knots.

    Args:
        n: Order of the bspline.

    Returns:
        Array of knots positions.

    Raises:
        RuntimeError: if n < 2

    """
    if k < 0:
        raise RuntimeError('Knots order n must be >= 0.')
    knots = np.arange(0, k + 2) - (k + 1) / 2
    return knots


def square_signal(x: np.ndarray, width=1) -> np.ndarray:
    """Return a centered square signal."""
    return np.heaviside(x + width / 2, 1) * np.heaviside(-x + width / 2, 1)


_TUnivariateBSplineElement = TypeVar('TUnivariateBSplineElement',
                                     bound='UnivariateBSplineElement')

class UnivariateBSplineElement():
    """Define a univariate b-spline element.

    Wrapper class to the scipy object.
    """

    def __init__(self, basis: _TUnivariateBSplineElement) -> None:
        """Initialize a uniform b-spline element using knots.

        Args:
            knots: Knots that completely define the basis element.

        """
        self._b = basis

        # Spline order
        self.k = self._b.k

    @classmethod
    def from_knots(cls, knots) -> _TUnivariateBSplineElement:
        """Create basis element from knots."""
        return cls(_BSpline.basis_element(knots, extrapolate=False))

    def knots(self) -> np.ndarray:
        """Return the original set of knots."""
        basis = self._b
        return basis.t[basis.k:-basis.k if basis.k != 0 else None]

    def derivative(self, *args, **kwargs):
        """Return the derivative of the element."""
        return UnivariateBSplineElement(self._b.derivative(*args, **kwargs))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Sample the bspline element in the domain."""
        return np.nan_to_num(self._b(x))


_TBSplineElementBase = TypeVar('TBSplineElementBase',
                               bound='BSplineElementBase')


class BSplineElement():
    """Implementation of a Univariate BSpline function."""

    def __init__(self, ubsplines: Iterable[UnivariateBSplineElement]) -> None:
        """Initialize a bspline element.

        Args:
            splines: Iterable of univariate bsplines.

        """
        self._ubsplines = ubsplines
        self._b = tensordot(*self._ubsplines)

    @classmethod
    def from_knots(cls, knots: Iterable[np.ndarray]) -> _TBSplineElementBase:
        """Construct the Bspline from the knots."""
        return cls([UnivariateBSplineElement.from_knots(k) for k in knots])

    @classmethod
    def create_cardinal(
            cls, c: Union[Tuple[float, ...], float] = 0.0,
            s: Union[Tuple[float, ...], float] = 1.0,
            k: Union[Tuple[int, ...], int] = 3) -> _TBSplineElementBase:
        """Return a cardinal bspline instance.

        Args:
            center: Center of the cardinal spline.
            scaling: Distance between the knots.
            n: Order of the spline, 3=cubic.
        Returns:
           BSplineElementBase child instance with uniform knots.

        """
        if not isinstance(c, Iterable):
            c = (c,)
        if not isinstance(s, Iterable):
            s = (s,)
        if not isinstance(k, Iterable):
            k = (k,)

        knots = [uniform_knots(k_i) * s_i + c_i
                 for c_i, s_i, k_i in zip(c, s, k)]
        bspline = cls.from_knots(knots)
        assert bspline.is_cardinal()
        return bspline

    def derivative(self) -> _TBSplineElementBase:
        """Returns the corresponding derivative of the bspline."""
        dsplines = [spline.derivative() for spline in self._ubsplines]
        return BSplineElement(dsplines)

    def knots(self) -> List[np.ndarray]:
        """Return a list of knots for every dimension."""
        return [b.knots() for b in self._ubsplines]

    def support_bounds(self) -> np.ndarray:
        """Return the non zero interval of the function."""
        return np.array(tuple((k[0], k[-1]) for k in self.knots())).squeeze()

    def dimensionality(self) -> int:
        """Return the number of dimensions of the basis element."""
        return len(self._ubsplines)

    def is_cardinal(self) -> bool:
        """Return true if knots are uniformly spaced."""
        spacing = tuple(map(np.ediff1d, self.knots()))
        return np.array(
            tuple(np.isclose(sp[0], sp).all() for sp in spacing)).all()

    def get_order(self) -> List[int]:
        """Return the spline order for every dimension."""
        return [b.k for b in self._ubsplines]

    def _sample(self, *args, **kwargs):
        """Backend to sample the bspline."""
        return self._b(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Sample the basis function."""
        return self._sample(*args, **kwargs)
