"""Generate a d-dimensional b spline kernel.

This module provides numpy based classes and functions helpful
to debug and plot bspline kernels.
"""

import random
from typing import Iterable, Tuple, TypeVar, Union

import numpy as np

from ebconv.splines import BSplineElement


def create_random_centers(bounds: Iterable[Tuple[float, ...]],
                          ncenters: int, integral_values=False):
    """Return n random centers within bounds."""
    # Sampler
    sampler = random.randint if integral_values else random.uniform

    # Generate our unique random list of coordinates
    c = set()
    for _ in range(ncenters):
        c.add(tuple(sampler(lb, ub) for lb, ub in bounds))
    c = np.array(list(c))
    return c


_TCardinalBSplineKernel = TypeVar('TCardinalBSplineKernel',
                                  bound='CardinalBSplineKernel')


class CardinalBSplineKernel:
    """Defines a n-dimensional, cardinal, BSplines kernel bases function.

    This class should behave as close as possible as a generic, cardinal
    and if necessary sparse b-splines kernel function.
    """

    def __init__(self, c: np.ndarray, s: np.ndarray, k: np.ndarray):
        """Initialize the kernel by providing a set of splines."""
        assert c.shape == s.shape == k.shape
        self._c = c
        self._s = s
        self._k = k

    @staticmethod
    def centered_region_from_params(c: np.ndarray,
                                    s: np.ndarray,
                                    k: np.ndarray) -> np.ndarray:
        """Return the size of the smallest centered region."""
        c = np.array(c)
        s = np.array(s)
        k = np.array(k)
        assert c.shape == s.shape == k.shape
        bounds = np.array([
            np.abs(BSplineElement.create_cardinal(*p).support_bounds()).T
            for p in zip(c, s, k)
        ])
        # bounds.shape == n, 2, dim
        bounds = bounds.reshape(len(bounds) * 2, -1)
        bounds = np.max(bounds, axis=0)
        return np.array([2 * b for b in bounds]).squeeze()

    def centered_region(self) -> np.ndarray:
        """Return the boundaries of the smallest centered domain."""
        return self.centered_region_from_params(self._c, self._s, self._k)

    @property
    def c(self):
        """Return read only member."""
        return self._c.view()

    @property
    def s(self):
        """Return read only var."""
        return self._s.view()

    @property
    def k(self):
        """Return read only var."""
        return self._k.view()

    def _sample(self, *args, **kwargs):
        """Evaluate the different bsplines in the domain provided by *args.

        Returns a tensor with the sampled function for each of the centers.
        """
        bases = []
        for parameters in zip(self.c, self.s, self.k):
            basis = BSplineElement.create_cardinal(*parameters)
            bases.append(basis(*args, **kwargs))
        return np.stack(bases)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Sample in the provided domain."""
        return self._sample(*args, **kwargs)

    def copy(self) -> _TCardinalBSplineKernel:
        """Return a duplicate instance of the kernel."""
        return CardinalBSplineKernel(self.c.copy(),
                                     self.s.copy(),
                                     self.k.copy())

    @classmethod
    def create(cls, c: Iterable[Tuple[float, ...]],
               s: Union[Iterable[Tuple[float, ...]], float] = 1.0,
               k: Union[Iterable[Tuple[int, ...]],
                        int] = 3) -> _TCardinalBSplineKernel:
        """Generate a bspline kernel with randomly positioned centers."""
        c = np.array(c)
        c = c[:, None] if len(c.shape) == 1 else c
        ncenters, ndims = c.shape

        if not isinstance(s, Iterable):
            s = float(s)
            s = (((s,) * ndims,) * ncenters)
        if not isinstance(k, Iterable):
            k = (((k,) * ndims,) * ncenters)

        s = np.array(s)
        s = s[:, None] if len(s.shape) == 1 else s
        k = np.array(k, dtype=int)
        k = k[:, None] if len(k.shape) == 1 else k

        # Create the basis
        return cls(c, s, k)
