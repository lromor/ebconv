"""Generate a d-dimensional b spline kernel.

This module provides numpy based classes and functions helpful
to debug and plot bspline kernels.
"""

import functools

import random

from typing import Iterable, Tuple, TypeVar, Union

import warnings

import numpy as np

from ebconv.splines import BSplineElement


def create_uniform_grid(region: Tuple[float, ...]):
    """Creates a uniform grid."""
    bounds = tuple((-width / 2, width / 2) for width in region)
    grid_axes = [np.arange(l, u) + 0.5
                 for l, u in bounds]
    grid = np.stack(np.meshgrid(*grid_axes, indexing='ij'), -1)
    return grid.reshape(-1, 2)


def create_random_centers(region: Tuple[float, ...],
                          ncenters: int, integrally_spaced=False,
                          unique=False):
    """Return n random centers within a region of specified width."""
    if functools.reduce(lambda x, y: x * y, region) < ncenters \
       and unique and integrally_spaced:
        warnings.warn('Number of centers is too high', RuntimeWarning)

    integral_sampler = lambda width: random.choice(sampling_domain(width))
    uniform_sampler = lambda width: random.uniform(-width / 2, width / 2)

    # Sampler
    sampler = integral_sampler if integrally_spaced else uniform_sampler

    # Generate our unique random list of coordinates
    c = []
    while len(c) < ncenters:
        center = tuple(sampler(width) for width in region)
        if unique and center in c:
            continue
        c.append(center)
    return np.array(c)


def sampling_domain(kernel_size: int) -> np.ndarray:
    """Sample over a discrete unit separated domain within two ranges a, b."""
    return np.arange(-kernel_size / 2, kernel_size / 2) + 0.5


_TCardinalBSplineKernel = TypeVar('TCardinalBSplineKernel',
                                  bound='CardinalBSplineKernel')


class CardinalBSplineKernel:
    """Defines a n-dimensional, cardinal, BSplines kernel bases function.

    This class should behave as close as possible as a generic, cardinal
    and if necessary sparse b-splines kernel function.
    """

    def __init__(self, c: np.ndarray, s: np.ndarray, k: np.ndarray):
        """Initialize the kernel by providing a set of splines."""
        self._c = c
        self._s = s
        self._k = k

    @staticmethod
    def centered_region_from_params(c: np.ndarray,
                                    s: np.ndarray,
                                    k: np.ndarray) -> np.ndarray:
        """Return the size of the smallest centered region."""
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
               s: Union[Iterable[float], float] = 1.0,
               k: Union[Iterable[int], int] = 2) -> _TCardinalBSplineKernel:
        """Generate a bsplines kernel given centers, scalings and order."""
        ncenters = len(c)
        c = np.array(c).reshape(ncenters, -1)

        if not isinstance(s, Iterable):
            s = float(s)
            s = ((s,) * ncenters)
        if not isinstance(k, Iterable):
            k = ((k,) * ncenters)

        s = np.array(s).reshape(ncenters)
        k = np.array(k, dtype=int).reshape(ncenters)

        # Create the basis
        return cls(c, s, k)
