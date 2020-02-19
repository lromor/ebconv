"""Generate a d-dimensional b spline kernel."""

import random
from typing import Iterable, Tuple, Union

from ebconv.splines import BSpline
from ebconv.utils import tensordot

import numpy as np


def create_random_centers(bounds: Iterable[Tuple[float, ...]],
                          n: int, integral_values=False):
    """Return n random centers within bounds."""
    # Sampler
    sampler = random.randint if integral_values else random.uniform

    # Generate our unique random list of coordinates
    c = set()
    for _ in range(n):
        c.add(tuple(sampler(lb, ub) for lb, ub in bounds))
    c = np.array(list(c))
    return c


class CardinalBSplineKernel:
    """Defines a n-dimensional, cardinal, BSplines kernel bases function.

    This class should behave as close as possible as a generic, cardinal
    and if necessary sparse b-splines kernel function.
    """

    def __init__(self, c, s, k, w):
        """Initialize the kernel by providing a set of splines."""
        self._c = c
        self._s = s
        self._k = k
        self._w = w
        splines = []
        for p in zip(self._c, self._s, self._k):
            splines.append(
                tuple(BSpline.create_cardinal(*ap) for ap in zip(*p)))
        self._splines = tuple(splines)

    @property
    def c(self):
        """Return read only var."""
        return self._c

    @property
    def s(self):
        """Return read only var."""
        return self._s

    @property
    def k(self):
        """Return read only var."""
        return self._k

    def _sample(self, *args, **kwargs):
        """Evaluate the different bsplines in the domain provided by *args.

        Returns a tensor with the sampled function for each of the centers.
        """
        bases = []
        for dsplines in self._splines:
            b = tensordot(dsplines)
            bases.append(b(*args, **kwargs))
        return np.stack(bases)

    def __call__(self, *args, **kwargs):
        """Sample in the provided domain."""
        return self._sample(*args, **kwargs)

    def copy(self):
        """Return a duplicate instance of the kernel."""
        return CardinalBSplineKernel(self._c, self._s, self._k, self._w)

    @classmethod
    def create(cls, c: Iterable[Tuple[float, ...]],
               s: Union[Iterable[Tuple[float, ...]], float] = 1.0,
               k: Union[Iterable[Tuple[int, ...]], int] = 3,
               w: Union[Iterable[float], None] = None):
        """Generate a bspline kernel with randomly positioned centers."""
        c = np.array(c)
        n, ndims = c.shape

        if isinstance(s, float) or isinstance(s, int):
            s = float(s)
            s = (((s,) * ndims,) * n)
        if isinstance(k, int):
            k = (((k,) * ndims,) * n)

        s = np.array(s)
        k = np.array(k)
        assert s.shape == k.shape

        if w is None:
            w = np.ones(n)
        w = np.array(w)
        assert w.shape == (n,)

        # Create the basis
        return cls(c, s, k, w)
