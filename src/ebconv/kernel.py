"""Generate a d-dimensional b spline kernel."""

import random
from typing import Iterable, Tuple, Union

from ebconv.splines import BSpline
from ebconv.utils import tensordot

import numpy as np


class BSplineKernelBasis:
    """Defines a n-dimensional, cardinal, BSplines kernel bases function.

    This class should behave as close as possible as a generic, cardinal
    and if necessary sparse b-splines kernel function.
    """

    def __init__(self, c, s, k):
        """Initialize the kernel by providing a set of splines."""
        self.c = c
        self.s = s
        self.k = k

    def __call__(self, *args, **kwargs):
        """Sample in the provided domain."""
        return self._sample(*args, **kwargs)

    def _sample(self, *args, **kwargs):
        """Evaluate the function in the domain provided by *args."""
        bases = []
        for p in zip(self.c, self.s, self.k):
            splines = [BSpline.create_cardinal(*ap) for ap in zip(*p)]
            b = tensordot(splines)
            bases.append(b(*args, **kwargs))
        return np.stack(bases)

    def sample_from_interval(*args, **kwargs):
        """Sample the kernel from in a given dd interval.

        This method should simply take care of generating the meshgrid.
        """
        raise NotImplementedError

    @classmethod
    def create_randomly_centered(
            cls, bounds: Iterable[Tuple[float, ...]], n: int,
            s: Union[Iterable[Tuple[float, ...]], float] = 1.0,
            k: Union[Iterable[Tuple[int, ...]], int] = 3,
            integral_values=False):
        """Generate a bspline kernel with randomly positioned centers."""
        ndims = len(bounds)

        if isinstance(s, float) or isinstance(s, int):
            s = float(s)
            s = (((s,) * ndims,) * n)
        if isinstance(k, int):
            k = (((k,) * ndims,) * n)

        s = np.array(s)
        k = np.array(k)
        assert s.shape == k.shape

        # Sampler
        sampler = random.randint if integral_values else random.uniform

        # Generate our unique random list of coordinates
        c = set()
        for _ in range(n):
            c.add(tuple(sampler(lb, ub) for lb, ub in bounds))
        c = np.array(list(c))

        # Create the basis
        return cls(c, s, k)
