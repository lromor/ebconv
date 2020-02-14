"""A collection of math utilities."""

from typing import Tuple

import numpy as np


def round_modf(x: float) -> Tuple[float, int]:
    """Similar to math.modf but using round."""
    i = int(round(x))
    return x - i, i


def tensordot(fns):
    """Return a callable that evalautes the tensor product."""
    def fn(*args, **kwargs):
        if len(args) > len(fns):
            raise TypeError('Too many arguments.')
        return np.multiply.reduce(
            [f(a) for f, a in zip(fns, reversed(args))])
    return fn
