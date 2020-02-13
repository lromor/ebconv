"""A collection of math utilities."""

from typing import Tuple


import numpy as np


def round_modf(x: float) -> Tuple[float, int]:
    """Similar to math.modf but using round."""
    i = int(round(x))
    return x - i, i


def tensorprod_fn(fns):
    """Return a callable that evalautes the tensor product."""
    def fn(*args, **kwargs):
        return np.multiply.reduce([f(args[0]) for f in fns])
    return fn
