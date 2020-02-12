"""A collection of math utilities."""

from typing import Tuple


def round_modf(x: float) -> Tuple[float, int]:
    """Similar to math.modf but using round."""
    i = int(round(x))
    return x - i, i
