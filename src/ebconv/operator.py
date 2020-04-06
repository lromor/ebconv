"""Operators compatible with both numpy and pytorch."""

from typing import Callable, Tuple


def tensordot(*fns: Tuple[Callable]):
    """Tensor product operator.

    The output function can be used with meshgrid values.
    Remember that meshgrid uses by default a different indexing.
    for instance:

    ```
    fns = [np.sin, np.cos]
    sincos = tensordot(fns)
    x = np.linspace(-10, 10)
    y = np.lisnapce(-10, 10)
    z = sincos(*np.meshgrid(x, y, indexing='ij'))
    ```
    """
    def tensor_fn(*args):
        if len(args) > len(fns):
            raise TypeError('Too many arguments.')
        out = 1
        for function, function_args in zip(fns, args):
            out *= function(function_args)
        return out
    return tensor_fn
