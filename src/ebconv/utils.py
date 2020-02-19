"""A collection of math utilities."""

from typing import Tuple, Union

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


def conv_output_shape(ishape: Tuple[int],
                      wshape: Tuple[int],
                      stride: Union[Tuple[int], int],
                      padding: Union[Tuple[int], int],
                      dilation: Union[Tuple[int], int]):
    """Compute the final tensor shape resulting after a dd convolution.

    Returns the output shape of a convolution using for instance
    pytorch torch.nn.functional.conv2d.
    Should return N, out_channels, h, w, d, ...

    ishape: N, cin, h, w, d, ...
    wshape: cout, cin / groups, kh, kw, kd, ...
    """
    ispatial_shape = np.array(ishape[2:])
    nspatial_dims = len(ispatial_shape)

    kspatial_shape = np.array(wshape[2:])

    if isinstance(stride, int):
        stride = ((stride,) * nspatial_dims)
    stride = np.array(stride)

    if isinstance(padding, int):
        padding = ((padding,) * nspatial_dims)
    padding = np.array(padding)

    if isinstance(dilation, int):
        dilation = ((dilation,) * nspatial_dims)
    dilation = np.array(dilation)

    sd = (ispatial_shape + 2 * padding - dilation * (kspatial_shape - 1) - 1) \
        / stride + 1
    sd = np.floor(sd).astype(int)

    return (ishape[0], wshape[0], *sd)
