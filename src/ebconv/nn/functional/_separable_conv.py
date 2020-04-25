"""Implementation of a d-dimensional separable convolution for pytorch."""

from typing import Iterable, Optional, Tuple, Union

import numpy as np

import torch


def convdd_separable(input_: torch.Tensor, weight: Iterable[torch.Tensor],
                     bias: Optional[torch.Tensor] = None,
                     stride: Union[int, Tuple[int, ...]] = 1,
                     padding: Union[int, Tuple[int, ...]] = 0,
                     dilation: Union[int, Tuple[int, ...]] = 1,
                     groups: int = 1):
    """Compute a separable d-dimensional convolution.

    input.shape = batch, iC, iX, iY, iZ, ...
    weight = dims, sizes
    """
    # If it's one dimensional, simply use conv1d.
    if len(weight) == 1:
        return torch.nn.functional.conv1d(
            input_, weight[0], bias=bias, stride=stride,
            padding=padding, dilation=dilation, groups=groups)

    # There should be a weight per dimension.
    weight = weight[::-1]
    spatial_shape = input_.shape[2:]
    spatial_dims = len(spatial_shape)
    batch = input_.shape[0]
    output_channels, group_ic = weight[0].shape[:2]

    assert spatial_dims == len(weight)
    if not isinstance(stride, Tuple):
        stride = ((stride,) * spatial_dims)

    if not isinstance(padding, Tuple):
        padding = ((padding,) * spatial_dims)

    if not isinstance(dilation, Tuple):
        dilation = ((dilation,) * spatial_dims)

    # Array of axes to permute
    axes = np.arange(len(input_.shape))
    conv = input_.permute(0, *axes[2:-1], 1, axes[-1])

    # Split input channels between groups
    conv = conv.reshape(*conv.shape[:-2], groups, group_ic, -1)
    # Swap the order of groups / input channels
    conv = conv.transpose(-2, -3)
    # Return to the previous shape
    conv = conv.reshape(*conv.shape[:-3], -1, conv.shape[-1])

    # Permutation indices
    paxes = (0, axes[-1], *axes[1:spatial_dims - 1], axes[-2], axes[-3])

    # We want to iterate through the group input channels during the sep conv.
    weight = [dweight.permute(1, 0, 2).reshape(-1, 1, dweight.shape[-1])
              for dweight in weight]

    # Iterate through the dimensions
    params = (stride, padding, dilation)
    for dstride, dpadding, ddilation, dweight in zip(*params, weight):
        original_shape = conv.shape
        conv = conv.reshape(-1, *conv.shape[-2:])
        conv = torch.nn.functional.conv1d(
            conv, dweight, stride=dstride,
            padding=dpadding, dilation=ddilation,
            groups=conv.shape[-2])

        conv = conv.reshape(*original_shape[:-2], -1, conv.shape[-1])
        conv = conv.permute(paxes)

    final_perm = (0, axes[-2], *axes[1:-2], -1)
    conv = conv.permute(final_perm)
    conv = conv.reshape(batch, group_ic, output_channels, *conv.shape[2:])
    conv = conv.sum(1)
    return conv
