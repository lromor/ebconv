"""Implementation of a d-dimensional separable convolution for pytorch."""

from typing import Iterable, Optional, Tuple, Union

import numpy as np

import torch


def _convdd_separable_per_filter(input_, weight, bias, stride, dilation):
    """Implement the separable convolution for a single filter.

    This functions takes care of evaluating for each group
    the separable convolution of a single filter. The filters
    should contracted afterwards.
    NOTE: In this implementation we are reshaping all the spatial
          dimensions in a single long array. It would be more efficient
          to reshape every non-last dimension into "batch". This would
          maybe speedup the parallel computation and simplify the code.
          This is because we just have to multiply for each row/batch
          for the same kernel value.
    """
    batch = input_.shape[0]
    device = input_.device
    output_channels = weight[0].shape[0]
    spatial_dims = len(input_.shape[2:])
    dtype = input_.dtype

    # Set of axes to permute. The first two are skipped
    # as they are the batch and the input channels.
    paxes = np.roll(np.arange(2, spatial_dims + 2), 1)
    conv = input_

    # We convolve the separable kernel.
    # We iterate per dimension.
    params = (weight, stride, dilation)
    for dweights, dstride, ddilation in zip(*params):
        # The iC changes to oC after the first conv
        input_channels = conv.shape[1]
        spatial_shape = conv.shape[2:]

        # Store the original spatial shape of the input tensor.
        width = dweights.shape[-1]
        width = ddilation * (width - 1) + 1

        # Extend the input to fit the striding.
        axis_size = spatial_shape[-1]
        nshifts = axis_size // dstride
        new_axis_size = dstride * (nshifts + 1)
        conv = torch.nn.functional.pad(
            conv, (0, new_axis_size - axis_size))

        # Extending the input to fit the striding might
        # have introduced output elements to be cropped.
        output_axis_size = int((axis_size - width) / dstride + 1)
        crop_ = int(nshifts - width / dstride + 2) - output_axis_size

        # Compute the theoretical ddconvolution
        # Perform the 1d convolution
        conv = torch.nn.functional.conv1d(
            conv.reshape(*conv.shape[:2], -1),
            dweights, bias=bias, stride=dstride, padding=0,
            dilation=ddilation, groups=input_channels)

        # Add at the end extra values to have the right shape
        # to remove the excess of values due to tha fake ddim
        # 1d conv.
        overlap_size = (width - 1) // dstride
        conv = torch.cat(
            [conv, torch.empty(*conv.shape[:2], overlap_size,
                               dtype=dtype, device=device)],
            dim=-1
        )

        # Remove the excess from the 1d convolution.
        conv = conv.reshape(batch, output_channels, *spatial_shape[:-1], -1)
        crop_ += overlap_size
        crop_ = None if crop_ == 0 else -crop_
        conv = conv[..., :crop_]
        # Permute axes to have the one we are dealing with as last.
        conv = conv.permute(0, 1, *paxes)
    return conv


def convdd_separable(input_: torch.Tensor, weight: Iterable[torch.Tensor],
                     bias: Optional[torch.Tensor] = None,
                     stride: Union[int, Tuple[int, ...]] = 1,
                     padding: Union[int, Tuple[int, ...]] = 0,
                     dilation: Union[int, Tuple[int, ...]] = 1,
                     groups: int = 1):
    """Compute a separable d-dimensional convolution."""
    spatial_shape = input_.shape[2:]
    spatial_dims = len(spatial_shape)

    # There should be a weight per dimension.
    assert spatial_dims == len(weight)
    assert groups > 0

    weight = weight[::-1]
    if not isinstance(stride, Tuple):
        stride = ((stride,) * spatial_dims)

    if not isinstance(padding, Tuple):
        padding = ((padding,) * spatial_dims)

    if not isinstance(dilation, Tuple):
        dilation = ((dilation,) * spatial_dims)

    # Add the padding
    pad = [(p, p) for p in padding]
    pad = sum(pad, ())
    input_ = torch.nn.functional.pad(input_, pad)

    group_input_channels = weight[0].shape[1]
    new_ishape = input_.shape[0], groups, -1, *input_.shape[2:]

    # Reshape into batch, group, group filters, ...
    input_ = input_.reshape(new_ishape)

    # Split along the filters the weights
    weight = [[w[:, i, ...].unsqueeze(1) for w in weight]
              for i in range(group_input_channels)]
    return torch.stack([
        _convdd_separable_per_filter(
            input_[:, :, i, ...], weight[i], bias, stride, dilation)
        for i in range(group_input_channels)
    ]).sum(dim=0)
