from collections import defaultdict

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

import torch

from ebconv.splines import BSplineElement
from ebconv.utils import convolution_output_shape
from ebconv.kernel import sampling_domain


def _convdd_separable_per_filter(input_, weight, bias, stride, dilation):
    """Implement the separable convolution for a single filter.

    This functions takes care of evaluating for each group
    the separable convolution of a single filter. The filters
    should contracted afterwards.
    """
    batch = input_.shape[0]
    output_channels = weight[0].shape[0]
    spatial_dims = len(input_.shape[2:])

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
        conv = torch.cat([conv, torch.empty(*conv.shape[:2], overlap_size)],
                         dim=-1)

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


class UnivariateCardinalBSpline(torch.autograd.Function):
    """Autograd for sampling a cardinal bspline."""

    @staticmethod
    # pylint: disable=arguments-differ
    def forward(ctx, input_, c, s, k):
        # ctx is a context object that can be used to stash information
        # for backward computation
        c_n = c.item()
        s_n = s.item()
        x_n = input_.numpy()

        spline = BSplineElement.create_cardinal(c_n, s_n, k)
        dspline = spline.derivative()

        # pylint: disable=E1102
        ctx.s = s

        y = spline(x_n)
        # pylint: disable=E1102
        ctx.derivative = torch.tensor(dspline(x_n), dtype=input_.dtype)
        # pylint: disable=E1102
        return torch.tensor(y, dtype=input_.dtype)

    @staticmethod
    # pylint: disable=arguments-differ
    def backward(ctx, grad_output):
        if ctx.derivative is None:
            return (None,) * 4

        c_grad = -ctx.derivative * grad_output
        s_grad = -ctx.derivative / (ctx.s * ctx.s) * grad_output
        return None, c_grad, s_grad, None


_KOptionalType = Union[int, Iterable[Tuple[int, ...]]]


def cbspline(sampling_x: torch.Tensor, c: torch.Tensor, s: torch.Tensor,
             k: _KOptionalType) -> List[torch.Tensor]:
    """Public interface to the functional to sample univariate bspines."""
    return [UnivariateCardinalBSpline.apply(x, ci, si, ki)
            for x, ci, si, ki in zip(sampling_x, c, s, k)]



class SplineWeightsHashing():
    """Class to define how to has the sampling of the bsplines."""
    def __init__(self, spline_weights: Iterable[torch.Tensor]) -> None:
        self.l_t = spline_weights

    def __hash__(self):
        return hash(tuple(tuple(w_i.data.numpy()) for w_i in self.l_t))


def _cbsconv_params(input_, output, kernel_x, weights, c, s, k, dilation):
    """Precompute the list of parameters required for each conv.

    Should return a list with each item:
    spline samples, input slice, weights and shifts.
    """
    assert c.shape == s.shape
    n_c = weights.shape[-1]
    new_shape = (-1, len(input_.shape[3:]))

    total_c = c.reshape(new_shape)
    total_s = s.reshape(new_shape)

    weights_map = defaultdict(lambda: (set(), []))
    for i, (b_c, b_s) in enumerate(zip(total_c, total_s)):
        # For each dimension, crop the sampling values to fit
        # only the support of the spline.
        bounds = BSplineElement.create_cardinal(
            b_c.data.numpy(), b_s.data.numpy(), k
        ).support_bounds().reshape(-1, 2)

        # Select only the part of the domain that intesects with the
        # support.
        spline_x = [d_x[(d_x >= l_b) & (d_x <= u_b)]
                    for d_x, (l_b, u_b) in zip(kernel_x, bounds)]

        if sum([len(s_x) == 0 for s_x in spline_x]) > 0:
            continue

        # Sample the torch weights carrying the gradient of c and s.
        w_t = cbspline(spline_x, b_c, b_s, k)

        # Calculate new shift (from left)
        shift = [ddilation * len(torch.where(d_x < s_x[0])[0])
                 for d_x, s_x, ddilation in zip(kernel_x, spline_x, dilation)]

        # Make the weights hashable.
        hashable_weights = SplineWeightsHashing(w_t)
        group_index = i // n_c
        n_index = i % n_c
        g_set, shifts_and_weights = weights_map[hashable_weights]
        g_set.add(group_index)
        shifts_and_weights.append(
            (group_index, shift, weights[group_index, ..., n_index]))

    return [(key.l_t, sorted(i_set), shifts_and_weights)
            for key, (i_set, shifts_and_weights) in weights_map.items()]


def cbsconv(input_: torch.Tensor, kernel_size: Tuple[int, ...],
            weights: torch.Tensor, c: torch.Tensor,
            s: torch.Tensor, k: int,
            bias: Optional[torch.Tensor] = None,
            stride: Union[int, Tuple[int, ...]] = 1,
            padding: Union[int, Tuple[int, ...]] = 0,
            dilation: Union[int, Tuple[int, ...]] = 1,
            groups: int = 1) -> torch.Tensor:
    """Compute a bspline separable convolution.

    input.shape = batch, iC, iX, iY, iZ, ...
    kernel_size.shape = kH, kW, kD, ...
    weights.shape = oC, iC / groups, nc
    c.shape = groups, nc, idim
    s.shape = groups, nc, idim

    The group notion is different from the usual torch one.
    For each group, we require the same number of basis parameters
    and values (c,s,k) across the filters. The weight though can differ
    both group and filter wise.
    """
    spatial_shape = input_.shape[2:]
    spatial_dims = len(spatial_shape)
    batch = input_.shape[0]
    input_channels = input_.shape[1]
    output_channels, group_ic = weights.shape[0], weights.shape[1]
    n_c = weights.shape[-1]
    dtype = input_.dtype
    group_oc = output_channels // groups

    if not isinstance(k, Iterable):
        k = ((k,) * spatial_dims)

    if not isinstance(stride, Tuple):
        stride = ((stride,) * spatial_dims)

    if not isinstance(padding, Tuple):
        padding = ((padding,) * spatial_dims)

    if not isinstance(dilation, Tuple):
        dilation = ((dilation,) * spatial_dims)

    if (np.array(spatial_shape) < np.array(kernel_size)).any():
        raise RuntimeError("Kernel size can't be greater than actual"
                           'input size')

    # Output tensor shape.
    output_shape = np.array(convolution_output_shape(
        input_.shape, (output_channels, input_channels, *kernel_size),
        stride=stride, padding=padding,
        dilation=dilation))
    output_spatial_shape = output_shape[2:]

    # Add the padding
    pad = [(p, p) for p in padding]
    pad = sum(pad, ())
    input_ = torch.nn.functional.pad(input_, pad)
    spatial_shape = input_.shape[2:]

    # Sampling domain.
    kernel_x = [torch.tensor(sampling_domain(s), dtype=dtype)
                for s in kernel_size]

    input_ = input_.reshape(
        batch, groups, group_ic, *spatial_shape)
    weights = weights.reshape(
        groups, group_oc, group_ic, n_c)
    output = torch.zeros(
        batch, groups, group_oc, *output_spatial_shape)

    # Presample the bsplines weights.
    bases_convs_params = _cbsconv_params(
        input_, output, kernel_x, weights, c, s, k, dilation)

    for spline_ws, input_indices, shifts_and_weights in bases_convs_params:
        g2i = {g: i for i, g in enumerate(input_indices)}
        b_input = input_[:, input_indices, ...]
        b_input = b_input.reshape(batch, -1, *spatial_shape)
        b_groups = len(g2i)

        sep_conv_groups = b_input.shape[1]

        # We need only a single output channel for the single
        # basis weights convolution but many for each input.
        spline_ws = [s_w.reshape(1, 1, -1).repeat(sep_conv_groups, 1, 1)
                     for s_w in spline_ws]
        basis_conv = convdd_separable(
            b_input, spline_ws, stride=stride, dilation=dilation,
            groups=sep_conv_groups)
        b_spatial_shape = basis_conv.shape[2:]
        # Each input is separately convolved. We now split the output
        # per-group.
        basis_conv = basis_conv.reshape(
            batch, b_groups, group_ic, *b_spatial_shape)

        # Duplicate for each group output channel the result.
        # The final shape should be batch, b_groups, group_oc, group_ic, ...
        basis_conv = basis_conv.repeat_interleave(group_oc, dim=1)
        basis_conv = basis_conv.reshape(
            batch, b_groups, group_oc, group_ic, *b_spatial_shape)

        # Iterate through each group and perform the required crop.
        for group_idx, shift, group_weight in shifts_and_weights:
            b_group = g2i[group_idx]
            group_conv = basis_conv[:, b_group, ...]
            shape_diff = b_spatial_shape - output_spatial_shape
            cropl = shift
            cropr = shape_diff - cropl
            crop_ = np.array((cropl, cropr))
            crop_ = crop_.T.flatten()
            group_conv = crop(group_conv, crop_)

            # Complete the tensordot and sum over the group_ic
            group_weight = group_weight.reshape(
                *group_weight.shape, *((1,) * spatial_dims))
            g_out = (group_conv * group_weight).sum(dim=2)
            output[:, group_idx, ...] += g_out

    return output.reshape(*output_shape)


def crop(input_: torch.Tensor, crop_: Tuple[int, ...]) -> torch.Tensor:
    """Crop the tensor using an array of values.

    Opposite operation of pad.
    Args:
        x: Input tensor.
        crop: Tuple of crop values.

    Returns:
        Cropped tensor.

    """
    assert len(crop_) % 2 == 0
    crop_ = [(crop_[i], crop_[i + 1]) for i in range(0, len(crop_) - 1, 2)]
    assert len(crop_) <= len(input_.shape)

    # Construct the bounds and padding list of tuples
    slices = [...]
    for left, right in crop_:
        left = left if left != 0 else None
        right = -right if right != 0 else None
        slices.append(slice(left, right, None))

    slices = tuple(slices)

    # Apply the crop and return
    return input_[slices]


def translate(input_: torch.Tensor, shift: Tuple[int, ...],
              mode='constant', value=0) -> torch.Tensor:
    """Translate the input.

    Args:
        x: Input tensor
        shift: Represents the shift values for each spatial axis.
        mode: Same argument as torch.nn.functional.pad
        value: Same as above.

    Returns:
        Translated tensor.

    """
    # Construct the bounds and padding list of tuples
    paddings = []
    slices = [...]
    for axis_shift in reversed(shift):
        abs_shift = abs(axis_shift)
        if axis_shift > 0:
            paddings.append(abs_shift)
            paddings.append(0)
            slices.insert(1, slice(0, -abs_shift if abs_shift else None))
        else:
            paddings.append(0)
            paddings.append(abs_shift)
            slices.insert(1, slice(abs_shift, None))

    slices = tuple(slices)
    output = torch.nn.functional.pad(input_, paddings, mode=mode, value=value)

    # Apply the crop and return
    return output[slices]
