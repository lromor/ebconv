"""Implementation of the cardinabl b-spline convolution"""

from collections import defaultdict

from functools import reduce

from typing import Iterable, List, Optional, Tuple, Union

import warnings

import numpy as np

import torch

from ebconv.nn.functional import crop

from ebconv.splines import BSplineElement
from ebconv.utils import convolution_output_shape
from ebconv.kernel import sampling_domain


class UnivariateCardinalBSpline(torch.autograd.Function):
    """Autograd for sampling a cardinal bspline."""

    @staticmethod
    # pylint: disable=arguments-differ
    def forward(ctx, input_, c, s, k):
        # ctx is a context object that can be used to stash information
        # for backward computation
        device = input_.device

        x_n = input_.data.cpu().numpy()
        c_n = c.data.item()
        s_n = s.data.item()

        spline = BSplineElement.create_cardinal(c_n, s_n, k)

        # pylint: disable=E1102
        ctx.input_ = input_
        ctx.s = s
        ctx.c = c

        y = spline(x_n)

        if k == 0:
            if c.requires_grad or s.requires_grad:
                warnings.warn('0-order spline gradient is not supported.')
            ctx.derivative = None
        else:
            dspline = spline.derivative()
            ctx.derivative = torch.tensor(
                dspline(x_n), dtype=input_.dtype, device=device)

        # pylint: disable=E1102
        return torch.tensor(y, dtype=input_.dtype, device=device)

    @staticmethod
    # pylint: disable=arguments-differ
    def backward(ctx, grad_output):
        if ctx.derivative is None:
            return (None,) * 4

        c_grad = -ctx.derivative * grad_output
        s_grad = c_grad * (ctx.input_ - ctx.c) / ctx.s
        return None, c_grad, s_grad, None


def cbspline(sampling_x: torch.Tensor, c: torch.Tensor, s: torch.Tensor,
             k: int) -> List[torch.Tensor]:
    """Public interface to the functional to sample univariate bspines."""
    return [UnivariateCardinalBSpline.apply(x, ci, s, k)
            for x, ci in zip(sampling_x, c)]


class SplineWeightsHashing():
    """Class to define how to has the sampling of the bsplines."""
    def __init__(self, spline_weights: Iterable[torch.Tensor],
                 shift_stride: int) -> None:
        self.l_t = spline_weights
        self.shift_stride = shift_stride

    def __hash__(self):
        h_1 = hash(tuple(tuple(w_i.data.cpu().numpy()) for w_i in self.l_t))
        h_2 = hash(self.shift_stride)
        return h_1 ^ h_2

    def __eq__(self, other):
        return hash(self) == hash(other)


def _cbsconv_params(input_, kernel_x, weights, c, s, k,
                    dilation, stride):
    """Precompute the list of parameters required for each conv.

    Should return a list with each item:
    spline samples, input slice, weights and shifts.
    """
    n_c = weights.shape[-1]
    c = c.reshape(-1, c.shape[-1])
    s = s.flatten()

    weights_map = defaultdict(lambda: ([], []))
    for i, (b_c, b_s) in enumerate(zip(c, s)):
        # For each dimension, crop the sampling values to fit
        # only the support of the spline.
        spline = BSplineElement.create_cardinal(
            b_c.data.cpu().numpy(), b_s.item(), k)
        bounds = spline.support_bounds().reshape(-1, 2)

        # Select only the part of the domain that intesects with the
        # support.
        spline_x = [d_x[(d_x >= l_b) & (d_x <= u_b)]
                    for d_x, (l_b, u_b) in zip(kernel_x, bounds)]

        # Empty result? The spline is outside the virtual kernel
        # region. Skip.
        if sum([len(s_x) == 0 for s_x in spline_x]) > 0:
            continue

        # Sample the torch weights carrying the gradient of c and s.
        w_t = cbspline(spline_x, b_c, b_s, k)

        # Calculate new shift (from left)
        shift = [ddilation * len(torch.where(d_x < s_x[0])[0])
                 for d_x, s_x, ddilation in zip(kernel_x, spline_x, dilation)]

        # What happens if a basis is 3 pixeks left but the stride is 2?
        # This values serves the purpose of computing the right convolution
        # if a stride != 1 is chosen.
        shift_stride_rem = tuple(s_h % s_t for s_h, s_t in zip(shift, stride))

        # Make the weights hashable.
        hashable_weights = SplineWeightsHashing(w_t, shift_stride_rem)
        basis_group_index = i // n_c
        n_index = i % n_c
        shifts_and_weights, samples = weights_map[hashable_weights]
        shifts_and_weights.append((basis_group_index, shift, n_index))
        samples.append(w_t)

    params = []
    for key, (shifts_and_weights, samples) in weights_map.items():
        samples_mean = list(zip(*samples))
        samples_mean = [torch.stack(sample_axis).mean(0)
                        for sample_axis in samples_mean]
        params.append((samples_mean, key.shift_stride, shifts_and_weights))

    return params


def _cbs_convdd_separable(input_, weights, stride, dilation):
    if not weights:
        return input_
    original_shape = input_.shape
    input_ = input_.reshape(input_.shape[0], -1, input_.shape[-1])
    weight = weights.pop().reshape(1, 1, -1).repeat(input_.shape[1], 1, 1)
    conv = torch.nn.functional.conv1d(
        input_, weight, stride=stride.pop(),
        dilation=dilation.pop(), groups=weight.shape[0])
    conv = conv.reshape(*original_shape[:-1], -1)
    # Permute axes
    axes = np.arange(len(conv.shape))
    paxes = np.roll(axes[2:], 1)
    return _cbs_convdd_separable(
        conv.permute(0, 1, *paxes), weights, stride, dilation)


def _cbsconv_separable_impl(input_, kernel_size, weights, c, s, k,
                            bias, stride, padding, dilation):
    """Compute a bspline separable convolution."""
    dtype = input_.dtype
    device = input_.device
    spatial_shape = input_.shape[2:]
    spatial_dims = len(spatial_shape)
    batch = input_.shape[0]
    input_channels = input_.shape[1]
    output_channels, group_ic = weights.shape[0], weights.shape[1]
    n_c = weights.shape[-1]
    basis_groups = c.shape[0]
    basis_groups_oc = output_channels // basis_groups

    # Sampling domain.
    # pylint: disable=E1102
    kernel_x = [torch.tensor(sampling_domain(s), dtype=dtype, device=device)
                for s in kernel_size]

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

    weights = weights.reshape(
        basis_groups, basis_groups_oc, group_ic, n_c)
    output = torch.zeros(
        basis_groups, basis_groups_oc, batch, *output_spatial_shape,
        dtype=dtype, device=device)
    # Let's move the groups as first dimension as we will access them by index
    # multiple times and they are supposed to be less than the batch size.
    #input_ = input_.transpose(0, 1)
    conv_input = input_.reshape(batch, -1, *spatial_shape)

    # Presample the bsplines weights.
    bases_convs_params = _cbsconv_params(
        input_, kernel_x, weights, c, s, k, dilation, stride)

    for spline_ws, shift_stride, shifts_and_weights in bases_convs_params:
        # Crop the input to fit the striding.
        crop_shift_stride = [(s_s, 0) for s_s in shift_stride]
        crop_shift_stride = sum(crop_shift_stride, ())
        b_input = crop(conv_input, crop_shift_stride)

        # At this point we just need to subsequently do 1d convolutions
        # and permute the spatial axes in the meantime.
        basis_conv = _cbs_convdd_separable(
            b_input, spline_ws, list(stride), list(dilation))

        b_spatial_shape = basis_conv.shape[2:]

        # Iterate through each group and perform the required crop.
        for basis_group_idx, shift, n_index in shifts_and_weights:
            b_weights = weights[basis_group_idx, ..., n_index]
            shape_diff = b_spatial_shape - output_spatial_shape
            cropl = (shift - np.array(shift_stride)) // np.array(stride)
            cropr = shape_diff - cropl
            crop_ = np.array((cropl, cropr))
            crop_ = crop_.T.flatten()
            cropped_conv = crop(basis_conv, crop_)
            t_d = torch.tensordot(b_weights, cropped_conv, dims=[(1,), (1,)])
            output[basis_group_idx] += t_d

    output = output.reshape(output_channels, batch, *output_spatial_shape)
    output = output.transpose(0, 1)
    output = output.reshape(*output_shape)
    output = output if bias is None \
        else output + bias.reshape(1, -1, *((1,) * spatial_dims))
    return output


_TORCH_CONVS_MAP = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d,
}


def _cbsconv_sample_impl(input_, kernel_size, weights, c, s, k, bias,
                         stride, padding, dilation):
    """Implementation of the simple bspline sampling method."""
    dtype = input_.dtype
    device = input_.device
    spatial_shape = input_.shape[2:]
    spatial_dims = len(spatial_shape)
    basis_groups = c.shape[0]
    weights = weights.reshape(basis_groups, -1, *weights.shape[1:])

    # Sampling domain.
    # pylint: disable=E1102
    kernel_x = [torch.tensor(sampling_domain(size), dtype=dtype, device=device)
                for size in kernel_size]

    kernel = []
    for i in range(basis_groups):
        bases = []
        g_weights = weights[i]
        for g_c, g_s in zip(c[i], s[i]):
            samples = cbspline(kernel_x, g_c, g_s, k)
            bases.append(
                reduce(lambda x, y: torch.tensordot(x, y, dims=0), samples))
        bases = torch.stack(bases)
        kernel.append(
            torch.tensordot(g_weights, bases, dims=1))

    kernel = torch.cat(kernel)
    conv = _TORCH_CONVS_MAP[spatial_dims]
    return conv(input_, kernel, bias, stride=stride,
                padding=padding, dilation=dilation)


def cbsconv(input_: torch.Tensor, kernel_size: Tuple[int, ...],
            weights: torch.Tensor, c: torch.Tensor,
            s: torch.Tensor, k: int,
            bias: Optional[torch.Tensor] = None,
            stride: Union[int, Tuple[int, ...]] = 1,
            padding: Union[int, Tuple[int, ...]] = 0,
            dilation: Union[int, Tuple[int, ...]] = 1,
            separable: bool = False) -> torch.Tensor:
    """Compute a bspline separable convolution.

    input.shape = batch, iC, iX, iY, iZ, ...
    kernel_size.shape = kH, kW, kD, ...
    weights.shape = oC, iC, nc
    c.shape = basis_groups, nc, idim
    s.shape = basis_groups, nc

    There are two types of groups. The standard "group"
    as in pytorch is associated with the groups parameter.
    The grouping will only be applied for the weights, meaning
    that the centers and scalings will be shared across the "standard"
    groups but not the weights.
    Another set of groups are the "channel_groups", these channels groups
    allow different centers positions per filter per group of output channels.
    Sample kernel allows to directly sampling the set of
    bsplines in a single kernel and then perform the convolution.
    """
    assert input_.dtype == weights.dtype == c.dtype == s.dtype
    assert c.shape[:-1] == s.shape
    spatial_shape = input_.shape[2:]
    spatial_dims = len(spatial_shape)
    input_channels = input_.shape[1]
    dims = len(kernel_size)
    assert dims == spatial_dims
    _, group_ic = weights.shape[0], weights.shape[1]

    if not isinstance(stride, Tuple):
        stride = ((stride,) * spatial_dims)

    if not isinstance(padding, Tuple):
        padding = ((padding,) * spatial_dims)

    if not isinstance(dilation, Tuple):
        dilation = ((dilation,) * spatial_dims)

    padded_input_shape = np.array(padding) * 2
    padded_input_shape += np.array(spatial_shape)
    if (padded_input_shape < np.array(kernel_size)).any():
        print(kernel_size, spatial_shape, padding)
        raise ValueError(
            "Kernel size can't be greater than actual input size")

    if group_ic != input_channels:
        raise ValueError(
            "Weights second axis(%d) should match"
            " the input channels(%d)."
            % (group_ic, input_channels))

    if dims not in (1, 2, 3) and separable is False:
        raise ValueError(
            "The sampling implementation works only for dims: 1, 2, 3."
            " Found %d dims instead." % dims)

    impl = _cbsconv_sample_impl
    if separable:
        impl = _cbsconv_separable_impl

    return impl(input_, kernel_size, weights, c, s, k,
                bias, stride, padding, dilation)
