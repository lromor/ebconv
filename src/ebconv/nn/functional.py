"""Definition of main workhorse functions made for pytorch."""

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

import torch

from ebconv.splines import BSplineElement
from ebconv.utils import conv_output_shape
from ebconv.utils import sampling_domain


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


_D2F = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d
}


_D2I = {
    0: 'a',
    1: 'b',
    2: 'c'
}


def _conv_separable_native(input_, weight, *args, **kwargs):
    spatial_dims = len(input_.shape[2:])

    # Compute outer product using einsum.
    einsum_eq = ['ij' + _D2I[dim] for dim in range(spatial_dims)]
    einsum_eq = ','.join(einsum_eq)
    einsum_eq += '->'
    einsum_eq += 'ij' + ''.join([_D2I[dim] for dim in range(spatial_dims)])
    weight = torch.einsum(einsum_eq, *weight)

    return _D2F[spatial_dims](input_, weight, *args, **kwargs)


def convdd_separable(input_: torch.Tensor, weight: Iterable[torch.Tensor],
                     bias: Optional[torch.Tensor] = None,
                     stride: Union[int, Tuple[int, ...]] = 1,
                     padding: Union[int, Tuple[int, ...]] = 0,
                     dilation: Union[int, Tuple[int, ...]] = 1,
                     groups: int = 1, use_native: bool = True):
    """Compute a separable d-dimensional convolution.

    use_native specifies if it's preferred to use the original
    torch.nn.functional.conv<d>d backends which support the convolution
    up to three dimensional.
    """
    spatial_shape = input_.shape[2:]
    spatial_dims = len(spatial_shape)

    # There should be a weight per dimension.
    assert spatial_dims == len(weight)
    assert groups > 0

    if use_native and spatial_dims in _D2F:
        return _conv_separable_native(
            input_, weight, bias=bias, stride=stride,
            padding=padding, dilation=dilation, groups=groups)

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


def sample_basis(spline, sampling_x):
    """Sample a spline object."""
    support_bounds = spline.support_bounds()
    support_bounds = support_bounds if len(support_bounds.shape) == 2 \
        else support_bounds[None, :]

    samples = []
    shift = []
    for domain, (lower, upper) in zip(sampling_x, support_bounds):
        x = domain[(domain >= lower) & (domain <= upper)]
        if len(x) == 0:
            return torch.Tensor(), None

        # Sample the spline over x.
        spline_w = torch.Tensor(spline(x))
        new_center = int(np.floor(np.mean(x)))
        samples.append(spline_w)
        shift.append(new_center)
    return tuple(samples), tuple(shift)


def cbsconv(input_: torch.Tensor, kernel_size: Tuple[int, ...],
            weights: torch.Tensor, c: np.ndarray,
            s: np.ndarray, k: Iterable[Tuple[int, ...]],
            bias=None, stride=1, padding=0, dilation=1,
            groups=1) -> torch.Tensor:
    """Compute a bspline separable convolution.

    input.shape = batch, iC, iX, iY, iZ, ...
    kernel_size.shape = kH, kW, kD, ...
    weights.shape = oC, iC / groups, nc
    c.shape = nc, idim
    s.shape = nc, idim
    k.shape = nc, idim

    For now we only allow the weights to be multichannel.
    Ideally we might want to have a different number of
    bases parameters per channel.
    For now it's possible to do that
    by manually generating more parameters since no
    optimization is yet available for the multichannel
    convolution. For now we ignore the gradient part
    of the parameters, making this function unsuitable
    for training.
    """
    output_channels, input_channels = weights.shape[0], weights.shape[1]
    spatial_shape = input_.shape[2:]
    spatial_dims = len(spatial_shape)
    k = np.array(k)

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
    output_shape = np.array(conv_output_shape(
        input_.shape, (output_channels, input_channels, *kernel_size),
        stride=stride, padding=padding,
        dilation=dilation))

    # Sampling domain.
    sampling_x = [sampling_domain(s) for s in kernel_size]

    # List of weights of the bases for which
    # the support is contained in the sampling region
    # of the kernel
    relevant_weights = []

    # Dict of cached dd convolutions.
    # This is a light optimization that can be
    # useful in case of identical basis functions with identical
    # decimal centers.
    cached_convs = {}
    bases = []
    for i, (spline_c, spline_s, spline_k) in enumerate(zip(c, s, k)):
        # Sample the basis function and check if it's inside the
        # sampling domain.
        spline = BSplineElement.create_cardinal(spline_c, spline_s, spline_k)
        spline_w, shift = sample_basis(spline, sampling_x)

        # Skip this convolution, it's not inside the sampling
        # region.
        if len(spline_w) == 0:
            continue

        # For each center convolve w.r.t the input
        if spline_w in cached_convs:
            conv = cached_convs[spline_w]
        else:
            sw_tensor = [torch.Tensor(w).reshape(1, 1, -1) for w in spline_w]
            # Convolve the input with the basis.
            conv = convdd_separable(input_, sw_tensor, bias, stride=stride,
                                    padding=padding, dilation=dilation)

            # Cache it.
            cached_convs[spline_w] = conv

        # Crop the values.
        spatial_os = output_shape[2:]
        spatial_cs = np.array(conv.shape)[2:]
        cropr = (spatial_cs - spatial_os) // 2 - shift * np.array(dilation)
        cropl = spatial_cs - cropr - spatial_os
        crop_ = np.array((cropl, cropr))
        crop_ = crop_.T.flatten()

        conv = crop(conv, crop_)
        assert (conv.shape == output_shape).all()

        # At this point we have for each input channel the convolution
        # with a basis.
        bases.append(conv)

        # Add the weight
        relevant_weights.append(weights[..., i])

    if len(relevant_weights) == 0:
        return torch.zeros(tuple(output_shape))

    weights = torch.stack(relevant_weights, dim=-1)

    # Stack the bases before doing the tp with the weights
    stacked_convs = torch.stack(bases, dim=2)
    group_input_channels = weights.shape[1]

    stacked_convs = stacked_convs.reshape(
        stacked_convs.shape[0], groups, group_input_channels,
        *stacked_convs.shape[2:])

    output_channels = []
    for i, weight in enumerate(weights):
        input_idx = (i % groups) * group_input_channels
        conv = stacked_convs[:, input_idx, ...]
        # Contract everything except the first two dims.
        res = torch.tensordot(weight, conv, dims=[(0, 1), (1, 2)])
        output_channels.append(res)

    result = torch.stack(output_channels, dim=1)
    assert (result.shape == output_shape).all()
    return result


def crop(input_: torch.Tensor, crop_: List[Tuple[int, ...]]) -> torch.Tensor:
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
    assert len(crop_) == len(input_.shape) - 2

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
