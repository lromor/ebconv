"""Definition of main workhorse functions made for pytorch."""

from typing import Iterable, List, Tuple

from ebconv.splines import BSplineElement
from ebconv.utils import conv_output_shape
from ebconv.utils import sampling_domain

import numpy as np

import torch
from torch.nn.functional import conv1d


def _separable_conv(input_, weights, stride, padding, dilation):
    """Compute a separable convolution."""
    iC = input_.shape[1]
    non_spatial_shape = input_.shape[:2]
    spatial_dims = len(input_.shape[2:])
    paxes = np.roll(np.arange(2, spatial_dims + 2), 1)
    conv = input_

    # We convolve the separable basis
    params = (weights, stride, padding, dilation)
    for dweights, dstride, dpadding, ddilation in zip(*params):
        # Store the original shape of the input tensor.
        initial_shape = conv.shape

        # Compute the number of 1d strides
        strides = np.prod(initial_shape[2:-1], dtype=int)
        dweights = dweights.reshape(1, -1)
        dweights = torch.stack([dweights for i in range(iC)])
        width = dweights.shape[-1]

        # Compute the theoretical ddconvolution
        # Perform the 1d convolution
        conv = conv1d(conv.reshape(*non_spatial_shape, -1),
                      dweights, stride=dstride, padding=dpadding,
                      dilation=ddilation, groups=iC)

        # Add at the end extra values to have the right shape
        # to remove the excess of values due to tha fake ddim
        # 1d conv.
        conv = torch.cat([conv, torch.empty(*non_spatial_shape, width - 1)],
                         dim=-1)

        # Remove the excess from the 1d convolution.
        conv = conv.reshape(*non_spatial_shape, strides, -1)
        crop = -(width - 1)
        crop = None if crop == 0 else crop
        conv = conv[..., :crop]
        conv = conv.reshape(*initial_shape[:-1], -1)

        # Permute to axes to have the one we are dealing with as last.
        conv = conv.permute(0, 1, *paxes)
    return conv


def sample_basis(spline, sx):
    """Sample a spline object."""
    support_bounds = spline.support_bounds()
    support_bounds = support_bounds if len(support_bounds.shape) == 2 \
        else support_bounds[None, :]

    samples = []
    shift = []
    for domain, (lb, ub) in zip(sx, support_bounds):
        x = domain[(domain >= lb) & (domain <= ub)]
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
    oC, iC = weights.shape[0], weights.shape[1]
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
        input_.shape, (oC, iC, *kernel_size), stride=stride, padding=padding,
        dilation=dilation))

    # Sampling domain.
    sx = [sampling_domain(s) for s in kernel_size]

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
    for i, (cc, cs, ck) in enumerate(zip(c, s, k)):
        # Sample the basis function and check if it's inside the
        # sampling domain.
        spline = BSplineElement.create_cardinal(cc, cs, ck)
        spline_w, shift = sample_basis(spline, sx)

        # Skip this convolution, it's not inside the sampling
        # region.
        if len(spline_w) == 0:
            continue

        # For each center convolve w.r.t the input
        if spline_w in cached_convs:
            conv = cached_convs[spline_w]
        else:
            # Convolve the input with the basis.
            conv = _separable_conv(input_, spline_w, stride=stride,
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
    group_iC = weights.shape[1]

    stacked_convs = stacked_convs.reshape(
        stacked_convs.shape[0], groups, group_iC,
        *stacked_convs.shape[2:])

    output_channels = []
    for i, w in enumerate(weights):
        input_idx = (i % groups) * group_iC
        cv = stacked_convs[:, input_idx, ...]
        # Contract everything except the first two dims.
        r = torch.tensordot(w, cv, dims=[(0, 1), (1, 2)])
        output_channels.append(r)

    result = torch.stack(output_channels, dim=1)
    assert (result.shape == output_shape).all()
    return result


def crop(x: torch.Tensor, crop: List[Tuple[int, ...]]) -> torch.Tensor:
    """Crop the tensor using an array of values.

    Opposite operation of pad.
    Args:
        x: Input tensor.
        crop: Tuple of crop values.

    Returns:
        Cropped tensor.

    """
    assert len(crop) % 2 == 0
    crop = [(crop[i], crop[i + 1]) for i in range(0, len(crop) - 1, 2)]
    assert len(crop) == len(x.shape) - 2

    # Construct the bounds and padding list of tuples
    slices = [...]
    for pl, pr in crop:
        pl = pl if pl != 0 else None
        pr = -pr if pr != 0 else None
        slices.append(slice(pl, pr, None))

    slices = tuple(slices)

    # Apply the crop and return
    return x[slices]


def translate(x: torch.Tensor, shift: Tuple[int, ...],
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
    for s in reversed(shift):
        shift = abs(s)
        if s > 0:
            paddings.append(shift)
            paddings.append(0)
            slices.insert(1, slice(0, -shift if shift else None))
        else:
            paddings.append(0)
            paddings.append(shift)
            slices.insert(1, slice(shift, None))

    slices = tuple(slices)
    y = torch.nn.functional.pad(x, paddings, mode=mode, value=value)

    # Apply the crop and return
    return y[slices]
