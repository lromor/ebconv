"""Definition of main workhorse functions made for pytorch."""

from typing import List, Tuple, Union

from ebconv.splines import BSplineElement
from ebconv.utils import conv_output_shape, round_modf

import numpy as np

import torch
from torch.nn.functional import conv1d


def _bsconv_subconv(input_, basis_paramaters, stride, padding, dilation):
    """Convolve a single basis."""
    axes = np.arange(input_.shape)
    conv = input_

    for dshift, s, k, dstride, dpadding, ddilation \
        in zip(*basis_paramaters, stride,
               padding, dilation):
        # Store the original shape of the input tensor.
        initial_shape = conv.shape
        non_spatial_shape = initial_shape[:3]

        # Compute the number of 1d strides
        strides = np.prod(initial_shape[3:-1])

        # Create the basis function and sample it.
        spline = BSpline.create_cardinal(dshift, s, k)

        # Sample the support of the spline
        lb, ub = np.floor(spline.get_support_interval())
        width = int(ub - lb) + 1
        x = np.arange(lb, ub) + 0.5
        spline_w = torch.Tensor(spline(x)).reshape(1, 1, -1)

        # Compute the theoretical ddconvolution
        # Perform the 1d convolution
        conv = conv1d(conv.reshape(*non_spatial_shape, -1),
                      spline_w, dstride, dpadding, ddilation)

        # Add at the end extra values to have the right shape
        # to remove the excess of values due to tha fake ddim
        # 1d conv.
        conv = torch.cat([conv, torch.empty(*non_spatial_shape, 2 * width)])

        # Remove the excess from the 1d convolution.
        conv = conv.reshape(*non_spatial_shape, strides, -1)
        conv = conv[..., :-2 * width]
        conv = conv.reshape(*initial_shape[:-1], -1)

        # Permute to axes to have the one we are dealing with as last.
        axes = np.roll(axes, 1)
        conv = conv.permute(0, 1, *axes)
    return conv


def _cbsconv_impl(input_, weights, c, s, k,
                  bias, stride, padding, dilation, groups):
    """Compute a bspline separable convolution.

    input.shape = batch, iC, iX, iY, iZ....
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
    convolution.
    """
    # Compute the minimum size of the kernel containing the
    # centers. The kernel can be rectangular but the center
    # of the kernel will always be the origin in the
    # centers space.
    bc = np.argmax(np.abs(c), axis=0)  # Boundary centers
    cmax, smax, kmax = c[bc], s[bc], k[bc]
    k_shape = (weights.shape[0], weights.shape[1],
               *(cmax * 2 + smax * kmax / 2))

    # Build the output tensor.
    output_shape = conv_output_shape(
        input_.shape, k_shape, stride=stride, padding=padding,
        dilation=dilation)
    output_shape = np.array(output_shape)

    # Dict of cached dd convolutions.
    # This is a light optimization that can be extremely
    # useful in case of identical basis functions with identical
    # decimal centers.
    cached_convs = {}
    bases = []
    for cc, cs, ck in zip(c, s, k):
        dshifts, ishifts = tuple(zip(*(round_modf(v) for v in cc)))
        hashable_params = (dshifts, cs, ck)

        # For each center convolve w.r.t the input
        if hashable_params in cached_convs:
            conv = cached_convs[hashable_params]
        else:
            conv = _bsconv_subconv(input_, hashable_params,
                                   stride=stride, padding=padding,
                                   dilation=dilation)
            cached_convs[hashable_params] = conv

        #  it to fit the output.
        cropl = (output_shape - np.array(conv.shape)) // 2
        cropr = conv.shape - cropl + output_shape
        crop = np.array((cropl, cropr)).T

        # Translate and crop the convolution to fit the output..
        conv = cropped_translate(conv,
                                 ishifts * dilation,
                                 crop,
                                 mode='constant', value=0)
        assert conv.shape == output_shape

        bases.append(conv)

    # Stack the bases before doing the tp with the weights
    stacked_convs = torch.stack(bases)
    return torch.tensordot(weights, stacked_convs)


def cbsconv(input_: torch.Tensor, kernel_size: Tuple[int, ...],
            weights: torch.Tensor, c: torch.Tensor,
            s: torch.Tensor, k: torch.Tensor,
            bias=None, stride=1, padding=0, dilation=1,
            groups=1) -> torch.Tensor:
    """Interface for the cardinal BSplines d-dimensional convolution."""
    # TODO Allow to specify a "virtual" kernel shape.
    # This allows to explicity define the output shape as if a kernel
    # of size kernel_size was used.
    if kernel_size is not None:
        raise NotImplementedError('Specifing the kernel size is'
                                  'not yet implemented.')
    return _cbsconv_impl(input_, kernel_size, weights, c, s, k, bias, stride,
                         padding, dilation, groups)


def _crop_impl(x: torch.Tensor, crop: List[Tuple[int, ...]]):
    # Construct the bounds and padding list of tuples
    slices = [...]
    for pl, pr in crop:
        pl = pl if pl else None
        pr = pr if pr else None
        slices.append(slice(pl, -pr))

    slices = tuple(slices)

    # Apply the crop and return
    return x[slices]


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

    # Call the actual implementation
    return _crop_impl(x, crop)


def _translate_impl(x: torch.Tensor, shift: Tuple[int, ...],
                    mode='constant', value=0) -> torch.Tensor:
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
    # Do some basic checks
    assert len(shift) == len(x.shape) - 2

    # Call the actual implementation
    return _translate_impl(x, shift, mode, value)


def cropped_translate(x: torch.Tensor, shift: Tuple[int, ...],
                      crop: List[Tuple[int, ...]],
                      mode='constant', value=0) -> torch.Tensor:
    """Apply a translation and crop the result."""
    crop = crop if not isinstance(crop, int) else [(crop. crop)] * len(shift)
    y = translate(x, shift, mode, value)
    return crop(y, crop)
