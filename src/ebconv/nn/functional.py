"""Definition of main workhorse functions made for pytorch."""

from typing import Iterable, List, Tuple

from ebconv.kernel import CardinalBSplineKernel
from ebconv.splines import BSplineElement
from ebconv.utils import conv_output_shape, round_modf

import numpy as np

import torch
from torch.nn.functional import conv1d


def _bsconv_subconv(input_, basis_parameters, stride, padding, dilation):
    """Convolve a single basis."""
    iC = input_.shape[1]
    non_spatial_shape = input_.shape[:2]
    spatial_dims = len(input_.shape[2:])
    paxes = np.roll(np.arange(2, spatial_dims + 2), 1)
    conv = input_

    # We convolve the separable basis
    for dshift, s, k, dstride, dpadding, ddilation \
        in zip(*basis_parameters, stride,
               padding, dilation):
        # Store the original shape of the input tensor.
        initial_shape = conv.shape

        # Compute the number of 1d strides
        strides = np.prod(initial_shape[2:-1], dtype=int)

        # Create the basis function and sample it.
        spline = BSplineElement.create_cardinal(dshift, s, k)

        # Sample the support of the spline
        lb, ub = spline.support_bounds().squeeze()
        x = np.arange(lb, ub) + 0.5
        spline_w = torch.Tensor(spline(x)).reshape(1, -1)
        spline_w = torch.stack([spline_w for i in range(iC)])
        width = spline_w.shape[-1]

        # Compute the theoretical ddconvolution
        # Perform the 1d convolution
        conv = conv1d(conv.reshape(*non_spatial_shape, -1),
                      spline_w,
                      stride=dstride,
                      padding=dpadding,
                      dilation=ddilation, groups=iC)

        # Add at the end extra values to have the right shape
        # to remove the excess of values due to tha fake ddim
        # 1d conv.
        conv = torch.cat([conv, torch.empty(*non_spatial_shape, width - 1)],
                         dim=-1)

        # Remove the excess from the 1d convolution.
        conv = conv.reshape(*non_spatial_shape, strides, -1)
        conv = conv[..., :-(width - 1)]
        conv = conv.reshape(*initial_shape[:-1], -1)

        # Permute to axes to have the one we are dealing with as last.
        conv = conv.permute(0, 1, *paxes)
    return conv


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
    spatial_dims = len(input_.shape[2:])
    k = np.array(k)

    if not isinstance(stride, Tuple):
        stride = ((stride,) * spatial_dims)

    if not isinstance(padding, Tuple):
        padding = ((padding,) * spatial_dims)

    if not isinstance(dilation, Tuple):
        dilation = ((dilation,) * spatial_dims)

    # For now we ignore the kernel size and we use the full
    # kernel size.
    kernel_size = CardinalBSplineKernel.centered_bounds_from_params(c, s, k)
    kernel_size = np.ceil(kernel_size[:, 1] - kernel_size[:, 0]).astype(int)
    kernel_size = (oC, iC, *kernel_size)

    # Output tensor shape.
    output_shape = np.array(conv_output_shape(
        input_.shape, kernel_size, stride=stride, padding=padding,
        dilation=dilation))

    # Dict of cached dd convolutions.
    # This is a light optimization that can be
    # useful in case of identical basis functions with identical
    # decimal centers.
    cached_convs = {}
    bases = []
    for cc, cs, ck in zip(c, s, k):
        dshifts, ishifts = tuple(zip(*(round_modf(v) for v in cc)))
        hashable_params = (tuple(dshifts), tuple(cs), tuple(ck))

        # For each center convolve w.r.t the input
        if hashable_params in cached_convs:
            conv = cached_convs[hashable_params]
        else:
            conv = _bsconv_subconv(input_, hashable_params,
                                   stride=stride, padding=padding,
                                   dilation=dilation)
            cached_convs[hashable_params] = conv

        cropl = (np.array(conv.shape) - output_shape) // 2
        cropr = conv.shape - cropl - output_shape
        crop = np.array((cropl, cropr)).T[2:].flatten()

        # Translate and crop the convolution to fit the output..
        shifts = ishifts * np.array(dilation)
        shifts = -shifts

        conv = cropped_translate(conv,
                                 shifts,
                                 crop,
                                 mode='constant', value=0)
        assert (conv.shape == output_shape).all()
        # At this point we have for each input channel the convolution
        # with a basis.
        bases.append(conv)

    # Stack the bases before doing the tp with the weights
    stacked_convs = torch.stack(bases, dim=2)
    group_iC = weights.shape[1]

    stacked_convs = stacked_convs.reshape(
        stacked_convs.shape[0],
        groups,
        group_iC,
        *stacked_convs.shape[2:]
    )

    output_channels = []
    for i, w in enumerate(weights):
        input_idx = (i % groups) * group_iC
        cv = stacked_convs[:, input_idx, ...]
        r = torch.tensordot(w, cv, dims=([0, 1], [1, 2]))
        output_channels.append(r)
    return torch.stack(output_channels, dim=1)


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
        pl = pl if pl else None
        pr = pr if pr else None
        slices.append(slice(pl, -pr))

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


def cropped_translate(x: torch.Tensor, shift: Tuple[int, ...],
                      crop_: List[Tuple[int, ...]],
                      mode='constant', value=0) -> torch.Tensor:
    """Apply a translation and crop the result."""
    crop_ = crop_ if not isinstance(crop, int) else [(crop_, crop_)] * len(shift)
    y = translate(x, shift, mode, value)
    return crop(y, crop_)
