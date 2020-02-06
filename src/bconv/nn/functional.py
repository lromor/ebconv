"""Definition of main workhorse functions made in pytorch."""

from typing import List, Tuple, Union

import torch


def _bconv_impl(x: torch.Tensor, cp: torch.Tensor, cx: torch.Tensor,
                s: float, n: int) -> torch.Tensor:
    """Compute a bspline separable convolution."""
    pass


def crop(x: torch.Tensor, crop: List[Tuple[int, ...]]):
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
    for l, r in crop:
        l = l if l else None
        r = r if r else None
        slices.append(slice(l, -r))

    slices = tuple(reversed(slices))

    # Apply the crop and return
    return x[slices]


def translate(x: torch.Tensor, shift: Tuple[int, ...],
              mode='constant', value=0):
    """Translate the input.

    Args:
        x: Input tensor
        shift: Represents the shift values for each spatial axis.
        mode: Same argument as torch.nn.functional.pad
        value: Same as above.

    Returns:
        Translated tensor.

    """
    assert len(shift) == len(x.shape) - 2

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
            slices.insert(1,slice(shift, None))

    slices = tuple(slices)
    y = torch.nn.functional.pad(x, paddings, mode=mode, value=value)
    # Apply the crop and return
    return y[slices]


def _cropped_translate_impl(x: torch.Tensor, shift: Tuple[int, ...],
                            crop: List[Tuple[int, int]], mode, value):
    y = translate(x, shift, mode, value)
    return crop(y, crop)


def cropped_translate(x: torch.Tensor, shift: torch.Tensor,
                      crop: Union[torch.Tensor, int],
                      mode='constant', value=0):
    """Apply a translation and crop the result."""
    crop = crop if not isinstance(crop, int) else [(crop. crop)] * len(shift)
    return _cropped_translate_impl(x, shift, crop, mode, value)
