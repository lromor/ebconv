"""Definition of main workhorse functions made in pytorch."""

from typing import List, Tuple, Union

import torch


def _bconv_impl(x: torch.Tensor, cp: torch.Tensor, cx: torch.Tensor,
                s: float, n: int) -> torch.Tensor:
    """Compute a bspline separable convolution."""
    pass


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


def cropped_translate(x: torch.Tensor, shift: torch.Tensor,
                      crop: Union[torch.Tensor, int],
                      mode='constant', value=0) -> torch.Tensor:
    """Apply a translation and crop the result."""
    crop = crop if not isinstance(crop, int) else [(crop. crop)] * len(shift)
    y = translate(x, shift, mode, value)
    return crop(y, crop)
