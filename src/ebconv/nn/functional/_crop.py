"""Impement the cropping functional for pytorch."""

from typing import Tuple

import torch


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
