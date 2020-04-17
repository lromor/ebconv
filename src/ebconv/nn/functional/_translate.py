"""Implement a torch translation."""

from typing import Tuple

import torch


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
