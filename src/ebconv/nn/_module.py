"""Implements a torch module for the bspline convolution."""

import math

from typing import Tuple

import torch
from torch.nn import init
from torch.nn.parameter import Parameter

from ebconv.nn.functional import cbsconv
from ebconv.kernel import create_random_centers


class CBSConv(torch.nn.Module):
    """Torch module for learning cardinal bspline kernels."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, ...], nc: int, k: int,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dims = dims = len(kernel_size)
        self.k = k
        self.nc = nc
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weights = Parameter(
            torch.Tensor(out_channels, in_channels // groups, nc))
        self.centers = Parameter(torch.Tensor(groups, nc, dims))
        self.scalings = Parameter(torch.Tensor(groups, nc, dims))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters using default initializers."""
        with torch.no_grad():
            self.centers = torch.stack([
                torch.from_numpy(
                    create_random_centers(self.kernel_size, self.nc))
                for _ in self.groups])

            scalings = torch.Tensor(
                [size / self.nc for size in self.kernel_size]
            ).reshape(1, 1, self.dims)
            self.scalings = torch.ones(
                self.groups, self.nc, self.dims) * scalings

        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            # pylint: disable=protected-access
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    # pylint: disable=arguments-differ
    def forward(self, input_):
        """Implement the forward pass of the module."""
        return cbsconv(
            input_, self.kernel_size, self.weights, self.centers,
            self.scalings, self.k, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
