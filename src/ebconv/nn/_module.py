"""Implements a torch module for the bspline convolution."""

import math
from functools import reduce

from typing import Tuple, Union

import warnings

import torch
from torch.nn import init
from torch.nn.parameter import Parameter

from ebconv.nn.functional import cbsconv
from ebconv.kernel import create_random_centers
from ebconv.kernel import create_uniform_grid


LAYOUTS = Union['grid', 'random']


class CBSConv(torch.nn.Module):
    """Torch module for learning cardinal bspline kernels."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, ...],
                 layout: LAYOUTS = 'random',
                 nc: Union[Tuple[int, ...], int, None] = None,
                 k: int = 2,
                 stride: Union[Tuple[int, ...], int] = 1,
                 padding: Union[Tuple[int, ...], int] = 0,
                 dilation: Union[Tuple[int, ...], int] = 1,
                 adaptive_centers: bool = True,
                 adaptive_scalings: bool = True,
                 basis_groups: int = 1, bias: bool = False,
                 padding_mode: str = 'zeros',
                 separable: bool = False):
        """Pytorch module that implements bspline convolutions.

        if layout grid is passed, nc can be set as a tuple
        """
        super().__init__()
        if not isinstance(kernel_size, Tuple):
            raise ValueError('kernel_size should be a Tuple')

        if k < 0:
            raise ValueError('k must be >= 0')

        dims = len(kernel_size)

        if isinstance(stride, int):
            stride = ((stride,) * dims)

        if isinstance(padding, int):
            padding = ((padding,) * dims)

        if isinstance(dilation, int):
            dilation = ((dilation,) * dims)

        if out_channels % basis_groups != 0:
            raise ValueError('out_channels must be divisible by basis_groups')

        if layout not in ('grid', 'random'):
            raise ValueError('layout must be either "grid" or "random"')

        if layout == 'random':
            if not isinstance(nc, int):
                raise TypeError('Layout "random" requires the '
                                'number of centers to be specified '
                                'as an integer.')

        grid_size = kernel_size
        if layout == 'grid':
            grid_size = kernel_size if nc is None else nc
            if not isinstance(grid_size, tuple):
                raise TypeError('If layout is grid, nc should be either '
                                'None or a tuple.')
            if len(grid_size) != len(kernel_size):
                raise ValueError('nc tuple and kernel_size should '
                                 'have the same length')
            nc = reduce(lambda x, y: x * y, grid_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dims = dims
        self.k = k
        self.nc = nc
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.adaptive_centers = adaptive_centers
        self.adaptive_scalings = adaptive_scalings
        self.layout = layout
        self.grid_size = grid_size
        self.basis_groups = basis_groups
        self.padding_mode = padding_mode
        self.separable = separable

        self.weights = Parameter(
            torch.Tensor(out_channels, in_channels, nc))
        self.centers = Parameter(torch.Tensor(basis_groups, nc, dims))
        if not adaptive_centers:
            self.centers.requires_grad = False
        self.scalings = Parameter(torch.Tensor(basis_groups, nc))
        if not adaptive_scalings:
            self.scalings.requires_grad = False

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        with torch.no_grad():
            self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters using default initializers."""
        if self.layout == 'grid':
            self.centers.data = torch.stack([
                torch.from_numpy(
                    create_uniform_grid(self.grid_size)).float()
                for _ in range(self.basis_groups)])
        else:
            self.centers.data = torch.stack([
                torch.from_numpy(
                    create_random_centers(self.kernel_size, self.nc)).float()
                for _ in range(self.basis_groups)])

        self.scalings.data = torch.ones_like(self.scalings)
        volume = reduce(lambda x, y: x * y, self.grid_size)
        self.scalings.data *= (volume / self.nc) ** (1 / self.dims)

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
            self.dilation, separable=self.separable)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', nc={nc}, adaptive_centers={adaptive_centers}'
             ', adaptive_scalings={adaptive_scalings}'
             ', layout={layout}, stride={stride}')
        if self.k != 2:
            s += ', k={k}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.basis_groups != 1:
            s += ', basis_groups={basis_groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class CBSConv2d(CBSConv):
    """CBSConv specialization for a 2d input."""

    def __init__(self, i_c, o_c, kernel_size, *args, **kwargs):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        super().__init__(i_c, o_c, kernel_size, *args, **kwargs)
