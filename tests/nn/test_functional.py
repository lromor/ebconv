"""Test the functional module."""
import numpy as np

import pytest

import torch

from ebconv.kernel import CardinalBSplineKernel
from ebconv.kernel import create_random_centers
from ebconv.nn.functional import cbsconv
from ebconv.nn.functional import UnivariateCardinalBSpline
from ebconv.nn.functional import convdd_separable
from ebconv.nn.functional import crop
from ebconv.kernel import sampling_domain


def test_crop_simple():
    """Check that cropping along the two axes gives the expected result."""
    tt_input = torch.Tensor((
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 2.0, 3.0, 0.0),
        (0.0, 4.0, 5.0, 6.0, 0.0),
        (0.0, 7.0, 8.0, 9.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ))[None, None, :]

    assert crop(tt_input, (1, 2, 2, 1)) \
        .equal(torch.Tensor((
            (2.0, 3.0),
            (5.0, 6.0),
        ))[None, None, :])


D2F = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d
}

D2I = {
    0: 'a',
    1: 'b',
    2: 'c'
}


@pytest.mark.parametrize('dilation', [1, 2])
@pytest.mark.parametrize('padding', [0, 1])
@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('w_size', [1, 2])
@pytest.mark.parametrize('i_c, o_c, groups', [
    (6, 4, 2),
    (3, 3, 1),
    (3, 3, 3),
])
def test_convdd_separable(i_c, o_c, groups, w_size, dim, stride,
                          padding, dilation):
    """Test consistent values with torch.

    Input size is fixed to 8, 16, 32 with 3 batches.
    """
    batch = 3
    isize = np.power(2, np.arange(3, 3 + dim))
    input_ = torch.rand(batch, i_c, *isize)

    weight = []
    for i in range(dim):
        weight.append(torch.rand(o_c, i_c // groups, w_size + i))

    # Compute tensordot using einsum.
    einsum_eq = ['ij' + D2I[d] for d in range(dim)]
    einsum_eq = ','.join(einsum_eq)
    einsum_eq += '->'
    einsum_eq += 'ij' + ''.join([D2I[d] for d in range(dim)])

    torch_weight = torch.einsum(einsum_eq, *weight)

    # Function to test against
    tconv = D2F[dim]
    torch_output = tconv(
        input_, torch_weight, stride=stride, padding=padding,
        dilation=dilation, groups=groups)

    output = convdd_separable(
        input_, weight, stride=stride, padding=padding, dilation=dilation,
        groups=groups)
    assert torch.allclose(torch_output, output)


@pytest.mark.parametrize('k', [2, 3])
@pytest.mark.parametrize('s', [0.1, 1, 2])
@pytest.mark.parametrize('c', [0, 0.3, -5])
def test_autograd_univariate_cardinalbspline(c, s, k):
    """Test that the autograd for cardinalbspline.

    Notice the for k = 0,1 the splines are not differentiable.
    """
    input_ = torch.arange(-10, 10, dtype=torch.double, requires_grad=False)
    # pylint: disable=E1102
    c = torch.tensor(c, dtype=torch.double, requires_grad=True).reshape(1)
    # pylint: disable=E1102
    s = torch.tensor(s, dtype=torch.double, requires_grad=False)

    params = (input_, c, s, k)
    torch.autograd.gradcheck(UnivariateCardinalBSpline.apply,
                             params, eps=1e-6, atol=1e-6)


@pytest.mark.parametrize('dilation', [1, 2])
@pytest.mark.parametrize('padding', [0, 1])
@pytest.mark.parametrize('stride', [1, 2])
@pytest.mark.parametrize('k', [2])
@pytest.mark.parametrize('dim', [1, 2, 3])
@pytest.mark.parametrize('i_c,o_c,groups', [
    (6, 4, 2),
    (3, 3, 1),
    (3, 3, 3),
])
def test_cbsconv(i_c, o_c, groups, dim, k, stride, padding, dilation):
    """Test the cbsconv torch functional."""
    # Extra params
    batch = 3
    n_c = 10
    input_min_size = 20
    kernel_min_size = 7
    input_spatial_shape = tuple(input_min_size + i * 3 for i in range(dim))
    kernel_size = tuple(kernel_min_size + i for i in range(dim))

    # Generate a random set of centers/scalings per group.
    group_input_channels = i_c // groups
    group_output_channels = o_c // groups
    group_total = group_input_channels * group_output_channels

    centers = []
    scalings = []
    weights = []
    virtual_weights = []
    for _ in range(groups):
        g_centers = create_random_centers(
            kernel_size, n_c, integrally_spaced=True)
        g_scalings = torch.rand(n_c, dim) * 3 + 0.5
        kernel = CardinalBSplineKernel.create(
            c=g_centers.numpy(), s=g_scalings.numpy(), k=k)

        sampling = np.meshgrid(*[sampling_domain(k_s) for k_s in kernel_size],
                               indexing='ij')
        bases = kernel(*sampling)
        g_weights = np.random.rand(group_total, n_c)
        v_w = np.concatenate(
            [np.tensordot(weight, bases, axes=1)
             for weight in g_weights]
        ).reshape(group_output_channels, group_input_channels, *kernel_size)
        centers.append(g_centers.reshape(1, n_c, dim))
        scalings.append(g_scalings.reshape(1, n_c, dim))

        weights.append(
            torch.from_numpy(g_weights).reshape(
                group_output_channels, group_input_channels, n_c))

        virtual_weights.append(torch.from_numpy(v_w))

    centers = torch.cat(centers).float()
    centers.requires_grad = True
    scalings = torch.cat(scalings).float()
    scalings.requires_grad = True
    weights = torch.cat(weights).float()
    weights.requires_grad = True
    virtual_weights = torch.cat(virtual_weights).float()
    virtual_weights.requires_grad = True

    bias = torch.rand(o_c)

    # Create the input
    input_ = torch.rand(batch, i_c, *input_spatial_shape)

    # Function to test against
    tconv = D2F[dim]
    torch_output = tconv(
        input_, virtual_weights, bias=bias, stride=stride, padding=padding,
        dilation=dilation, groups=groups)

    output = cbsconv(input_, kernel_size, weights, centers, scalings, k,
                     bias=bias, stride=stride, padding=padding,
                     dilation=dilation, groups=groups)
    assert torch.allclose(torch_output, output)
