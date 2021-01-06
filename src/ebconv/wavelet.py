"""Defines a set of wavelet transforms.

https://arxiv.org/pdf/1212.3530.pdf

See Equation 13.
"""
import math
import numpy as np

import torch
from torch.fft import fftn, ifftn

from ebconv.splines import BSplineElement


def fourier_domain2d(region):
    """Constructs a centered region in the fourier domain.

    In the fourier domain the 0 frequency is placed as the first
    element of the matrix (top left). The meshgrid
    will represent a domain such as:
    ( 0, 0), (0,  1), ( 0, -1)
    ( 1, 0), (1,  1), ( 1, -1)
    (-1, 0), (1, -1), (-1,  1)
    """
    space = []
    for width in region:
        pos_half = width // 2 + 1
        neg_half = width - pos_half
        space.append(
            np.concatenate([np.arange(pos_half), np.arange(-neg_half, 0)]))
    return np.meshgrid(*space, indexing='ij')


class PsiFourier2D:
    """Cake wavelet representation in fourier domain."""

    def __init__(self, no, n, t, k):
        """Construct a cake wavelet in the fourier domain.
        - no: angular resolution in radians.
        - n: taylor expansion order
        - t: scale parameter
        - k: b-spline order
        """
        self.n_o = no

        # pylint: disable=invalid-name
        self.n = n

        # pylint: disable=invalid-name
        self.t = t
        self.k = k

    def __call__(self, f_x, f_y):
        bspline = BSplineElement.create_cardinal(k=self.k)
        phi = np.arctan2(f_y, f_x)
        norm2 = f_x ** 2 + f_y ** 2
        argument = norm2 / self.t
        stheta = 2 * np.pi / self.n_o

        m_n = 0
        for i in range(self.n + 1):
            m_n += argument ** i / math.factorial(i)
        m_n *= np.exp(-argument)

        samples = []
        for i in range(self.n_o):
            res = (phi - i * stheta) % (2 * np.pi) - np.pi / 2
            res = bspline(res / stheta)
            samples.append(res * m_n)
        return np.stack(samples)


# pylint: disable=too-few-public-methods
def ost2d(input_, no, n, t, k):
    """2D Orientation score transform.

    Expects a batch of images ..., iH, iW
    spits out a batch of ost values
    ..., n0, iH, iW, 2.

    The other parameters are described in PsiFourier2D
    """
    input_shape = input_.shape
    height, width = input_shape[-2:]
    input_ = input_.reshape(-1, 1, height, width)

    # Construct the meshgrid
    # pylint: disable=invalid-name
    xx, yy = fourier_domain2d((height, width))

    # Construct the cake function
    psi = PsiFourier2D(no, n, t, k)

    # pylint: disable=E1102
    samples = torch.tensor(psi(xx, yy)).to(input_.device)

    # Compute fft of input signal
    input_fft = fftn(input_, dim=(2, 3))

    # perform fft for each different rotation.
    # We do some magic permutation and reshaping
    # to allow broadcasting.
    samples = samples.reshape(1, *samples.shape)
    t = input_fft * samples

    # Perform inverse fft
    res = ifftn(t, dim=(2, 3))
    return res.reshape(*input_shape[:-2], no, height, width)
