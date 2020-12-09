"""Defines a set of wavelet transforms.

https://arxiv.org/pdf/1212.3530.pdf

See Equation 13.
"""
from ebconv.splines import BSplineElement
import numpy as np
import math


class PsiFourier2D:

    def __init__(self, no, n, t, k):
        """Construct a cake wavelet in the fourier domain.
        - no: angular resolution in radians.
        - n: taylor expansion order
        - t: scale parameter
        - k: b-spline order
        """
        self.no = no
        self.n = n
        self.t = t
        self.k = k

    def __call__(self, w1, w2):
        bspline = BSplineElement.create_cardinal(k=self.k)
        phi = np.arctan2(w2, w1)
        norm2 = w1 ** 2 + w2 ** 2
        c = norm2 / self.t

        mn = 0
        for i in range(self.n + 1):
            mn += c ** i / math.factorial(i)
        mn *= np.exp(-c)

        res = phi % (2 * np.pi) - np.pi / 2
        stheta = 2 * np.pi / self.no
        res = bspline(res / stheta)
        return res * mn


def ost2d(input_, n0, n, t, k, ratio=0.5):
    """2D Orientation score transform.

    Expects a batch of images ..., iH, iW
    spits out a batch of ost values
    ..., n0, iH, iW, 2.

    Ratio is a frequency cutoff.
    If ratio is 1, the cake wavelet has the same size as the input images,
    if 0 it's 0.
    """
    input_shape = input_.shape
    spatial_dims = np.array(input_shape[:-2])
    cake_dims = (spatial_dims * ratio).astype(int)

    # Construct the meshgrid

    # perform fft for each different rotation.

    # stack and return the results
    pass
