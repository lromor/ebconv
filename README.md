<!-- README.md -->

[1]: https://arxiv.org/abs/1909.12057
[2]: https://docs.nvidia.com/deeplearning/sdk/pdf/cuDNN-API.pdf
[3]: https://travis-ci.com/lromor/ebconv.svg?token=qj78wiWyraZW4FD7pwLr&branch=master


# EBC✸NV [![Build Status][3]](https://travis-ci.com/lromor/ebconv)

Equivariant B-splines CONVolutions.

## Installation

This package has not been yet published on PyPI.
It's possible to install it directly from github using:

``` sh
pip install git+https://github.com/lromor/ebconv
```


## Development

To install a development version of the package
clone the repository with:

``` sh
git clone https://github.com/lromor/ebconv.git && cd ebconv
```

After cloning the repository and entering its directory, it's possible to
conveniently add to your path an editable version of the package using:

``` sh
pip install -e ".[dev, test]"
```

# Tests

You can run the full test suite by running `tox` or simply run:

``` sh
pytest -sv
```

# Related work

* [B-Spline CNNs on Lie Groups][1]
