#!/usr/bin/env python3

"""EBConv setup module."""

from os import path

from setuptools import find_packages, setup

HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='ebconv',
    version='0.0.1',
    description='Library for Equivariant B-spline Convolutions.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/lromor/ebconv',
    author='Erik Bekkers, Leonardo Romor',
    # author_email='pypa-dev@googlegroups.com',
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by 'pip install'. See instead
        # 'python_requires' below.
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='convolution b-splines ai torch equivariant',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.5, <4',
    install_requires=['torch', 'numpy', 'scipy'],
    extras_require={
        'dev': ['check-manifest'],
        'test': ['pytest', 'pytest-cov'],
    },
    project_urls={
        'Bug Reports': 'https://github.com/lromor/ebconv/issues',
        'B-Spline paper': 'https://arxiv.org/pdf/1909.12057.pdf2',
        'Source': 'https://github.com/lromor/ebconv',
    },
)
