"""Tests of the ebconv.utils module."""

from ebconv.utils import round_modf

import numpy as np

import pytest


@pytest.mark.parametrize('x,expected', [
    (0, (0.0, 0)),
    (5.000, (0.0, 5)),
    (6.0001, (0.0001, 6)),
    (7.9999, (-0.0001, 8)),
    (8.5, (0.5, 8))
])
def test_round_modf(x, expected):
    """Test round modf."""
    df, di = round_modf(x)
    r_df, r_di = expected
    assert isinstance(di, int)
    assert df + di == x
    assert np.isclose(df, r_df)
    assert di == r_di
