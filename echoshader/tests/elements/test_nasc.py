import holoviews
import numpy
import pytest
import xarray

from echoshader.elements.nasc import nasc


def test_nasc():
    ds = xarray.Dataset(
        {
            "NASC": (
                ("channel", "distance"),
                numpy.ones((2, 4)),
            ),
        },
        coords={
            "channel": ["38", "120"],
            "distance": [0, 1, 2, 3],
        },
    )

    plot = nasc(ds, channel="38")

    assert isinstance(plot, holoviews.Curve)


def test_nasc_requires_1d():
    ds = xarray.Dataset(
        {
            "NASC": (
                ("distance", "depth"),
                numpy.ones((4, 3)),
            ),
        },
        coords={
            "distance": [0, 1, 2, 3],
            "depth": [10, 20, 30],
        },
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        nasc(ds)
