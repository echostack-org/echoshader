import holoviews
import numpy
import pytest
import xarray

from echoshader.elements.echogram import (
    echogram,
    tricolor_echogram,
)


@pytest.fixture
def ds():
    return xarray.Dataset(
        {
            "Sv": (
                ("channel", "ping_time", "depth"),
                numpy.ones((3, 3, 4)),
            ),
        },
        coords={
            "channel": ["18", "38", "120"],
            "ping_time": numpy.array(
                [0, 1, 2],
            ),
            "depth": [10, 20, 30, 40],
        },
    )


def test_echogram(ds):
    plot = echogram(
        ds,
        channel="38",
        vert_dim="depth",
    )

    assert isinstance(plot, holoviews.Image)


def test_multichannel_echogram(ds):
    plot = echogram(
        ds,
        channel=["18", "38"],
        vert_dim="depth",
    )

    assert isinstance(plot, holoviews.Layout)


def test_tricolor(ds):
    plot = tricolor_echogram(
        ds,
        channel=["120", "38", "18"],
        vmin=-80,
        vmax=-30,
        vert_dim="depth",
    )

    assert isinstance(plot, holoviews.RGB)


def test_irregular_ping_time(ds):
    ds = ds.assign_coords(
        ping_time=[0, 1, 3],
    )

    plot = echogram(
        ds,
        channel="38",
        vert_dim="depth",
    )

    assert isinstance(plot, holoviews.QuadMesh)


def test_invalid_dataset():
    ds = xarray.Dataset(
        {
            "abc": (
                ("channel", "ping_time", "depth"),
                numpy.ones((1, 3, 4)),
            ),
        },
        coords={
            "channel": ["38"],
            "ping_time": [0, 1, 2],
            "depth": [10, 20, 30, 40],
        },
    )

    with pytest.raises(
        ValueError,
        match="Dataset must contain a variable named 'Sv'",
    ):
        _ = ds.eshader
