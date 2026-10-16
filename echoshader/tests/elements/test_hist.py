import holoviews
import numpy
import xarray

from echoshader.elements.hist import hist


def test_hist():
    ds = xarray.Dataset(
        {
            "Sv": (
                ("channel", "ping_time", "depth"),
                numpy.ones((2, 3, 4)),
            ),
        },
        coords={
            "channel": ["18", "38"],
            "ping_time": [0, 1, 2],
            "depth": [10, 20, 30, 40],
        },
    )

    plot = hist(
        ds,
        bins=10,
        overlay=True,
    )

    assert isinstance(
        plot,
        holoviews.core.overlay.NdOverlay,
    )


def test_hist_separate():
    ds = xarray.Dataset(
        {
            "Sv": (
                ("channel", "ping_time", "depth"),
                numpy.ones((2, 3, 4)),
            ),
        },
        coords={
            "channel": ["18", "38"],
            "ping_time": [0, 1, 2],
            "depth": [10, 20, 30, 40],
        },
    )

    plot = hist(
        ds,
        bins=10,
        overlay=False,
    )

    assert isinstance(
        plot,
        holoviews.NdLayout,
    )
