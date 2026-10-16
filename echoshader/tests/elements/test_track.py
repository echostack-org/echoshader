import geoviews
import holoviews
import xarray

from echoshader.elements.track import track


def test_track():
    ds = xarray.Dataset(
        coords={
            "ping_time": [0, 1, 2],
            "longitude": (
                "ping_time",
                [-125.0, -124.9, -124.8],
            ),
            "latitude": (
                "ping_time",
                [48.0, 48.1, 48.2],
            ),
        },
    )

    plot = track(
        ds,
        tile=None,
    )

    assert isinstance(
        plot,
        holoviews.core.overlay.Overlay,
    )


def test_moored_track():
    ds = xarray.Dataset(
        coords={
            "ping_time": [0, 1, 2],
            "longitude": (
                "ping_time",
                [-125.0, -125.0, -125.0],
            ),
            "latitude": (
                "ping_time",
                [48.0, 48.0, 48.0],
            ),
        },
    )

    plot = track(
        ds,
        tile=None,
    )

    assert isinstance(plot, geoviews.Points)
