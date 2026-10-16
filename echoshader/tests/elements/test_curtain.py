import numpy
import plotly.graph_objects
import pyvista
import xarray

from echoshader.elements.curtain import curtain


def test_curtain_plotly():
    ds = xarray.Dataset(
        {
            "Sv": (
                ("channel", "ping_time", "depth"),
                numpy.ones((1, 4, 3)),
            ),
        },
        coords={
            "channel": ["38"],
            "ping_time": [0, 1, 2, 3],
            "depth": [10, 20, 30],
            "longitude": (
                "ping_time",
                [-125.0, -124.9, -124.8, -124.7],
            ),
            "latitude": (
                "ping_time",
                [48.0, 48.1, 48.2, 48.3],
            ),
        },
    )

    plot = curtain(
        ds,
        channel="38",
        engine="plotly",
    )

    assert isinstance(
        plot,
        plotly.graph_objects.Figure,
    )


def test_curtain_pyvista():
    ds = xarray.Dataset(
        {
            "Sv": (
                ("channel", "ping_time", "depth"),
                numpy.ones((1, 4, 3)),
            ),
        },
        coords={
            "channel": ["38"],
            "ping_time": [0, 1, 2, 3],
            "depth": [10, 20, 30],
            "longitude": (
                "ping_time",
                [-125.0, -124.9, -124.8, -124.7],
            ),
            "latitude": (
                "ping_time",
                [48.0, 48.1, 48.2, 48.3],
            ),
        },
    )

    plot = curtain(
        ds,
        channel="38",
        engine="pyvista",
    )

    assert isinstance(
        plot,
        pyvista.Plotter,
    )
