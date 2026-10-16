import holoviews
import numpy
import xarray

from echoshader.elements.table import table


def test_table():
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

    plot = table(ds)

    assert isinstance(
        plot,
        holoviews.Table,
    )
