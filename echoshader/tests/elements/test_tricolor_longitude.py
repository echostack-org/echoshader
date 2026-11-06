import holoviews as hv
import numpy as np
import pytest
import xarray as xr

from echoshader.elements.echogram import tricolor_echogram

hv.extension("bokeh")


def data(longitude):
    values = np.full((3, len(longitude), 3), -60.0)
    values[:, 1, :] = -30.0
    return xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "depth"), values),
            "longitude": ("ping_time", longitude),
        },
        coords={
            "channel": ["18", "38", "70"],
            "depth": [5.0, 15.0, 25.0],
            "ping_time": np.datetime64("2024-08-05T13:20:00", "ns")
            + np.arange(len(longitude)) * np.timedelta64(1, "s"),
        },
    )


@pytest.mark.parametrize("longitude", [[0.0, 2.0, 1.0, 3.0], [3.0, 2.0, 1.0, 0.0]])
def test_longitude_uses_actual_cells_without_sorting(longitude):
    ds = data(longitude)
    before = ds.copy(deep=True)
    mesh = tricolor_echogram(ds, ["18", "38", "70"], -90, -30, x_dim="longitude", vert_dim="depth")
    assert isinstance(mesh, hv.QuadMesh)
    np.testing.assert_array_equal(mesh.data["longitude"], longitude)
    plot = hv.renderer("bokeh").get_plot(mesh)
    source = plot.handles["source"].data
    indices = np.asarray(source["rgb_index"], dtype=int)
    np.testing.assert_array_equal(indices, np.arange(12).reshape(3, 4).T.ravel())
    expected = np.full((3, 4), "#808080", dtype="<U7")
    expected[:, 1] = "#ffffff"
    np.testing.assert_array_equal(source["rgb_color"], expected.ravel()[indices])
    assert plot.handles["glyph_renderer"].glyph.fill_color.field == "rgb_color"
    xr.testing.assert_identical(ds, before)


def test_time_axis_keeps_existing_rgb_element():
    ds = data([0.0, 1.0, 2.0, 3.0])
    assert isinstance(tricolor_echogram(ds, ["18", "38", "70"], -90, -30, vert_dim="depth"), hv.RGB)
