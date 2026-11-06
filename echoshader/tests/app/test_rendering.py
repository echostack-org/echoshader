import datashader as dsh
import holoviews as hv
import numpy as np
import xarray as xr
from holoviews.operation.datashader import rasterize

from echoshader.elements.echogram import echogram, tricolor_echogram
from echoshader.elements.raster import rasterize_echogram

hv.extension("bokeh")


def source(curved=False, descending=False, missing=False):
    time = np.array(
        [
            "2024-01-01T00:00:00",
            "2024-01-01T00:00:01",
            "2024-01-01T00:00:03",
            "2024-01-01T00:00:04",
        ],
        dtype="datetime64[ns]",
    )
    y = np.tile(np.arange(1, 7, dtype=float)[None, :], (4, 1))
    if curved:
        y += np.arange(4)[:, None] * 0.01
    values = np.full((3, 4, 6), -60.0)
    if missing:
        y[:, -1] = np.nan
        values[:, :, -1] = np.nan
    return xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"), values),
            "echo_range": (
                ("channel", "ping_time", "range_sample"),
                np.tile(y[None, :, :], (3, 1, 1)),
            ),
            "longitude": (
                "ping_time",
                [-124.1, -124.2, -124.3, -124.4]
                if descending
                else [-124.4, -124.3, -124.2, -124.1],
            ),
        },
        coords={"channel": ["a", "b", "c"], "ping_time": time, "range_sample": np.arange(6)},
    )


def test_linear_mean_matches_explicit_reference():
    values = np.array([[-60.0, -80.0], [-60.0, -80.0]])
    mesh = hv.QuadMesh(([0.0, 1.0], [0.0, 1.0], values), vdims=["Sv"])
    actual = rasterize_echogram(mesh, width=1, height=1, dynamic=False)
    expected = rasterize(
        mesh.clone(([0.0, 1.0], [0.0, 1.0], 10 ** (values / 10))),
        width=1,
        height=1,
        aggregator=dsh.mean("Sv"),
        dynamic=False,
    )
    np.testing.assert_allclose(
        actual.dimension_values(2), 10 * np.log10(expected.dimension_values(2))
    )


def test_datetime_curved_mesh_and_rgb_render():
    ds = source(curved=True)
    for plot in [
        echogram(ds, "a", rasterized=True),
        tricolor_echogram(ds, ["a", "b", "c"], -100, -30, rasterized=True),
    ]:
        hv.render(plot)
        image = plot[()]
        if type(image) is hv.Image:
            np.testing.assert_allclose(
                image.dimension_values(2)[np.isfinite(image.dimension_values(2))], -60.0, atol=1e-8
            )


def test_missing_tail_and_descending_longitude():
    ds = source(curved=True, descending=True, missing=True)
    plot = tricolor_echogram(ds, ["a", "b", "c"], -100, -30, x_dim="longitude", rasterized=True)
    hv.render(plot)
    im = plot[()]
    assert isinstance(im, hv.RGB)
    assert im.range("longitude")[0] < im.range("longitude")[1]
    alpha = im.dimension_values("A")
    assert np.any(alpha > 0)


def test_mismatched_rgb_geometry_rejected():
    ds = source(curved=True)
    ds.echo_range.values[1] += 1
    import pytest

    with pytest.raises(ValueError, match="differs"):
        tricolor_echogram(ds, ["a", "b", "c"], -100, -30, rasterized=True)
