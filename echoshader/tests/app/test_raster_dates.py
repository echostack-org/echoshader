import holoviews as hv
import numpy as np
import pandas as pd
import xarray as xr
from bokeh.models import DatetimeTicker, HoverTool

from echoshader.elements.echogram import _format_time_axis, single_echogram
from echoshader.elements.raster import _coordinates, rasterize_echogram, rasterize_rgb

hv.extension("bokeh")


def samples():
    x = np.arange(40, dtype=float) * 3000 + 1628793600000
    y = np.arange(160, dtype=float)[:, None] * 5 + np.sin(np.arange(40)[None, :] / 5) * 2
    z = np.broadcast_to(np.where(np.arange(160)[:, None] < 120, -70.0, 0.0), (160, 40)).copy()
    y[130:] = np.nan
    return hv.Points(
        pd.DataFrame(
            {"ping_time": np.broadcast_to(x, y.shape).ravel(), "depth": y.ravel(), "Sv": z.ravel()}
        ),
        kdims=["ping_time", "depth"],
        vdims=["Sv"],
    )


def test_scalar_nan_tail_does_not_spread_bottom_values():
    im = rasterize_echogram(samples(), width=80, height=160, dynamic=False)
    _, y, z = _coordinates(im)
    water = z[(y > 50) & (y < 500)]
    assert np.isfinite(water).any()
    np.testing.assert_allclose(water[np.isfinite(water)], -70)
    assert np.isnan(z[y > 700]).all()


def test_rgb_nan_tail_does_not_spread_bottom_colors():
    im = rasterize_rgb(
        [samples(), samples(), samples()], -90, -20, width=80, height=160, dynamic=False
    )
    y = im.dimension_values(1, expanded=False)
    alpha = im.dimension_values(5, flat=False)
    for i in range(2, 5):
        color = im.dimension_values(i, flat=False)
        water = (y > 50) & (y < 500)
        valid = alpha[water] > 0
        assert valid.any()
        np.testing.assert_allclose(color[water][valid], 20 / 70)
    assert (alpha[y > 700] == 0).all()


def test_rasterization_does_not_fill_coordinates_or_change_source_values():
    points = samples()
    original = points.data.copy(deep=True)
    rasterize_echogram(points, width=80, height=160, dynamic=False)
    pd.testing.assert_frame_equal(points.data, original)
    assert points.data.depth.isna().sum() == 30 * 40


def test_datetime_hover_and_ticks_after_raster_and_overlay():
    x = np.datetime64("2021-08-12T18:40:00", "ms") + np.arange(4) * np.timedelta64(3, "s")
    ds = xr.Dataset(
        {"Sv": (("channel", "ping_time", "depth"), np.ones((1, 4, 3)))},
        coords={"channel": ["a"], "ping_time": x, "depth": np.arange(3.0)},
    )
    p = single_echogram(ds, "a", vert_dim="depth", rasterized=True, dynamic=False).opts(
        hooks=[_format_time_axis], tools=["hover"]
    )
    overlay = p * hv.Curve(
        ([float(x[0].astype("int64")), float(x[-1].astype("int64"))], [1, 2]),
        kdims=["ping_time"],
        vdims=["depth"],
    ).opts(tools=["hover"])
    fig = hv.render(overlay)
    assert isinstance(fig.xaxis[0].ticker, DatetimeTicker)
    assert fig.xaxis[0].axis_label == "Time (UTC)"
    hovers = list(fig.select(type=HoverTool))
    assert hovers
    assert all(h.tooltips for h in hovers)


def test_fully_missing_ping_is_not_given_neighbor_coordinates():
    points = samples()
    missing_x = points.data.ping_time.unique()[20]
    frame = points.data.copy()
    frame.loc[frame.ping_time == missing_x, "depth"] = np.nan
    points = points.clone(frame)
    original = frame.copy(deep=True)
    image = rasterize_echogram(points, width=40, height=160, dynamic=False)
    _, _, values = _coordinates(image)
    assert np.isnan(values[:, 20]).all()
    pd.testing.assert_frame_equal(points.data, original)
    assert points.data.loc[points.data.ping_time == missing_x, "depth"].isna().all()


def test_rgb_datetime_hover():
    im = rasterize_rgb([samples(), samples(), samples()], -90, -20, dynamic=False)
    fig = hv.render(im.opts(hooks=[_format_time_axis], tools=["hover"]))
    assert isinstance(fig.xaxis[0].ticker, DatetimeTicker)
    assert fig.xaxis[0].axis_label == "Time (UTC)"
    hovers = list(fig.select(type=HoverTool))
    assert hovers
    assert all(h.tooltips for h in hovers)
