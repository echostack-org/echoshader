import asyncio

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import pytest
import xarray as xr

from echoshader.extensions.cps import app as cps
from echoshader.validation import resolve_rgb_channels, validate_dataset

hv.extension("bokeh")


def dataset():
    times = pd.date_range("2021-08-12", periods=4, freq="s")
    return xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "depth"), np.ones((4, 4, 3))),
            "Sv_water_column": (("channel", "ping_time", "depth"), np.ones((4, 4, 3))),
            "frequency_nominal": ("channel", [38000.0, 70000.0, 120000.0, 200000.0]),
            "longitude": ("ping_time", np.linspace(-122, -121.9, 4)),
            "latitude": ("ping_time", np.ones(4) * 36),
        },
        coords={
            "channel": ["38", "70", "120", "200"],
            "ping_time": times,
            "depth": [0.0, 10.0, 20.0],
        },
    )


def test_rgb_exact_and_missing_requests():
    ds = dataset()
    assert resolve_rgb_channels(ds, [120000, 38000, 70000]) == ("120", "38", "70")
    assert resolve_rgb_channels(ds) == ("120", "38", "70")
    with pytest.raises(ValueError, match="18 kHz"):
        resolve_rgb_channels(ds, [120000, 38000, 18000])
    assert validate_dataset(ds, tricolor_channels=["120", "38", "70"]).tricolor_ready
    assert not validate_dataset(ds, tricolor_channels=["120", "38", "38"]).tricolor_ready


@pytest.mark.parametrize(
    "freqs",
    [
        [120000, 38000, 38000],
        [120000, np.nan, 70000],
        [True, 38000, 70000],
        [[120000], [38000], [70000]],
        [120000, 38000],
    ],
)
def test_rgb_rejects_invalid_requests(freqs):
    with pytest.raises(ValueError):
        resolve_rgb_channels(dataset(), freqs)


def test_rgb_rejects_ambiguous_frequency():
    ds = dataset().assign(frequency_nominal=("channel", [38000.0, 70000.0, 120000.0, 120500.0]))
    with pytest.raises(ValueError, match="found 2"):
        resolve_rgb_channels(ds, [120000, 38000, 70000])


def make_view(tmp_path, monkeypatch, shared=False, requested=None):
    ds = dataset()
    times = ds.ping_time.values
    transects = pd.DataFrame(
        [
            {
                "transect_part": "001",
                "transect_number": "001",
                "start_time": pd.Timestamp(times[0], tz="UTC"),
                "end_time": pd.Timestamp(times[-1], tz="UTC"),
                "status": "completed",
            }
        ]
    )
    files = pd.DataFrame(
        [
            {
                "cps_filename": "a.zarr",
                "first_ping_time": pd.Timestamp(times[0], tz="UTC"),
                "last_ping_time": pd.Timestamp(times[-1], tz="UTC"),
                "status": "completed",
            }
        ]
    )
    monkeypatch.setattr(cps, "_catalogue", lambda *args, **kwargs: (transects, files))
    monkeypatch.setattr(cps, "_bottom_paths", lambda *args: [])
    ds.attrs.update(
        transect_start=transects.iloc[0].start_time.isoformat(),
        transect_end=transects.iloc[0].end_time.isoformat(),
    )
    path = tmp_path / "viz_cache_CPS/transects/transect_001.zarr"
    path.parent.mkdir(parents=True)
    ds.to_zarr(path, zarr_format=2, consolidated=True)
    callbacks = []

    def watch(revision, changed, **kwargs):
        def check():
            changed(revision())

        callbacks.append(check)
        return lambda: None

    monkeypatch.setattr(cps, "watch_changes", watch)
    calls = []

    def plot(data, **kwargs):
        calls.append(kwargs)
        return hv.Curve(([-122.0, -121.9], [0.0, 20.0]), kdims=["longitude"], vdims=["depth"])

    monkeypatch.setattr(cps, "tricolor_echogram", plot)
    monkeypatch.setattr(cps, "echogram", plot)
    monkeypatch.setattr(cps, "_transect_map", lambda *args: pn.Spacer())
    config = {"show_seafloor": False, "shared_xaxis": shared}
    if requested is not None:
        config["rgb_frequencies"] = requested
    source_path = tmp_path / "CPS_Sv" / "a.zarr"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(source_path, zarr_format=2, consolidated=True)
    _, tabs = cps.build_tabs(
        ds=xr.Dataset(), root=tmp_path, config=config, max_pings=100, max_range_samples=100
    )
    view = tabs[0][1](None, (-90, -20))
    # Render bindings just as the server does.
    asyncio.run(view[3].eval(view[3].object))
    return view, calls, callbacks


def test_configured_rgb_uses_nearest_available_frequency(tmp_path, monkeypatch):
    view, calls, _ = make_view(tmp_path, monkeypatch, requested=[120000, 38000, 18000])
    assert calls[-1]["channel"] == ["120", "38", "38"]

    view[1][5].value = "70"
    view[0][2].clicks += 1
    asyncio.run(view[3].eval(view[3].object))

    assert calls[-1]["channel"] == ["120", "38", "70"]


def test_apply_batches_changes_and_refresh_retains_committed_controls(tmp_path, monkeypatch):
    view, calls, callbacks = make_view(tmp_path, monkeypatch)
    controls = view[1]
    variable, _mode, _channel, red, _green, _blue, _bottom = controls
    apply = view[0][2]
    assert len(calls) == 1
    assert calls[-1]["channel"] == ["120", "38", "70"]
    variable.value = "Sv_water_column"
    red.value = "200"
    assert len(calls) == 1
    apply.clicks += 1
    asyncio.run(view[3].eval(view[3].object))
    assert len(calls) == 2
    assert calls[-1]["var_name"] == "Sv_water_column"
    assert calls[-1]["channel"] == ["200", "38", "70"]
    variable.value = "Sv"
    # Refresh even with an uncached render must use applied values.
    red.value = "120"
    callbacks[0]()
    assert all(c["var_name"] == "Sv_water_column" for c in calls[1:])


def test_cps_axes_are_linked_and_longitude_is_visible(tmp_path, monkeypatch):
    view, _, _ = make_view(tmp_path, monkeypatch)
    rendered = asyncio.run(view[3].eval(view[3].object))
    panes = rendered.select(pn.pane.HoloViews)

    assert panes
    assert all(p.linked_axes for p in panes)
    for pane in panes:
        options = pane.object.opts.get("plot").kwargs
        assert options["shared_axes"] is True
        assert options["xaxis"] == "bottom"
        assert options["xlim"][1] - options["xlim"][0] < 0.2


def test_single_file_rgb_defaults_choose_nearest_channels():
    from echoshader.app.single_viewer import AcousticSource, build_acoustic_app

    source = AcousticSource(dataset().drop_vars(["latitude", "longitude"]))
    app = build_acoustic_app(source)
    selectors = app.main[0].select(pn.widgets.Select)
    values = {widget.label: widget.value for widget in selectors}

    assert [values[f"{color} channel"] for color in ("Red", "Green", "Blue")] == ["120", "38", "38"]
