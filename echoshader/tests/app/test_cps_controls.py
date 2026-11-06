"""CPS controls reuse elements, bound reads and expose clipped transects."""

import sqlite3

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import pytest
import xarray as xr

from echoshader.extensions.cps import app as cps

hv.extension("bokeh")


@pytest.fixture
def survey(tmp_path):
    with sqlite3.connect(tmp_path / "processing.db") as con:
        con.execute(
            "CREATE TABLE transects (transect_part TEXT, transect_number TEXT, start_time TEXT, end_time TEXT, status TEXT)"
        )
        con.execute(
            "CREATE TABLE sv_cps (cps_filename TEXT, first_ping_time TEXT, last_ping_time TEXT, status TEXT)"
        )
        con.execute(
            "INSERT INTO transects VALUES ('001','001','2024-01-01T00:00:00Z','2024-01-01T00:02:00Z','completed')"
        )
        con.execute(
            "INSERT INTO sv_cps VALUES ('a.zarr','2024-01-01T00:00:00Z','2024-01-01T00:02:00Z','completed')"
        )
    data = xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"), np.full((4, 3, 4), -50.0)),
            "Sv_water_column": (
                ("channel", "ping_time", "range_sample"),
                np.full((4, 3, 4), -60.0),
            ),
            "depth": ("range_sample", [1.0, 2.0, 3.0, 4.0]),
            "longitude": ("ping_time", [-125.2, -125.15, -125.1]),
            "latitude": ("ping_time", [40.0, 40.0, 40.0]),
            "frequency_nominal": ("channel", [18000.0, 38000.0, 70000.0, 120000.0]),
        },
        coords={
            "channel": ["18", "38", "70", "120"],
            "ping_time": pd.date_range("2024-01-01", periods=3, freq="min"),
            "range_sample": range(4),
        },
    )
    (tmp_path / "CPS_Sv").mkdir()
    data.to_zarr(tmp_path / "CPS_Sv/a.zarr", zarr_format=2)
    return tmp_path


def view(root, monkeypatch, **config):
    monkeypatch.setattr(cps, "_transect_map", lambda *a, **k: pn.Spacer())
    _, tabs = cps.build_tabs(
        ds=None,
        root=root,
        config={"refresh_seconds": 0, **config},
        max_pings=100,
        max_range_samples=20,
    )
    return tabs[0][1](None, (-100, -30))


def test_selectors_route_variable_and_channels_to_existing_elements(survey, monkeypatch):
    calls = []

    def record(kind):
        def plot(data, **kwargs):
            calls.append((kind, kwargs, list(data.channel.values)))
            return hv.Image(np.ones((2, 2))) if kind == "single" else hv.Curve([1, 2])

        return plot

    monkeypatch.setattr(cps, "tricolor_echogram", record("RGB"))
    monkeypatch.setattr(cps, "echogram", record("single"))
    app = view(survey, monkeypatch)
    variable, mode, channel, red, green, blue, show_bottom = app[1]
    assert variable.options == ["Sv", "Sv_water_column"]
    assert calls[-1][1]["channel"] == ["18", "38", "70"]
    assert any("120 kHz" in label for label in red.options)
    red.value = "120"
    assert calls[-1][1]["channel"] == ["120", "38", "70"]
    variable.value = "Sv_water_column"
    assert calls[-1][1]["var_name"] == "Sv_water_column"
    mode.value = "Single channel"
    channel.value = "120"
    assert calls[-1][0] == "single"
    assert calls[-1][1]["channel"] == "120"
    assert calls[-1][2] == ["120"]
    count = len(calls)
    app[0][1].clicks += 1
    assert len(calls) == count


def test_clipped_001_shows_warning_instead_of_empty_plot(survey, monkeypatch):
    app = view(survey, monkeypatch, xlim=[-125.0, -124.25])
    assert any("outside xlim" in str(alert.object) for alert in app.select(pn.pane.Alert))


def test_single_element_renders_001_with_automatic_bounds(survey, monkeypatch):
    app = view(survey, monkeypatch, mode="Single channel")
    panes = app.select(pn.pane.HoloViews)
    assert panes
    for pane in panes:
        hv.render(pane.object)
    assert not any("outside xlim" in str(alert.object) for alert in app.select(pn.pane.Alert))
    app[1][1].value = "Single channel"
    for pane in app.select(pn.pane.HoloViews):
        hv.render(pane.object)


def test_empty_startup_waits_for_metadata(tmp_path, monkeypatch):
    app = view(tmp_path, monkeypatch)
    assert any("Waiting for acoustic" in str(alert.object) for alert in app.select(pn.pane.Alert))


def test_bottom_matches_pings_without_filling_missing_detections(survey):
    directory = survey / "CPS_Seafloor_CSVs"
    directory.mkdir()
    path = directory / "a.zarr_bottom_line.csv"
    pd.DataFrame(
        {
            "time": ["2024-01-01T00:00:00Z", "2024-01-01T00:02:00Z"],
            "depth": [50.0, 70.0],
        }
    ).to_csv(path, index=False)
    with xr.open_zarr(survey / "CPS_Sv/a.zarr") as data:
        result = cps._bottom_samples([path], data)
        np.testing.assert_allclose(result, [50.0, np.nan, 70.0], equal_nan=True)
        assert np.isnan(cps._bottom_samples([directory / "missing.csv"], data)).all()
    assert cps._bottom_paths(directory, ["Hake_example_CPS.zarr"]) == (
        directory / "Hake_example_bottom_line.csv",
    )


def test_seafloor_overlay_toggle_and_new_csv_reuses_acoustic_preview(survey, monkeypatch):
    calls = []
    original_open = xr.open_zarr

    def opened(*args, **kwargs):
        calls.append(args[0])
        return original_open(*args, **kwargs)

    monkeypatch.setattr(xr, "open_zarr", opened)
    monkeypatch.setattr(cps, "tricolor_echogram", lambda *a, **k: hv.Image(np.ones((2, 2))))
    app = view(survey, monkeypatch)
    count = len(calls)
    directory = survey / "CPS_Seafloor_CSVs"
    directory.mkdir()
    path = directory / "a.zarr_bottom_line.csv"
    pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=3, freq="min"),
            "depth": [50.0, 60.0, 70.0],
        }
    ).to_csv(path, index=False)
    app[0][1].clicks += 1
    assert len(calls) == count  # Only the line changed: bounded acoustic preview stays cached.
    plot = app.select(pn.pane.HoloViews)[0].object
    assert isinstance(plot, hv.Overlay)
    line = plot.get(1)
    np.testing.assert_allclose(line.dimension_values("depth"), [50.0, 60.0, 70.0])
    hv.render(plot)
    app[1][-1].value = False
    assert not isinstance(app.select(pn.pane.HoloViews)[0].object, hv.Overlay)
    assert len(calls) == count


def test_single_file_catalogue_includes_bottom_and_its_revision(survey):
    directory = survey / "CPS_Seafloor_CSVs"
    directory.mkdir()
    path = directory / "a.zarr_bottom_line.csv"
    pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=3, freq="min"),
            "depth": [50.0, 60.0, 70.0],
        }
    ).to_csv(path, index=False)
    entries = cps.build_files(root=survey)
    for label in ("Latest completed transect", "CPS · a.zarr"):
        with entries[label]() as data:
            np.testing.assert_allclose(data.seafloor_depth, [50.0, 60.0, 70.0])
    before = entries["Latest completed transect"].revision
    pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=3, freq="min"),
            "depth": [51.0, 61.0, 71.0],
        }
    ).to_csv(path, index=False)
    assert cps.build_files(root=survey)["Latest completed transect"].revision != before


def test_single_viewer_line_uses_depth_milliseconds_and_gaps():
    # Isolate this pure overlay helper from the application's optional full
    # echopype/validation dependencies, absent from the uploaded source snapshot.
    import ast
    from pathlib import Path

    from echoshader.elements.curve import curve

    path = Path(__file__).resolve().parents[2] / "app/single_viewer.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_with_seafloor"
    )
    namespace = {"pd": pd, "np": np, "curve": curve}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    overlay = namespace["_with_seafloor"]
    times = pd.date_range("2024-01-01", periods=3, freq="min")
    data = xr.Dataset(
        {"seafloor_depth": ("ping_time", [50.0, np.nan, 70.0])},
        coords={"ping_time": times},
    )
    base = hv.Curve(([1.0, 2.0], [1.0, 2.0]))
    result = overlay(base, data)
    line = result.get(1)
    np.testing.assert_allclose(line.dimension_values("depth"), [50.0, np.nan, 70.0], equal_nan=True)
    expected = times.to_numpy(dtype="datetime64[ns]").astype("int64") / 1e6
    np.testing.assert_allclose(line.dimension_values("ping_time"), expected)
    hv.render(result)
    missing = data.assign(seafloor_depth=("ping_time", [np.nan] * 3))
    assert overlay(base, missing) is base
    native = overlay(base, data, rasterized=False)
    assert np.issubdtype(native.get(1).dimension_values("ping_time").dtype, np.datetime64)
