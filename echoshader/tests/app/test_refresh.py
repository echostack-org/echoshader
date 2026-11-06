"""Refresh regression checks; no network downloads or running server required."""

import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
import sys

import holoviews as hv
import geoviews as gv
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr

from echoshader.app.refresh import watch_changes
from echoshader.app.data import source_revision
from echoshader.extensions.spasso.data import NavigationReader
from echoshader.extensions.spasso import app as spasso
from echoshader.extensions.cps import app as cps
from echoshader.app import pages

hv.extension("bokeh")


def test_unchanged_and_failed_updates():
    revision = [0]
    calls = []
    errors = []
    fail = [True]

    def update(value):
        calls.append(value)
        if fail[0]:
            raise OSError("writer busy")

    check = watch_changes(lambda: revision[0], update, seconds=0, on_error=errors.append)
    assert not check()
    fail[0] = False
    assert check()
    assert not check()
    revision[0] = 1
    assert check()
    assert calls == [0, 0, 1] and len(errors) == 1


def test_false_update_is_retried():
    attempts = []

    def update(value):
        attempts.append(value)
        return len(attempts) > 1

    check = watch_changes(lambda: 0, update, seconds=0)
    assert not check()
    assert check()
    assert not check()


def test_callback_stops_with_session(monkeypatch):
    callbacks = []
    destroy = []
    timer = SimpleNamespace(stop=lambda: callbacks.append("stopped"))
    monkeypatch.setattr(pn.state, "add_periodic_callback", lambda fn, **kw: timer)
    monkeypatch.setattr(pn.state, "on_session_destroyed", destroy.append)
    watch_changes(lambda: 0, lambda revision: None, seconds=60)
    destroy[0](None)
    assert callbacks == ["stopped"]


def test_same_metadata_republication_changes_revision(tmp_path):
    path = tmp_path / "cache.zarr"
    path.mkdir()
    metadata = path / ".zmetadata"
    metadata.write_text("{}")
    old = source_revision(path)
    stat = metadata.stat()
    os.utime(metadata, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    assert source_revision(path) != old


def navigation_file(directory, minute=0):
    directory.mkdir(exist_ok=True)
    path = directory / f"ship_navigation_20240101_00{minute:02}.parquet"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2024-01-01", periods=2, freq="min", tz="UTC")
            + pd.Timedelta(minutes=minute),
            "latitude": [40.0, 40.01],
            "longitude": [-125.0, -124.99],
        }
    ).to_parquet(path, index=False)
    return path


def test_navigation_reads_only_changed_chunks(tmp_path, monkeypatch):
    navigation_file(tmp_path)
    reader = NavigationReader({"dashboard": {"navigation_path": str(tmp_path)}})
    original = pd.read_parquet
    calls = []

    def read(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(pd, "read_parquet", read)
    assert len(reader.read()) == 2
    reader.read()
    assert len(calls) == 1
    navigation_file(tmp_path, 2)
    assert len(reader.read()) == 4
    assert len(calls) == 2


def capture_checks(monkeypatch, module):
    checks = []

    def watch(*args, **kwargs):
        kwargs["seconds"] = 0
        check = watch_changes(*args, **kwargs)
        checks.append(check)
        return check

    monkeypatch.setattr(module, "watch_changes", watch)
    return checks


def test_spasso_waits_and_only_rebuilds_changed_products(tmp_path, monkeypatch):
    checks = capture_checks(monkeypatch, spasso)
    calls = []

    def plot(**kwargs):
        calls.append(kwargs["product_name"])
        return gv.Image((np.array([-126.0, -124.0]), np.array([39.0, 41.0]), np.ones((2, 2))))

    monkeypatch.setattr(spasso, "create_product_plot", plot)
    config = {
        "dashboard": {
            "spasso_path": str(tmp_path),
            "navigation_path": str(tmp_path / "navigation"),
            "refresh_seconds": 0,
        },
        "products": {"sst": {"patterns": ["*_SST.nc"], "variables": ["sst"], "tab": "Products"}},
    }
    app = spasso.create_spasso_app(config)
    lazy = app[-1][0]
    lazy.eval(lazy.object)
    assert "Waiting" in app[2].object
    navigation_file(tmp_path / "navigation")
    (tmp_path / "20240101_SST.nc").write_bytes(b"placeholder")
    assert checks[0]()
    assert calls == ["sst"]
    assert not checks[0]()
    assert calls == ["sst"]
    navigation_file(tmp_path / "navigation", 2)
    checks[0]()
    assert calls == ["sst"]
    (tmp_path / "20240102_SST.nc").write_bytes(b"new")
    checks[0]()
    assert calls == ["sst", "sst"]


def fake_builder(monkeypatch, calls):
    import types

    builder = types.ModuleType("echoshader.app.builder")

    def data_path(value, directory):
        path = Path(value)
        return (
            (Path(directory) / path).resolve()
            if not path.is_absolute() and directory
            else path.resolve()
        )

    def build(config, datasets, **kwargs):
        calls.append(datasets)
        return SimpleNamespace(sidebar=[], main=[pn.pane.Markdown("Loaded")])

    builder._build_file_app = build
    builder._data_path = data_path
    monkeypatch.setitem(sys.modules, "echoshader.app.builder", builder)


def test_file_page_waits_then_loads_without_rebuilding_unchanged(tmp_path, monkeypatch):
    calls = []
    fake_builder(monkeypatch, calls)
    checks = capture_checks(monkeypatch, pages)
    config = {
        "app": {"refresh_seconds": 0},
        "data": {"survey": {"type": "zarr", "path": "data.zarr"}},
    }
    app = pages.build_pages(config, config_dir=tmp_path)
    lazy = app.main[0][1]
    view = lazy.eval(lazy.object)
    assert len(calls) == 0
    xr.Dataset({"Sv": ("ping_time", [1.0, 2.0])}, coords={"ping_time": [0, 1]}).to_zarr(
        tmp_path / "data.zarr", zarr_format=2
    )
    assert checks[0]()
    assert len(calls) == 1
    assert not checks[0]()
    assert len(calls) == 1
    assert "Loaded" in view[0].object


def test_file_catalogue_refresh_preserves_selection(tmp_path, monkeypatch):
    calls = []
    fake_builder(monkeypatch, calls)
    checks = capture_checks(monkeypatch, pages)
    version = [0]

    def catalogue(**kwargs):
        if not version[0]:
            return {}

        def loader():
            return xr.Dataset({"Sv": ("ping_time", [1.0, 2.0])})

        loader.revision = version[0]
        return {"Latest completed transect": loader, "Transect 001": loader}

    monkeypatch.setattr(pages, "_factory", lambda path: catalogue)
    app = pages.build_pages(
        {
            "app": {"refresh_seconds": 0},
            "data": {"survey": {"type": "zarr", "path": "missing.zarr"}},
            "file_catalogue": {"factory": "unused:factory"},
        },
        config_dir=tmp_path,
    )
    lazy = app.main[0][1]
    lazy.eval(lazy.object)
    version[0] = 1
    checks[0]()
    menu = app.sidebar[0][0]
    menu.value = "Transect 001"
    before = len(calls)
    checks[0]()
    assert len(calls) == before
    version[0] = 2
    checks[0]()
    assert menu.value == "Transect 001" and len(calls) == before + 1


def ledger(path):
    with sqlite3.connect(path) as con:
        con.execute(
            "CREATE TABLE transects (transect_part TEXT, transect_number TEXT, start_time TEXT, end_time TEXT, status TEXT)"
        )
        con.execute(
            "CREATE TABLE sv_cps (cps_filename TEXT, first_ping_time TEXT, last_ping_time TEXT, status TEXT)"
        )
        con.execute(
            "INSERT INTO transects VALUES ('001','001','2024-01-01T00:00:00Z','2024-01-01T00:02:00Z','completed')"
        )


def test_cps_catalogue_waits_before_database_and_handles_table_startup(tmp_path):
    assert cps.build_files(root=tmp_path) == {}
    sqlite3.connect(tmp_path / "processing.db").close()
    assert cps.build_files(root=tmp_path) == {}
    ledger(tmp_path / "processing.db")
    entries = cps.build_files(root=tmp_path)
    assert "Latest completed transect" in entries
    assert hasattr(entries["Latest completed transect"], "revision")


def test_cps_unchanged_refresh_keeps_existing_plot(tmp_path, monkeypatch):
    ledger(tmp_path / "processing.db")
    times = pd.date_range("2024-01-01", periods=3, freq="min").values
    ds = xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"), np.full((3, 3, 4), -50.0)),
            "depth": ("range_sample", [1.0, 2.0, 3.0, 4.0]),
            "longitude": ("ping_time", [-125.0, -124.9, -124.8]),
            "latitude": ("ping_time", [40.0, 40.0, 40.0]),
            "frequency_nominal": ("channel", [18000, 38000, 70000]),
        },
        coords={"channel": ["18", "38", "70"], "ping_time": times, "range_sample": range(4)},
    )
    store = tmp_path / "CPS_Sv" / "a.zarr"
    store.parent.mkdir()
    ds.to_zarr(store, zarr_format=2)
    with sqlite3.connect(tmp_path / "processing.db") as con:
        con.execute(
            "INSERT INTO sv_cps VALUES ('a.zarr','2024-01-01T00:00:00Z','2024-01-01T00:02:00Z','completed')"
        )
    calls = []
    monkeypatch.setattr(
        cps, "tricolor_echogram", lambda *a, **k: calls.append(1) or hv.Curve([1, 2])
    )
    monkeypatch.setattr(cps, "_transect_map", lambda *a, **k: pn.Spacer())
    checks = capture_checks(monkeypatch, cps)
    _, tabs = cps.build_tabs(
        ds=xr.Dataset(),
        root=tmp_path,
        config={"refresh_seconds": 0},
        max_pings=400,
        max_range_samples=160,
    )
    view = tabs[0][1](None, (-100, -30))
    assert len(calls) == 1
    view[0][1].clicks += 1
    assert len(calls) == 1
    assert not checks[0]()
    assert len(calls) == 1
    metadata = store / ".zmetadata"
    stat = metadata.stat()
    os.utime(metadata, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    checks[0]()
    assert len(calls) == 2
    with sqlite3.connect(tmp_path / "processing.db") as con:
        con.execute(
            "INSERT INTO transects VALUES ('002','002','2024-01-02T00:00:00Z','2024-01-02T00:02:00Z','completed')"
        )
    checks[0]()
    assert len(calls) == 2  # unchanged first transect retains its plot


def test_single_file_reopens_changed_metadata_only(tmp_path, monkeypatch):
    calls = []
    fake_builder(monkeypatch, calls)
    checks = capture_checks(monkeypatch, pages)
    path = tmp_path / "data.zarr"
    xr.Dataset({"Sv": ("ping_time", [1.0, 2.0])}).to_zarr(path, zarr_format=2)
    app = pages.build_pages(
        {"app": {"refresh_seconds": 0}, "data": {"survey": {"type": "zarr", "path": str(path)}}}
    )
    lazy = app.main[0][1]
    lazy.eval(lazy.object)
    assert len(calls) == 1
    checks[0]()
    assert len(calls) == 1
    metadata = path / ".zmetadata"
    stat = metadata.stat()
    os.utime(metadata, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    checks[0]()
    assert len(calls) == 2


def test_generic_transects_reuse_unchanged_views(tmp_path, monkeypatch):
    from echoshader.extensions import transects

    calls = []
    monkeypatch.setattr(
        transects, "tricolor_echogram", lambda *a, **k: calls.append(1) or hv.Curve([1, 2])
    )
    monkeypatch.setattr(transects, "transect_map", lambda *a, **k: pn.Spacer())
    checks = capture_checks(monkeypatch, transects)
    _, tabs = transects.build_tabs(
        ds=xr.Dataset(),
        root=tmp_path,
        config={"refresh_seconds": 0, "vert_dim": "depth"},
        max_pings=400,
        max_range_samples=160,
    )
    tabs[0][1]()
    assert len(calls) == 0

    def write(name):
        xr.Dataset(
            {
                "Sv": (("channel", "ping_time", "range_sample"), np.ones((3, 3, 4))),
                "depth": ("range_sample", [1.0, 2.0, 3.0, 4.0]),
            },
            coords={"channel": ["18", "38", "70"], "ping_time": range(3), "range_sample": range(4)},
        ).to_zarr(tmp_path / name, zarr_format=2)

    write("a.zarr")
    checks[0]()
    assert len(calls) == 1
    checks[0]()
    assert len(calls) == 1
    write("b.zarr")
    checks[0]()
    assert len(calls) == 2


