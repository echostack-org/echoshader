import sys

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import pytest
import xarray as xr
from bokeh.document import Document

from echoshader.app.builder import build_app
from echoshader.extensions.spasso.data import read_navigation
from echoshader.extensions.spasso.plotting import (
    create_currents_plot,
    create_product_plot,
)


def settings(tmp_path):
    return {
        "dashboard": {
            "spasso_path": str(tmp_path),
            "navigation_path": str(tmp_path / "navigation"),
            "refresh_seconds": 0,
            "plot": {"width": 400, "height": 300},
        },
        "products": {},
    }


def write_navigation(tmp_path):
    directory = tmp_path / "navigation"
    directory.mkdir()

    navigation = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2024-01-01", periods=3, freq="min", tz="UTC"
            ),
            "longitude": [-125.0, -124.5, -124.0],
            "latitude": [40.0, 41.0, 42.0],
            "speed_knots": [8.0, 8.0, 8.0],
            "course_deg": [30.0, 30.0, 30.0],
        }
    )
    navigation.to_parquet(
        directory / "ship_navigation_20240101_0000.parquet",
        index=False,
    )
    return navigation


def test_parquet_navigation(tmp_path):
    expected = write_navigation(tmp_path)

    actual = read_navigation(settings(tmp_path))

    pd.testing.assert_frame_equal(actual, expected)


def test_missing_navigation_reports_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_navigation(settings(tmp_path))


def test_navigation_refresh_reads_new_files(tmp_path):
    original = write_navigation(tmp_path)
    config = settings(tmp_path)
    assert len(read_navigation(config)) == 3

    new_positions = original.iloc[[-1]].copy()
    new_positions["timestamp_utc"] += pd.Timedelta(minutes=1)
    new_positions["longitude"] = -123.5
    new_positions.to_parquet(
        tmp_path / "navigation" / "ship_navigation_20240101_0003.parquet",
        index=False,
    )

    refreshed = read_navigation(config)

    assert len(refreshed) == 4
    assert refreshed.longitude.iloc[-1] == -123.5
    assert refreshed.timestamp_utc.is_monotonic_increasing


def test_spasso_standalone_has_no_acoustic_source(tmp_path, monkeypatch):
    write_navigation(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("Opened acoustic data")

    monkeypatch.setattr(xr, "open_zarr", fail)
    modules_before = set(sys.modules)

    config = {
        "extensions": {
            "spasso": {
                "factory": "echoshader.extensions.spasso:build_tabs",
                "requires_source": False,
                "root": str(tmp_path),
                **settings(tmp_path),
            }
        }
    }

    app = build_app(config)
    pages = app.main[0]
    assert len(pages) == 2

    extension = pages[1]
    content = extension.eval(extension.object)
    content.get_root(Document())

    newly_imported = set(sys.modules) - modules_before
    assert not any(
        name == "echodataflow" or name.startswith("echodataflow.")
        for name in newly_imported
    )


def test_products_and_current_maps_render_without_downloads(tmp_path):
    latitude = np.linspace(39, 43, 10)
    longitude = np.linspace(-126, -123, 12)

    ds = xr.Dataset(
        {
            "analysed_sst": (
                ("time", "latitude", "longitude"),
                np.full((1, 10, 12), 288.15),
            ),
            "ugos": (
                ("time", "latitude", "longitude"),
                np.full((1, 10, 12), 0.1),
            ),
            "vgos": (
                ("time", "latitude", "longitude"),
                np.full((1, 10, 12), 0.2),
            ),
        },
        coords={
            "time": [np.datetime64("2024-01-01")],
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    ds.to_netcdf(tmp_path / "20240101_Copernicus_PHY.nc")
    ds.to_netcdf(tmp_path / "20240101_Copernicus_SST_L4.nc")

    plot = create_product_plot(
        product_name="sst",
        product_config={
            "patterns": ["*_Copernicus_SST_L4.nc"],
            "variables": ["analysed_sst"],
            "coastlines": False,
            "transform": "kelvin_to_celsius",
            "clim": [10, 20],
        },
        spasso_dir=tmp_path,
        bounds=(-126, -123, 39, 43),
        width=400,
        height=300,
    )
    assert not isinstance(plot, pn.viewable.Viewable)
    hv.render(plot)

    currents = create_currents_plot(
        currents_config={"coastlines": False},
        spasso_dir=tmp_path,
        bounds=(-126, -123, 39, 43),
        width=400,
        height=300,
    )
    assert not isinstance(currents, pn.viewable.Viewable)
    hv.render(currents)