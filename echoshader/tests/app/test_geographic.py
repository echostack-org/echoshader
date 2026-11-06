"""Focused integration checks for the standalone geographic extraction."""

import importlib

import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr

hv.extension("bokeh")
geo = importlib.import_module("echoshader.elements.map")
data = importlib.import_module("echoshader.extensions.spasso.data")
plotting = importlib.import_module("echoshader.extensions.spasso.plotting")


def field():
    return xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=("lat", "lon"),
        coords={"lat": [40.0, 41.0, 42.0], "lon": [-127.0, -126.0, -125.0, -124.0]},
        name="sst",
    )


def test_scalar_and_vectors_render():
    scalar = geo.map(field(), clim=(0, 12), width=200, height=150)
    assert hv.render(scalar, backend="bokeh") is not None
    vector = geo.vector_field(field() + 1, field() * 0, step=1)
    assert hv.render(vector, backend="bokeh") is not None
    with pytest.raises(ValueError, match="positive integer"):
        geo.vector_field(field(), field(), step=0)


def test_product_reuses_element_and_selects_filename_date(tmp_path):
    for date in ("20260919", "20260918"):
        field().to_dataset().to_netcdf(tmp_path / f"{date}_SST.nc")
    assert data.find_latest_product_file(tmp_path, ["*_SST.nc"]).name == "20260919_SST.nc"
    plot = plotting.create_product_plot(
        product_name="sst",
        product_config={
            "patterns": ["*_SST.nc"],
            "variables": ["sst"],
            "coastlines": False,
        },
        spasso_dir=tmp_path,
        bounds=(-128, -123, 39, 43),
        width=200,
        height=150,
    )
    assert hv.render(plot, backend="bokeh") is not None


def test_watcher_schema_navigation(tmp_path):
    positions = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-09-18", periods=150, freq="min", tz="UTC"),
            "latitude": np.linspace(40, 42, 150),
            "longitude": np.linspace(-126, -125, 150),
            "speed_knots": 8.0,
            "course_deg": 30.0,
        }
    )
    # Out-of-order chunks and duplicate timestamps must reconcile correctly.
    positions.iloc[100:].to_parquet(tmp_path / "ship_navigation_b.parquet", index=False)
    positions.iloc[:101].to_parquet(tmp_path / "ship_navigation_a.parquet", index=False)
    loaded = data.read_navigation({"dashboard": {"navigation_path": str(tmp_path)}})
    pd.testing.assert_frame_equal(loaded, positions)
    track = importlib.import_module("echoshader.elements.ship_track").navigation_layer(loaded)
    assert hv.render(track, backend="bokeh") is not None
