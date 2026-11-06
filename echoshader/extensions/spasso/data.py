"""Data access and preparation for SPASSO visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def load_navigation(
    navigation_dir: Path,
) -> pd.DataFrame:
    """Load all available ship-navigation Parquet files."""

    return NavigationReader({"dashboard": {"navigation_path": navigation_dir}}).read()


def find_latest_product_file(
    spasso_dir: Path,
    patterns: list[str],
) -> Path | None:
    """Find the newest local file matching any product pattern."""

    files: list[Path] = []

    for pattern in patterns:
        files.extend(spasso_dir.glob(pattern))

    files = list(set(files))

    if not files:
        return None

    return max(
        files,
        key=lambda path: path.name,
    )


def find_variable(
    ds: xr.Dataset,
    candidates: list[str],
) -> str | None:
    """Find the first matching variable in a dataset."""

    for variable in candidates:
        if variable in ds.data_vars:
            return variable

    return None


def get_coordinate_values(
    ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find latitude and longitude values in a SPASSO dataset."""

    lat = next(
        (ds[name].values for name in ("lats", "lat", "latitude") if name in ds.variables),
        None,
    )

    lon = next(
        (ds[name].values for name in ("lons", "lon", "longitude") if name in ds.variables),
        None,
    )

    if lat is None or lon is None:
        return None

    return np.asarray(lat), np.asarray(lon)


def spatial_subset(
    data: xr.DataArray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> xr.DataArray:
    """Subset a DataArray regardless of coordinate direction."""

    lon = data["lon"]
    lat = data["lat"]

    lon_slice = (
        slice(lon_min, lon_max) if lon.values[0] < lon.values[-1] else slice(lon_max, lon_min)
    )

    lat_slice = (
        slice(lat_min, lat_max) if lat.values[0] < lat.values[-1] else slice(lat_max, lat_min)
    )

    return data.sel(
        lon=lon_slice,
        lat=lat_slice,
    )


def prepare_data_array(
    ds: xr.Dataset,
    variable: str,
) -> xr.DataArray:
    """Prepare a SPASSO variable for geographic plotting."""

    data = ds[variable]

    if "time" in data.dims:
        data = data.isel(time=0)

    if "depth" in data.dims and data.sizes["depth"] == 1:
        data = data.isel(depth=0)

    coordinates = get_coordinate_values(ds)

    if coordinates is None:
        raise ValueError("Could not identify latitude/longitude coordinates.")

    lat_values, lon_values = coordinates

    lat_dim = next(
        (dim for dim in data.dims if dim.lower() in {"lat", "latitude", "y"}),
        None,
    )

    lon_dim = next(
        (dim for dim in data.dims if dim.lower() in {"lon", "longitude", "x"}),
        None,
    )

    if lat_dim is None or lon_dim is None:
        raise ValueError(
            f"Could not identify geographic dimensions for {variable}. Dimensions are {data.dims}."
        )

    data = data.assign_coords(
        {
            lat_dim: lat_values,
            lon_dim: lon_values,
        }
    )

    rename = {}

    if lat_dim != "lat":
        rename[lat_dim] = "lat"

    if lon_dim != "lon":
        rename[lon_dim] = "lon"

    if rename:
        data = data.rename(rename)

    return data


def read_navigation(config):
    """Read recorded navigation; example and live watcher use the same files."""
    return load_navigation(Path(config["dashboard"]["navigation_path"]))


class NavigationReader:
    """Cache recorded chunks per session; reread only new or changed files."""

    def __init__(self, config):
        self.directory = Path(config["dashboard"]["navigation_path"])
        self._chunks = {}

    def revision(self):
        from ...app.refresh import file_revision

        return tuple(
            file_revision(path) for path in sorted(self.directory.glob("ship_navigation_*.parquet"))
        )

    def read(self):
        from ...app.refresh import file_revision

        paths = sorted(self.directory.glob("ship_navigation_*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No navigation Parquet files found in {self.directory}")
        chunks = {}
        for path in paths:
            revision = file_revision(path)
            previous = self._chunks.get(path)
            if previous is not None and previous[0] == revision:
                chunks[path] = previous
            else:
                data = pd.read_parquet(path)
                if file_revision(path) != revision:
                    raise OSError(f"Navigation chunk changed while reading: {path}")
                chunks[path] = (revision, data)
        navigation = pd.concat([entry[1] for entry in chunks.values()], ignore_index=True)
        navigation["timestamp_utc"] = pd.to_datetime(navigation["timestamp_utc"], utc=True)
        navigation = navigation.dropna(subset=["timestamp_utc", "latitude", "longitude"])
        navigation = (
            navigation.sort_values("timestamp_utc")
            .drop_duplicates("timestamp_utc")
            .reset_index(drop=True)
        )
        self._chunks = chunks
        return navigation
