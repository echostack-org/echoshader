"""Plotting functions for SPASSO visualization services."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import panel as pn
import xarray as xr

from .data import (
    find_latest_product_file,
    find_variable,
    prepare_data_array,
    spatial_subset,
)
from ...elements.map import CRS, create_geographic_layers, map as geographic_map, vector_field




def _apply_transform(
    data: xr.DataArray,
    transform: str | None,
) -> xr.DataArray:
    """Apply a configured transformation to a product."""

    if transform is None:
        return data

    if transform == "kelvin_to_celsius":
        return data - 273.15

    raise ValueError(f"Unknown SPASSO visualization transform: {transform}")


def _compute_clim(
    data: xr.DataArray,
    config: dict,
) -> tuple[float, float]:
    """Determine plotting color limits."""

    configured_clim = config.get("clim")

    if configured_clim is not None:
        return (
            float(configured_clim[0]),
            float(configured_clim[1]),
        )

    values = np.asarray(data.values)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return (0.0, 1.0)

    vmin = float(finite.min())
    vmax = float(finite.max())

    if config.get("zero_min", False):
        vmin = 0.0

    if vmin == vmax:
        vmax = vmin + 1.0

    return vmin, vmax


def _product_time(
    data: xr.DataArray,
) -> str | None:
    """Extract a readable product time when available."""

    if "time" not in data.coords:
        return None

    values = np.asarray(data["time"].values).reshape(-1)

    if values.size == 0:
        return None

    try:
        value = np.datetime_as_string(
            values[0],
            unit="m",
        )
    except (TypeError, ValueError):
        return str(values[0])

    return value.replace(
        "T",
        " ",
    )


def create_product_plot(
    *,
    product_name: str,
    product_config: dict,
    spasso_dir: Path,
    bounds,
    width: int,
    height: int,
):
    """Create one configured SPASSO product plot."""

    patterns = product_config.get(
        "patterns",
        [],
    )

    product_file = find_latest_product_file(
        spasso_dir,
        patterns,
    )

    title = product_config.get(
        "title",
        product_name,
    )

    if product_file is None:
        return pn.pane.Markdown(
            (f"### {title}\n\nNo matching SPASSO product found."),
            width=width,
            height=height,
        )

    try:
        with xr.open_dataset(product_file) as ds:
            variable = find_variable(
                ds,
                product_config["variables"],
            )

            if variable is None:
                raise ValueError(f"No configured variable found: {product_config['variables']}")
            data = prepare_data_array(ds, variable)

            data = _apply_transform(
                data,
                product_config.get("transform"),
            )

            lon_min, lon_max, lat_min, lat_max = bounds

            data = spatial_subset(
                data,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
            )

            # Load before closing the NetCDF file.
            data = data.load()

        clim = _compute_clim(
            data,
            product_config,
        )

        product_time = _product_time(data)

        plot_title = title

        if product_time is not None:
            plot_title += f" — {product_time}"

        raster = geographic_map(
            data,
            variable=variable,
            title=plot_title,
            cmap=product_config.get(
                "cmap",
                "Viridis",
            ),
            clim=clim,
            width=width,
            height=height,
        )

        geographic_layers = (
            create_geographic_layers() if product_config.get("coastlines", True) else None
        )

        result = raster * geographic_layers if geographic_layers is not None else raster
        return result.opts(
            projection=CRS,
            xlim=(float(data.lon.min()), float(data.lon.max())),
            ylim=(float(data.lat.min()), float(data.lat.max())),
            framewise=False,
        )

    except Exception as exc:  # noqa: BLE001 -- report an individual product error in the dashboard
        return pn.pane.Markdown(
            (f"### {title}\n\nCould not load `{product_file.name}`.\n\n`{exc}`"),
            width=width,
            height=height,
        )


def create_currents_plot(
    *,
    currents_config: dict,
    spasso_dir: Path,
    bounds,
    width: int,
    height: int,
):
    """Create surface-current speed and vector visualization."""

    title = currents_config.get(
        "title",
        "Surface Geostrophic Currents",
    )

    pattern = currents_config.get(
        "pattern",
        "????????_Copernicus_PHY.nc",
    )

    product_file = find_latest_product_file(
        spasso_dir,
        [pattern],
    )

    if product_file is None:
        return pn.pane.Markdown(
            (f"### {title}\n\nNo matching physical-oceanography product found."),
            width=width,
            height=height,
        )

    try:
        with xr.open_dataset(product_file) as ds:
            u_name = find_variable(
                ds,
                [
                    currents_config.get(
                        "u_variable",
                        "ugos",
                    )
                ],
            )

            v_name = find_variable(
                ds,
                [
                    currents_config.get(
                        "v_variable",
                        "vgos",
                    )
                ],
            )

            u = prepare_data_array(
                ds,
                u_name,
            )

            v = prepare_data_array(
                ds,
                v_name,
            )

            lon_min, lon_max, lat_min, lat_max = bounds

            u = spatial_subset(
                u,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
            )

            v = spatial_subset(
                v,
                lon_min,
                lon_max,
                lat_min,
                lat_max,
            )

            u, v = xr.align(
                u,
                v,
                join="inner",
            )

            speed = np.sqrt(u**2 + v**2)

            speed.name = "current_speed"

            u = u.load()
            v = v.load()
            speed = speed.load()

        clim = _compute_clim(
            speed,
            currents_config,
        )

        product_time = _product_time(speed)

        plot_title = title

        if product_time is not None:
            plot_title += f" — {product_time}"

        raster = geographic_map(
            speed,
            variable="current_speed",
            title=plot_title,
            cmap=currents_config.get(
                "cmap",
                "Viridis",
            ),
            clim=clim,
            width=width,
            height=height,
        )

        result = raster

        vector_config = currents_config.get(
            "vectors",
            {},
        )

        if vector_config.get(
            "enabled",
            True,
        ):
            vectors = vector_field(
                u, v,
                step=int(vector_config.get("step", 2)),
                scale=float(vector_config.get("scale", 0.5)),
            )

            result = result * vectors

        geographic_layers = (
            create_geographic_layers() if currents_config.get("coastlines", True) else None
        )

        result = result * geographic_layers if geographic_layers is not None else result
        return result.opts(
            projection=CRS,
            xlim=(float(speed.lon.min()), float(speed.lon.max())),
            ylim=(float(speed.lat.min()), float(speed.lat.max())),
            framewise=False,
        )

    except Exception as exc:  # noqa: BLE001 -- report an individual product error in the dashboard
        return pn.pane.Markdown(
            (f"### {title}\n\nCould not load `{product_file.name}`.\n\n`{exc}`"),
            width=width,
            height=height,
        )
