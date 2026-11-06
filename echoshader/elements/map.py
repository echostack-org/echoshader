"""Standalone geographic scalar maps, current vectors, and background layers."""

import math
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geoviews as gv
import numpy as np
from holoviews.operation.datashader import rasterize

CRS = ccrs.PlateCarree()


def map(
    data,
    *,
    variable=None,
    title="",
    cmap="Viridis",
    clim=None,
    width=520,
    height=400,
):
    """Render a named scalar DataArray with one-dimensional lon/lat coordinates.

    Returns a composable geographic plot; no files, application state, or
    acoustic logarithmic transformations are involved.
    """

    if variable is None:
        variable = data.name
    if variable is None:
        raise ValueError("Map data must have a name or an explicit variable")

    quadmesh = gv.QuadMesh(
        data,
        kdims=["lon", "lat"],
        vdims=[variable],
        crs=CRS,
    )

    return rasterize(
        quadmesh,
        width=width,
        height=height,
    ).opts(
        cmap=cmap,
        clim=clim,
        colorbar=True,
        width=width,
        height=height,
        xlabel="Longitude",
        ylabel="Latitude",
        tools=[
            "hover",
            "pan",
            "wheel_zoom",
            "box_zoom",
            "reset",
        ],
        active_tools=["wheel_zoom"],
        framewise=False,
        title=title,
        projection=CRS,
        xlim=(float(data.lon.min()), float(data.lon.max())),
        ylim=(float(data.lat.min()), float(data.lat.max())),
        padding=0,
    )


def geographic_bounds(
    latitude: float,
    longitude: float,
    radius_km: float,
) -> tuple[float, float, float, float]:
    """Return approximate geographic bounds around a position."""

    delta_lat = radius_km / 111.0

    delta_lon = radius_km / (111.0 * math.cos(math.radians(latitude)))

    return (
        longitude - delta_lon,
        longitude + delta_lon,
        latitude - delta_lat,
        latitude + delta_lat,
    )


def create_geographic_layers():
    """Create reusable land/coastline/state layers."""

    land = gv.feature.land.opts(
        fill_color="lightgray",
        line_color="black",
    )

    coastline = gv.feature.coastline.opts(
        line_color="black",
        line_width=1,
    )

    states_feature = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="50m",
        facecolor="none",
    )

    states = gv.Feature(states_feature).opts(
        line_color="gray",
        line_width=0.6,
    )

    return land * coastline * states


def vector_field(u, v, *, step=2, scale=0.5):
    """Render geographic eastward/northward components on an aligned lon/lat grid."""
    if not isinstance(step, int) or step < 1:
        raise ValueError("Vector step must be a positive integer")
    import xarray as xr

    u, v = xr.align(u, v, join="exact")
    u = u.transpose("lat", "lon").isel(lat=slice(None, None, step), lon=slice(None, None, step))
    v = v.transpose("lat", "lon").isel(lat=slice(None, None, step), lon=slice(None, None, step))
    lon, lat = np.meshgrid(u.lon.values, u.lat.values)
    angle = np.arctan2(v.values, u.values)
    magnitude = np.hypot(u.values, v.values)
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(angle) & np.isfinite(magnitude)
    return gv.VectorField(
        (lon[valid], lat[valid], angle[valid], magnitude[valid]),
        kdims=["Longitude", "Latitude"], vdims=["Angle", "Magnitude"], crs=CRS,
    ).opts(magnitude="Magnitude", scale=scale, pivot="mid")
