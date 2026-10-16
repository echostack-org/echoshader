import geoviews
import numpy
import pandas
import xarray
from pyproj import Transformer

from echoshader.utils import (
    EPSG_coordsys,
    EPSG_mercator,
)


def track(
    MVBS_ds: xarray.Dataset,
    tile: str | None = "OSM",
):
    """
    Create a ship-track or moored-location visualization.

    For moving platforms, the plot contains a black trajectory plus an
    invisible Points layer containing every valid ping position. The Points
    layer is the interaction surface used by box selection.
    """
    data = _to_dataframe(MVBS_ds)

    if data.empty:
        raise ValueError("No valid longitude/latitude positions are available.")

    if data["Longitude"].nunique() == 1 and data["Latitude"].nunique() == 1:
        plot = _point(data)

    else:
        plot = _path(data)

    if tile is None:
        return plot

    return _tile(tile) * plot


def _to_dataframe(
    MVBS_ds: xarray.Dataset,
):
    """Convert track coordinates to a DataFrame."""
    return pandas.DataFrame(
        {
            "Longitude": MVBS_ds.longitude.values,
            "Latitude": MVBS_ds.latitude.values,
            "Ping Time": MVBS_ds.ping_time.values,
            "Ping Index": numpy.arange(MVBS_ds.sizes["ping_time"]),
        }
    ).dropna(
        subset=[
            "Longitude",
            "Latitude",
        ]
    )


def _path(
    data: pandas.DataFrame,
):
    """Create a black ship path and selectable ping positions."""

    ship_path = geoviews.Path(
        [data],
        kdims=[
            "Longitude",
            "Latitude",
        ],
        vdims=[
            "Ping Time",
            "Ping Index",
        ],
    ).opts(
        color="black",
        line_width=2,
    )

    selectable_points = geoviews.Points(
        data,
        kdims=[
            "Longitude",
            "Latitude",
        ],
        vdims=[
            "Ping Time",
            "Ping Index",
        ],
    ).opts(
        size=6,
        color="black",
        alpha=0.0,
        nonselection_alpha=0.0,
        selection_alpha=0.0,
    )

    return ship_path * selectable_points


def _point(
    data: pandas.DataFrame,
):
    """Create a point for a stationary/moored instrument."""
    return geoviews.Points(
        data,
        kdims=[
            "Longitude",
            "Latitude",
        ],
        vdims=[
            "Ping Time",
            "Ping Index",
        ],
    ).opts(
        color="black",
        size=8,
    )


def track_points(
    MVBS_ds: xarray.Dataset,
):
    """
    Return valid ping positions as GeoViews Points.

    This helper is intentionally tile-free and is used by the application
    layer to draw the current shared selection over the full track.
    """
    data = _to_dataframe(MVBS_ds)

    return geoviews.Points(
        data,
        kdims=[
            "Longitude",
            "Latitude",
        ],
        vdims=[
            "Ping Time",
            "Ping Index",
        ],
    )


def _tile(
    map_tiles: str,
):
    """Return a GeoViews tile source."""
    try:
        return getattr(
            geoviews.tile_sources,
            map_tiles,
        )

    except AttributeError as exc:
        raise ValueError(f"Unknown map tile source: {map_tiles!r}.") from exc


def get_track_corners(
    MVBS_ds: xarray.Dataset,
):
    """Return left, bottom, right, and top geographic bounds."""
    return (
        numpy.nanmin(MVBS_ds.longitude.values),
        numpy.nanmin(MVBS_ds.latitude.values),
        numpy.nanmax(MVBS_ds.longitude.values),
        numpy.nanmax(MVBS_ds.latitude.values),
    )


def convert_epsg(
    lat: float,
    lon: float,
    mercator_to_coord: bool = True,
):
    """Convert between geographic and Web Mercator coordinates."""

    if mercator_to_coord:
        transformer = Transformer.from_crs(
            EPSG_mercator,
            EPSG_coordsys,
        )

        lat, lon = transformer.transform(
            xx=lon,
            yy=lat,
        )

    else:
        transformer = Transformer.from_crs(
            EPSG_coordsys,
            EPSG_mercator,
        )

        lon, lat = transformer.transform(
            xx=lat,
            yy=lon,
        )

    return (
        lat,
        lon,
    )
