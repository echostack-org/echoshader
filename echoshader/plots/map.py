from typing import Union

import geoviews
import numpy
import pandas
import xarray
from pyproj import Transformer

from ..utils import EPSG_coordsys, EPSG_mercator, gram_opts

opt_tile = geoviews.opts(tools=["box_select"])


def get_track_corners(MVBS_ds: xarray.Dataset):
    """
    Calculate the geographic bounding box corners of the track from a given MVBS_ds.

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        A dataset containing the track data.

    Returns
    -------
    tuple
        A tuple containing the four geographic bounding box corners of the track.

    Notes
    -----
    This function calculates the minimum and maximum latitude and longitude values from the
    provided xarray dataset `MVBS_ds`. These values represent the geographic bounding box of the
    track. The `MVBS_ds` should contain thenecessary variables 'longitude' and 'latitude'
    representing the spatial coordinates of the track.

    The returned tuple contains four values:
    - left: The minimum longitude value (westernmost point) of the track.
    - bottom: The minimum latitude value (southernmost point) of the track.
    - right: The maximum longitude value (easternmost point) of the track.
    - top: The maximum latitude value (northernmost point) of the track.

    Example
    -------
        corners = get_track_corners(MVBS_ds)
        print(corners)
    """
    left = numpy.nanmin(MVBS_ds.longitude.values)
    bottom = numpy.nanmin(MVBS_ds.latitude.values)
    right = numpy.nanmax(MVBS_ds.longitude.values)
    top = numpy.nanmax(MVBS_ds.latitude.values)
    return left, bottom, right, top


def convert_EPSG(
    lat: Union[int, float], lon: Union[int, float], mercator_to_coord: bool = True
):
    """
    Converts coordinates between EPSG coordinate reference systems (CRS).

    Args:
        lat (int, float): Latitude value.
        lon (int, float): Longitude value.
        mercator_to_coord (bool, optional):
            If True, converts from EPSG Mercator to coordinate system.
            If False, converts from coordinate system to EPSG Mercator. Default is True.

    Returns:
        tuple: A tuple containing the converted latitude and longitude values.

    Example usage:
        # Convert from EPSG Mercator to coordinate system
        lat, lon = convert_EPSG(0, 0)

        # Convert from coordinate system to EPSG Mercator
        lat, lon = convert_EPSG(40, -75, False)
    """
    if mercator_to_coord is True:
        transformer = Transformer.from_crs(EPSG_mercator, EPSG_coordsys)
        (lat, lon) = transformer.transform(xx=lon, yy=lat)
    else:
        transformer = Transformer.from_crs(EPSG_coordsys, EPSG_mercator)
        (lon, lat) = transformer.transform(xx=lat, yy=lon)

    return lat, lon


def convert_MVBS_to_pandas(MVBS_ds: xarray.Dataset):
    """
    Convert MVBS (Multibeam Backscatter) data from xarray.Dataset to a pandas DataFrame.

    The DataFrame will have columns for "Longitude", "Latitude", and "Ping Time".

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        xarray.Dataset containing MVBS data.
        It should include variables 'longitude', 'latitude', and 'ping_time'.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with columns "Longitude", "Latitude", and "Ping Time".
        The DataFrame will exclude rows with any missing (NaN) values.

    Examples
    --------
    # Assuming MVBS_ds is an xarray.Dataset containing MVBS data
    # Convert the MVBS data to a pandas DataFrame
    mvbs_df = convert_MVBS_to_pandas(MVBS_ds)

    # Now you can work with the data as a pandas DataFrame
    print(mvbs_df.head())
    """
    all_pd_data = pandas.concat(
        [
            pandas.DataFrame(MVBS_ds.longitude.values, columns=["Longitude"]),
            pandas.DataFrame(MVBS_ds.latitude.values, columns=["Latitude"]),
            pandas.DataFrame(MVBS_ds.ping_time.values, columns=["Ping Time"]),
        ],
        axis=1,
    )

    all_pd_data = all_pd_data.dropna()

    return all_pd_data


def tile_plot(map_tiles: str):
    """
    Load and customize map tiles from GeoViews tile sources.

    This function loads a specific map tile layer from GeoViews tile sources module
    and applies the specified `opt_tile` (plotting options) to customize the appearance.

    Parameters
    ----------
    map_tiles : str
        The name of the map tile layer to load. It should be one of the supported
        tile source names available in the GeoViews tile_sources module.
        Source: https://holoviews.org/reference/elements/bokeh/Tiles.html

    Returns
    -------
    geoviews.element.Tiles
        A customized map tile layer (tiles) loaded from GeoViews tile sources,
        rendered with the provided `opt_tile`.

    Examples
    --------
    # Load and customize the "OSM" (OpenStreetMap) map tiles
    osm_tiles = tile_plot("OSM")
    """
    tiles = getattr(geoviews.tile_sources, map_tiles).opts(opt_tile)

    return tiles


def track_plot(MVBS_ds: xarray.Dataset):
    """
    Plot the ship's track on a map using GeoViews.

    This function takes an xarray.Dataset containing MVBS (Multibeam Backscatter) data,
    converts it into a pandas DataFrame, and plots the ship's track on a map using GeoViews.
    The starting point of the ship's track is plotted as a single point, and the entire path
    of the ship's track is plotted as a line on the map.

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        xarray.Dataset containing MVBS data.
        It should include variables 'longitude', 'latitude', and 'ping_time'.

    Returns
    -------
    geoviews.element.Path * geoviews.element.Points
        The ship's track plotted as a line (ship_path) and the starting point plotted as
        a point (starting_point).
        Both elements are combined and rendered using GeoViews with the provided `gram_opts`.

    Examples
    --------
    # Assuming MVBS_ds is an xarray.Dataset containing MVBS data
    # Plot the ship's track on a map using GeoViews
    track_plot = track_plot(MVBS_ds)

    # Load and customize the "OSM" (OpenStreetMap) map tiles
    osm_tiles = tile_plot("OSM")

    # Display the ship's track using Panel
    panel.Column(track_plot * osm_tiles)
    """

    # check if all rows has the same latitude and Longitude
    all_pd_data = convert_MVBS_to_pandas(MVBS_ds)

    if all_pd_data["Longitude"].nunique() == 1 & all_pd_data["Latitude"].nunique() == 1:
        return point_plot(MVBS_ds)

    starting_data = all_pd_data.iloc[0].values.tolist()

    # plot starting point
    starting_point = geoviews.Points(
        [starting_data],
        kdims=["Longitude", "Latitude"],
    ).opts(gram_opts)

    # plot ship path
    ship_path = geoviews.Path(
        [all_pd_data],
        kdims=["Longitude", "Latitude"],
        vdims=["Ping Time", "Longitude", "Latitude"],
    ).opts(gram_opts)

    return ship_path * starting_point


def point_plot(MVBS_ds: xarray.Dataset):
    """
    Plot a moored point on a map using GeoViews.

    This function takes an xarray.Dataset containing MVBS (Multibeam Backscatter) data,
    converts it into a pandas DataFrame, and plots a moored point on a map using GeoViews.
    The moored point represents the location at the starting data point in the dataset.

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        xarray.Dataset containing MVBS data.
        It should include variables 'longitude', 'latitude', and 'ping_time'.

    Returns
    -------
    geoviews.element.Points
        A moored point plotted on a map using GeoViews with the provided `gram_opts`.

    Examples
    --------
    # Assuming MVBS_ds is an xarray.Dataset containing MVBS data
    # Plot a moored point on a map using GeoViews
    moored_point_plot = point_plot(MVBS_ds)

    # Load and customize the "OSM" (OpenStreetMap) map tiles
    osm_tiles = tile_plot("OSM")

    # Display the moored point using Panel
    Panel.Row(moored_point_plot * osm_tiles)
    """
    all_pd_data = convert_MVBS_to_pandas(MVBS_ds)

    starting_data = all_pd_data.iloc[0].values.tolist()

    # plot moored point
    moored_point = geoviews.Points(
        [starting_data],
        kdims=["Longitude", "Latitude"],
    ).opts(gram_opts)

    return moored_point
