# import numpy as np
# import geoviews as gv
# import holoviews as hv
# import geopandas as gpd
# from geoviews import dim
# from typing import Optional, Union


# def create_geo_plot(
#     gdf: gpd.GeoDataFrame,
#     value_column: Optional[str] = None,
#     cmap: str = "viridis",
#     alpha: float = 0.7,
#     point_size: Union[int, float] = 10,
#     width: int = 800,
#     height: int = 600,
#     tile_source: str = "OSM",
#     title: Optional[str] = None
# ):
#     """
#     Create an interactive map visualization from a GeoDataFrame.

#     Parameters
#     ----------
#     gdf : gpd.GeoDataFrame
#         GeoDataFrame containing the geographic data to plot.
#     value_column : str, optional
#         Column name to use for coloring features. If None, uses uniform color.
#     cmap : str, optional
#         Colormap for the visualization. Default is 'viridis'.
#     alpha : float, optional
#         Transparency of features (0-1). Default is 0.7.
#     point_size : int or float, optional
#         Size of points (for Point geometries). Default is 10.
#     width : int, optional
#         Plot width in pixels. Default is 800.
#     height : int, optional
#         Plot height in pixels. Default is 600.
#     tile_source : str, optional
#         Background tile source. Options: 'OSM', 'CartoDB', 'Stamen'. Default is 'OSM'.
#     title : str, optional
#         Plot title. If None, auto-generates based on data.

#     Returns
#     -------
#     geoviews.Overlay
#         The map visualization as a GeoViews Overlay object.
#     """
#     # Set up tile source
#     tile_sources = {
#         'OSM': gv.tile_sources.OSM,
#         'EsriImagery': gv.tile_sources.EsriImagery,
#         'OpenTopoMap': gv.tile_sources.StamenTerrain
#     }
#     tiles = tile_sources.get(tile_source, gv.tile_sources.OSM)

#     # Get geometry type
#     geom_type = gdf.geometry.geom_type.iloc[0]

#     # Prepare hover tooltips
#     hover_tooltips = []

#     # Add value column to tooltips if specified
#     if value_column and value_column in gdf.columns:
#         hover_tooltips.append((value_column, f'@{value_column}'))

#     # Add all other non-geometry columns to tooltips
#     for col in gdf.columns:
#         if col not in ['geometry', value_column]:
#             hover_tooltips.append((col, f'@{col}'))

#     # If no tooltips, add a basic one
#     if not hover_tooltips:
#         hover_tooltips = [('Index', '@index')]

#     # Common options for all geometry types
#     common_opts = {
#         'alpha': alpha,
#         'tools': ['hover', 'pan', 'zoom_in', 'zoom_out', 'reset'],
#         'hover_tooltips': hover_tooltips,
#         'width': width,
#         'height': height,
#         'title': title or f'{geom_type} Features'
#     }

#     # Add color options if value column is specified
#     if value_column and value_column in gdf.columns:
#         common_opts.update({
#             'color': value_column,
#             'cmap': cmap,
#             'colorbar': True,
#             'colorbar_opts': {'title': value_column}
#         })
#     else:
#         common_opts['color'] = 'blue'

#     # Create appropriate GeoViews element based on geometry type
#     if geom_type in ['Point', 'MultiPoint']:
#         # For points
#         geo_element = gv.Points(gdf)
#         if 'color' in common_opts and common_opts['color'] != 'blue':
#             # If we have a value column, don't override with size
#             geo_element = geo_element.opts(size=point_size, **common_opts)
#         else:
#             geo_element = geo_element.opts(size=point_size, **common_opts)

#     elif geom_type in ['LineString', 'MultiLineString']:
#         # For lines
#         geo_element = gv.Path(gdf)
#         geo_element = geo_element.opts(line_width=2, **common_opts)

#     elif geom_type in ['Polygon', 'MultiPolygon']:
#         # For polygons
#         geo_element = gv.Polygons(gdf)
#         geo_element = geo_element.opts(line_color='black', line_width=0.5, **common_opts)

#     else:
#         # For mixed or unknown geometry types, try to use Shape
#         geo_element = gv.Shape(gdf)
#         geo_element = geo_element.opts(**common_opts)

#     # Create the final map
#     geo_map = tiles * geo_element

#     return geo_map


# def get_data_bounds(gdf: gpd.GeoDataFrame) -> dict:
#     """
#     Get the spatial bounds of the GeoDataFrame.

#     Parameters
#     ----------
#     gdf : gpd.GeoDataFrame
#         GeoDataFrame to get bounds from

#     Returns
#     -------
#     dict
#         Dictionary containing bounds information
#     """
#     bounds = gdf.total_bounds  # Returns [minx, miny, maxx, maxy]

#     return {
#         'min_x': bounds[0],
#         'min_y': bounds[1],
#         'max_x': bounds[2],
#         'max_y': bounds[3],
#         'center_x': (bounds[0] + bounds[2]) / 2,
#         'center_y': (bounds[1] + bounds[3]) / 2,
#         'width': bounds[2] - bounds[0],
#         'height': bounds[3] - bounds[1]
#     }
