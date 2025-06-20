import warnings
from typing import Optional, Union
from pathlib import Path



import geopandas as gpd
import holoviews as hv
import geoviews as gv


from .plots.geo_grid_plot import create_geo_plot
from bokeh.util.warnings import BokehUserWarning
from pandas.api.extensions import register_dataframe_accessor

warnings.simplefilter(action="ignore", category=BokehUserWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

hv.extension("bokeh", logo=False)


@register_dataframe_accessor("geo_viz")
class GeoVisualizer:
    """
    GeoVisualizer - A simple visualization tool for GeoJSON data.

    This class provides a lightweight visualization toolset for displaying GeoJSON 
    data on interactive maps.

    Methods
    -------
        plot(value_column, cmap, alpha, point_size, width, height, tile_source):
            Display GeoJSON data on an interactive map.

        save(filename, format):
            Save the current visualization.

    Examples
    --------
    Basic usage:
        gdf = gpd.read_file("data.geojson")
        map_plot = gdf.geo_viz.plot(value_column='population')
        map_plot.show()

    Custom styling:
        map_plot = gdf.geo_viz.plot(
            value_column='density',
            cmap='viridis',
            alpha=0.8,
            point_size=10
        )
    """

    def __init__(self, gdf: gpd.GeoDataFrame):
        super().__init__()
        self.gdf = gdf
        self._validate_geodataframe()

    def _validate_geodataframe(self):
        """Validate that we have a proper GeoDataFrame with geometry"""
        if 'geometry' not in self.gdf.columns:
            raise ValueError("GeoDataFrame must have a 'geometry' column")
        
        if self.gdf.geometry.empty:
            raise ValueError("GeoDataFrame geometry cannot be empty")

    def plot(
        self,
        value_column: Optional[str] = None,
        cmap: str = "viridis",
        alpha: float = 0.7,
        point_size: Union[int, float] = 10,
        width: int = 800,
        height: int = 600,
        tile_source: str = "OSM",
        title: Optional[str] = None
    ):
        """
        Display GeoJSON data on an interactive map.

        Parameters
        ----------
        value_column : str, optional
            Column name to use for coloring features. If None, uses uniform color.
        cmap : str, optional
            Colormap for the visualization. Default is 'viridis'.
        alpha : float, optional
            Transparency of features (0-1). Default is 0.7.
        point_size : int or float, optional
            Size of points (for Point geometries). Default is 10.
        width : int, optional
            Plot width in pixels. Default is 800.
        height : int, optional
            Plot height in pixels. Default is 600.
        tile_source : str, optional
            Background tile source. Options: 'OSM', 'CartoDB', 'Stamen'. Default is 'OSM'.
        title : str, optional
            Plot title. If None, auto-generates based on data.

        Returns
        -------
        holoviews.Overlay
            The map visualization.

        Examples
        --------
        Basic plot:
            map_plot = gdf.geo_viz.plot()

        Plot with value column:
            map_plot = gdf.geo_viz.plot(value_column='population', cmap='plasma')
        """
        
        # Create the plot
        return create_geo_plot(
            self.gdf,
            value_column=value_column,
            cmap=cmap,
            alpha=alpha,
            point_size=point_size,
            width=width,
            height=height,
            tile_source=tile_source,
            title=title
        )

    def save(self, filename: str = "geo_plot.html", format: str = "html"):
        """
        Save the current visualization.

        Parameters
        ----------
        filename : str, optional
            Output filename. Default is "geo_plot.html".
        format : str, optional
            Output format ('html', 'png', 'svg'). Default is "html".

        Returns
        -------
        str
            Path to saved file.

        Examples
        --------
        Save as HTML:
            path = gdf.geo_viz.save("my_map.html")

        Save as PNG:
            path = gdf.geo_viz.save("my_map.png", format="png")
        """
        # Create the plot with default settings
        plot = self.plot()
        
        try:
            if format.lower() == "html":
                hv.save(plot, filename)
            else:
                hv.save(plot, filename, fmt=format)
            
            return str(Path(filename).absolute())
            
        except Exception as e:
            raise ValueError(f"Failed to save plot: {e}")


def load_and_plot(file_path: Union[str, Path], **kwargs):
    """
    Convenience function to load GeoJSON and create a plot in one step.
    
    Parameters
    ----------
    file_path : str or Path
        Path to GeoJSON file
    **kwargs : dict
        Additional arguments passed to plot()
        
    Returns
    -------
    holoviews.Overlay
        The map visualization
        
    Examples
    --------
    Quick plot:
        map_plot = load_and_plot("data.geojson", value_column="population")
    """
    gdf = gpd.read_file(file_path)
    return gdf.geo_viz.plot(**kwargs)