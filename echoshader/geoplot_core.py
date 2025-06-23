import panel
import logging
import warnings
import echoshader
import numpy as np
import panel as pn
import pandas as pd
import xarray as xr
import geoviews as gv
import holoviews as hv
import geopandas as gpd
import cartopy.crs as ccrs
import shapely.geometry as sg
import geopy.distance as distance
import cartopy.feature as cfeature


from pathlib import Path
from geoviews import dim
from geoviews import tile_sources as gvts
from shapely.geometry import box, Polygon
from typing import Optional, Tuple, Dict, Any, List, Union

from bokeh.util.warnings import BokehUserWarning
from pandas.api.extensions import register_dataframe_accessor



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

warnings.simplefilter(action="ignore", category=BokehUserWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

hv.extension("bokeh", logo=False)


@register_dataframe_accessor("geo_viz")
class GridDataVisualizer:
    """
    A class to handle data aggregation and visualization on a spatial grid.
    
    This class provides methods to aggregate any numerical data by grid cells
    and create interactive map visualizations. It's dimension-agnostic and can
    handle single or multiple metrics.
    """
    
    def __init__(self, extension: str = 'bokeh'):
        """
        Initialize the Grid Data Visualizer.
        
        Parameters:
        -----------
        extension : str, default='bokeh'
            The GeoViews extension to use for interactive plots.
        """
        gv.extension(extension)
        self.extension = extension
        logger.info(f"Initialized GridDataVisualizer with {extension} backend")
    
    def utm_string_generator(self, longitude: float, latitude: float) -> str:
        """
        Converts projection string from longitude/latitude (WGS84) to equivalent UTM.
        
        Parameters:
        -----------
        longitude : float
            Longitude coordinate
        latitude : float
            Latitude coordinate
            
        Returns:
        --------
        str
            EPSG code string for the UTM projection
        """
        # Calculate UTM band value
        utm_value = str((np.floor((longitude + 180) / 6) % 60 + 1).astype(int))
        
        # Construct string to create equivalent EPSG code
        if len(utm_value) == 1:
            utm_value = "0" + utm_value
        
        if latitude >= 0.0:
            epsg = "326" + utm_value
        else:
            epsg = "327" + utm_value
        
        logger.info(f"Generated UTM EPSG code: {epsg} for coordinates ({longitude}, {latitude})")
        return epsg
    
    def create_boundary_gdf(self, 
                           bounds: Dict[str, List[float]], 
                           projection: str = "epsg:4326") -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, int]:
        """
        Create a GeoDataFrame for the boundary rectangle from coordinate bounds.
        
        Parameters:
        -----------
        bounds : dict
            Dictionary with 'latitude' and 'longitude' keys, each containing [min, max]
        projection : str, default="epsg:4326"
            CRS projection string
            
        Returns:
        --------
        tuple
            (boundary_gdf, boundary_gdf_utm, utm_num)
            - boundary_gdf: Original boundary in specified projection
            - boundary_gdf_utm: Boundary in UTM projection
            - utm_num: UTM EPSG code as integer
        """
        # Validate bounds
        if 'latitude' not in bounds or 'longitude' not in bounds:
            raise ValueError("Bounds must contain 'latitude' and 'longitude' keys")
        
        # Get boundary coordinates
        x = bounds["longitude"]
        y = bounds["latitude"]
        
        logger.info(f"Creating boundary with longitude: {x}, latitude: {y}")
        
        # Create polygon from bounds
        boundary_polygon = Polygon([
            (np.min(x), np.min(y)),  # Bottom-left
            (np.max(x), np.min(y)),  # Bottom-right
            (np.max(x), np.max(y)),  # Top-right
            (np.min(x), np.max(y)),  # Top-left
            (np.min(x), np.min(y))   # Close polygon
        ])
        
        # Create GeoDataFrame with polygon
        boundary_gdf = gpd.GeoDataFrame(
            data={'geometry': [boundary_polygon]},
            crs=projection
        )
        
        # Convert to UTM using center point
        center_lon = (np.min(x) + np.max(x)) / 2
        center_lat = (np.min(y) + np.max(y)) / 2
        utm_code = self.utm_string_generator(center_lon, center_lat)
        utm_num = int(utm_code)
        
        boundary_gdf_utm = boundary_gdf.to_crs(utm_num)
        
        logger.info(f"Created boundary GeoDataFrame with UTM projection: EPSG:{utm_num}")
        
        return boundary_gdf, boundary_gdf_utm, utm_num
    
    def create_grid_cells(self, 
                         boundary_gdf_utm: gpd.GeoDataFrame, 
                         x_step: float, 
                         y_step: float,
                         clip_to_boundary: bool = True) -> gpd.GeoDataFrame:
        """
        Generate grid cells efficiently using vectorized operations.
        
        Parameters:
        -----------
        boundary_gdf_utm : GeoDataFrame
            Boundary GeoDataFrame in UTM projection
        x_step : float
            Cell width in meters
        y_step : float
            Cell height in meters
        clip_to_boundary : bool, default=True
            Whether to clip grid cells to exact boundary
            
        Returns:
        --------
        GeoDataFrame
            Grid cells with x, y coordinates
        """
        # Get boundaries
        xmin, ymin, xmax, ymax = boundary_gdf_utm.total_bounds
        
        logger.info(f"Creating grid cells within bounds: {xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}")
        logger.info(f"Grid resolution: {x_step}m x {y_step}m")
        
        # Create coordinate arrays
        x_coords = np.arange(xmin, xmax, x_step)
        y_coords = np.arange(ymin, ymax, y_step)
        
        # Create mesh grid
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        logger.info(f"Number of grid cells: {len(x_coords) * len(y_coords)}")
        
        # Flatten arrays for vectorized box creation
        x0 = xx.ravel()
        y0 = yy.ravel()
        x1 = x0 + x_step
        y1 = y0 + y_step
        
        # Create all boxes at once
        grid_cells = [box(x0[i], y0[i], x1[i], y1[i]) for i in range(len(x0))]
        
        # Create grid indices
        x_indices, y_indices = np.meshgrid(range(1, len(x_coords) + 1), 
                                          range(1, len(y_coords) + 1))
        
        # Create GeoDataFrame
        cells_gdf = gpd.GeoDataFrame({
            'geometry': grid_cells,
            'x': x_indices.ravel(),
            'y': y_indices.ravel()
        }, crs=boundary_gdf_utm.crs)
        
        # Optional: Clip to exact boundary
        if clip_to_boundary and boundary_gdf_utm.geometry[0].geom_type == 'Polygon':
            cells_gdf = cells_gdf[cells_gdf.intersects(boundary_gdf_utm.geometry[0])]
            logger.info(f"Grid cells after clipping: {len(cells_gdf)}")
        
        return cells_gdf
    
    def create_grid_from_bounds(self,
                               bounds: Dict[str, List[float]],
                               grid_resolution: Optional[Dict[str, float]] = None,
                               projection: str = "epsg:4326") -> Tuple[gpd.GeoDataFrame, int]:
        """
        Create a complete grid from bounds with specified resolution.
        
        This is a convenience method that combines boundary creation and grid generation.
        
        Parameters:
        -----------
        bounds : dict
            Dictionary with 'latitude' and 'longitude' keys, each containing [min, max]
        grid_resolution : dict, optional
            Dictionary with 'x_distance' and 'y_distance' in meters.
            Default is 500m x 500m for acoustic data.
        projection : str, default="epsg:4326"
            Initial CRS projection string
            
        Returns:
        --------
        tuple
            (cells_gdf, utm_num)
            - cells_gdf: Grid cells GeoDataFrame in UTM projection
            - utm_num: UTM EPSG code as integer
        """
        # Set default grid resolution for acoustic data
        if grid_resolution is None:
            grid_resolution = {
                "x_distance": 500.0,  # 500 meters for acoustic data
                "y_distance": 500.0   # 500 meters for acoustic data
            }
        
        logger.info(f"Creating grid with resolution: {grid_resolution['x_distance']}m x {grid_resolution['y_distance']}m")
        
        # Create boundary
        boundary_gdf, boundary_gdf_utm, utm_num = self.create_boundary_gdf(bounds, projection)
        
        # Create grid cells
        cells_gdf = self.create_grid_cells(
            boundary_gdf_utm,
            x_step=grid_resolution["x_distance"],
            y_step=grid_resolution["y_distance"]
        )
        
        return cells_gdf, utm_num
    
    def aggregate_data_by_grid(self, 
                               df: pd.DataFrame,
                               dimensions: Union[str, List[str]],
                               grid_x_col: str = 'grid_x', 
                               grid_y_col: str = 'grid_y',
                               agg_functions: Optional[Union[str, List[str], Dict[str, List[str]]]] = None) -> pd.DataFrame:
        """
        Aggregate data by grid cell coordinates for specified dimensions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing data and grid coordinates.
        dimensions : str or list of str
            Column name(s) for the dimension(s) to aggregate.
        grid_x_col : str, default='grid_x'
            Column name for grid x-coordinates.
        grid_y_col : str, default='grid_y'
            Column name for grid y-coordinates.
        agg_functions : str, list, or dict, optional
            Aggregation functions to apply. Can be:
            - String: Single function applied to all dimensions ('mean', 'sum', etc.)
            - List: Multiple functions applied to all dimensions
            - Dict: Dimension-specific functions {dimension: ['mean', 'sum'], ...}
            Default is ['mean', 'count', 'sum'] for all dimensions.
            
        Returns:
        --------
        pd.DataFrame
            Aggregated data by grid cell with specified statistics.
        """
        logger.info(f"Starting data aggregation by grid cells for dimensions: {dimensions}")
        
        # Ensure dimensions is a list
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        
        # Validate input columns
        required_cols = [grid_x_col, grid_y_col] + dimensions
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Set default aggregation functions if not provided
        if agg_functions is None:
            agg_functions = ['mean', 'count', 'sum']
        
        # Build aggregation dictionary
        if isinstance(agg_functions, str):
            agg_dict = {dim: [agg_functions] for dim in dimensions}
        elif isinstance(agg_functions, list):
            agg_dict = {dim: agg_functions for dim in dimensions}
        elif isinstance(agg_functions, dict):
            agg_dict = agg_functions
        else:
            raise ValueError("agg_functions must be str, list, or dict")
        
        # Group by grid coordinates and calculate statistics
        grouped_data = df.groupby([grid_x_col, grid_y_col]).agg(agg_dict)
        
        # Flatten multi-level column names
        grouped_data.columns = ['_'.join(col).strip() for col in grouped_data.columns.values]
        grouped_data = grouped_data.reset_index()
        
        # Rename grid columns for consistency
        grouped_data = grouped_data.rename(columns={grid_x_col: 'x', grid_y_col: 'y'})
        
        # Log summary statistics for the first dimension
        first_dim = dimensions[0]
        mean_col = f"{first_dim}_mean"
        if mean_col in grouped_data.columns:
            logger.info(f"Grid cells with data: {len(grouped_data)}")
            logger.info(f"{first_dim} value range: {grouped_data[mean_col].min():.2f} to "
                       f"{grouped_data[mean_col].max():.2f}")
            logger.info(f"Non-zero {first_dim} cells: {sum(grouped_data[mean_col] > 0)}")
        
        return grouped_data
    
    def merge_data_with_grid(self, 
                             cells_gdf: gpd.GeoDataFrame, 
                             aggregated_data: pd.DataFrame,
                             fill_value: float = 0) -> gpd.GeoDataFrame:
        """
        Merge aggregated data with grid cell geometries.
        
        Parameters:
        -----------
        cells_gdf : gpd.GeoDataFrame
            GeoDataFrame containing grid cell polygons with x, y columns.
        aggregated_data : pd.DataFrame
            Aggregated data from aggregate_data_by_grid method.
        fill_value : float, default=0
            Value to use for cells without data.
            
        Returns:
        --------
        gpd.GeoDataFrame
            Grid cells with aggregated values attached.
        """
        logger.info("Merging aggregated data with grid cells")
        
        # Validate merge keys
        merge_keys = ['x', 'y']
        for key in merge_keys:
            if key not in cells_gdf.columns:
                raise ValueError(f"Grid GeoDataFrame missing column: {key}")
            if key not in aggregated_data.columns:
                raise ValueError(f"Aggregated data missing column: {key}")
        
        # Merge data
        cells_with_data = cells_gdf.merge(
            aggregated_data,
            on=merge_keys,
            how='left'
        )
        
        # Fill NaN values for all numeric columns except geometry
        numeric_cols = cells_with_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in merge_keys:  # Don't fill x, y coordinates
                cells_with_data[col] = cells_with_data[col].fillna(fill_value)
        
        # Log merge statistics
        data_cols = [col for col in aggregated_data.columns if col not in merge_keys]
        if data_cols:
            count_col = next((col for col in data_cols if 'count' in col), None)
            if count_col:
                logger.info(f"Total grid cells: {len(cells_with_data)}")
                logger.info(f"Cells with data: {sum(cells_with_data[count_col] > 0)}")
        
        return cells_with_data
    
    def prepare_visualization_data(self, 
                                   cells_with_data: gpd.GeoDataFrame,
                                   display_column: str,
                                   target_crs: str = 'EPSG:3857',
                                   use_log_scale: bool = False,
                                   custom_transforms: Optional[Dict[str, callable]] = None) -> gpd.GeoDataFrame:
        """
        Prepare data for visualization by reprojecting and adding display columns.
        
        Parameters:
        -----------
        cells_with_data : gpd.GeoDataFrame
            Grid cells with aggregated data.
        display_column : str
            Primary column to use for visualization.
        target_crs : str, default='EPSG:3857'
            Target coordinate reference system for web mapping.
        use_log_scale : bool, default=False
            Whether to add log-scaled values for better visualization.
        custom_transforms : dict, optional
            Custom transformations to apply {new_column_name: transform_function}.
            
        Returns:
        --------
        gpd.GeoDataFrame
            Prepared data ready for visualization.
        """
        logger.info(f"Preparing data for visualization with CRS: {target_crs}")
        
        # Validate display column
        if display_column not in cells_with_data.columns:
            raise ValueError(f"Display column '{display_column}' not found in data")
        
        # Reproject to target CRS
        cells_transformed = cells_with_data.to_crs(target_crs)
        
        # Add display column
        cells_transformed['display_value'] = cells_transformed[display_column].copy()
        
        # Add log scale if requested
        if use_log_scale:
            cells_transformed[f'{display_column}_log'] = np.log10(
                cells_transformed[display_column] + 1
            )
            logger.info(f"Added log-scaled values for {display_column}")
        
        # Apply custom transformations
        if custom_transforms:
            for new_col, transform_func in custom_transforms.items():
                try:
                    cells_transformed[new_col] = transform_func(cells_transformed)
                    logger.info(f"Applied custom transformation: {new_col}")
                except Exception as e:
                    logger.warning(f"Failed to apply transformation {new_col}: {str(e)}")
        
        return cells_transformed
    
    def create_grid_polygons(self, 
                             cells_data: gpd.GeoDataFrame,
                             color_column: str,
                             colormap: str = 'viridis',
                             colorbar_title: str = 'Value',
                             alpha: float = 0.7,
                             hover_columns: Optional[List[str]] = None,
                             hover_formatters: Optional[Dict[str, str]] = None,
                             plot_width: int = 900,
                             plot_height: int = 700) -> gv.Polygons:
        """
        Create GeoViews Polygons with customizable styling.
        
        Parameters:
        -----------
        cells_data : gpd.GeoDataFrame
            Prepared grid cells in Web Mercator projection.
        color_column : str
            Column to use for color mapping.
        colormap : str, default='viridis'
            Colormap to use ('viridis', 'plasma', 'hot', 'coolwarm', etc.).
        colorbar_title : str, default='Value'
            Title for the colorbar.
        alpha : float, default=0.7
            Fill transparency (0-1).
        hover_columns : list, optional
            Columns to include in hover tooltips.
        hover_formatters : dict, optional
            Format strings for hover columns {column: format_string}.
        plot_width : int, default=900
            Width of the plot in pixels.
        plot_height : int, default=700
            Height of the plot in pixels.
            
        Returns:
        --------
        gv.Polygons
            Styled polygon layer for the grid.
        """
        logger.info(f"Creating grid polygons with {colormap} colormap")
        
        # Determine columns to include in visualization
        if hover_columns is None:
            # Include all numeric columns by default
            hover_columns = [col for col in cells_data.columns 
                           if cells_data[col].dtype in [np.float64, np.int64] 
                           and col not in ['geometry']]
        
        # Add x, y if not already included
        for coord in ['x', 'y']:
            if coord in cells_data.columns and coord not in hover_columns:
                hover_columns.append(coord)
        
        # Build hover tooltips
        hover_tooltips = []
        default_formatters = {
            'x': '@x',
            'y': '@y'
        }
        
        if hover_formatters is None:
            hover_formatters = {}
        
        for col in hover_columns:
            if col in default_formatters:
                hover_tooltips.append((col.upper(), default_formatters[col]))
            else:
                # Create readable label
                label = col.replace('_', ' ').title()
                # Determine format
                if col in hover_formatters:
                    format_str = hover_formatters[col]
                elif 'count' in col.lower():
                    format_str = f'@{col}'
                else:
                    format_str = f'@{col}{{0.00}}'
                hover_tooltips.append((label, format_str))
        
        # Create polygon layer
        grid_polygons = gv.Polygons(
            cells_data,
            vdims=hover_columns,
            crs=ccrs.GOOGLE_MERCATOR
        ).opts(
            color=color_column,
            cmap=colormap,
            colorbar=True,
            colorbar_opts={'title': colorbar_title},
            fill_alpha=alpha,
            line_color='white',
            line_width=0.5,
            tools=['hover'],
            hover_tooltips=hover_tooltips,
            width=plot_width,
            height=plot_height
        )
        
        return grid_polygons
    
    def create_base_layers(self, 
                           tile_source: str = 'OSM',
                           include_coastline: bool = True,
                           coastline_color: str = 'black',
                           coastline_width: int = 2) -> Tuple[Any, Optional[Any]]:
        """
        Create base map layers (tiles and optional coastline).
        
        Parameters:
        -----------
        tile_source : str, default='OSM'
            Tile source to use ('OSM', 'CartoLight', 'CartoDark', 'EsriImagery').
        include_coastline : bool, default=True
            Whether to include coastline feature.
        coastline_color : str, default='black'
            Color for coastline.
        coastline_width : int, default=2
            Width of coastline in pixels.
            
        Returns:
        --------
        tuple
            Tile layer and optional coastline feature layer.
        """
        logger.info(f"Creating base map layers with {tile_source} tiles")
        
        # Create tile layer based on source
        tile_sources = {
            'OSM': gvts.OSM,
            'CartoLight': gvts.CartoLight,
            'CartoDark': gvts.CartoDark,
            'EsriImagery': gvts.EsriImagery
        }
        
        if tile_source not in tile_sources:
            logger.warning(f"Unknown tile source {tile_source}, using OSM")
            tile_source = 'OSM'
        
        tiles = tile_sources[tile_source]()
        
        # Create coastline layer if requested
        coastline = None
        if include_coastline:
            coastline = gv.Feature(cfeature.COASTLINE, crs=ccrs.PlateCarree()).opts(
                line_color=coastline_color,
                line_width=coastline_width
            )
        
        return tiles, coastline
    
    def create_visualization(self,
                             df: pd.DataFrame,
                             cells_gdf: gpd.GeoDataFrame,
                             dimensions: Union[str, List[str]],
                             primary_dimension: Optional[str] = None,
                             agg_functions: Optional[Union[str, List[str], Dict[str, List[str]]]] = None,
                             title: str = "Grid Data Visualization",
                             colormap: str = 'viridis',
                             colorbar_title: Optional[str] = None,
                             width: int = 1000,
                             height: int = 800,
                             use_log_scale: bool = False,
                             tile_source: str = 'OSM',
                             **kwargs) -> Any:
        """
        Create the complete grid data visualization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing data to visualize.
        cells_gdf : gpd.GeoDataFrame
            Grid cell polygons.
        dimensions : str or list of str
            Dimension(s) to aggregate and visualize.
        primary_dimension : str, optional
            Primary dimension for color mapping. If None, uses first dimension + '_mean'.
        agg_functions : various, optional
            Aggregation functions to apply (see aggregate_data_by_grid).
        title : str, default="Grid Data Visualization"
            Plot title.
        colormap : str, default='viridis'
            Colormap for values.
        colorbar_title : str, optional
            Title for colorbar. If None, uses primary dimension name.
        width : int, default=1000
            Plot width in pixels.
        height : int, default=800
            Plot height in pixels.
        use_log_scale : bool, default=False
            Whether to use log scale for visualization.
        tile_source : str, default='OSM'
            Base map tile source.
        **kwargs : dict
            Additional parameters passed to component methods.
            
        Returns:
        --------
        Any
            Complete GeoViews plot object.
        """
        logger.info(f"Creating complete grid visualization for dimensions: {dimensions}")
        
        # Ensure dimensions is a list
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        
        # Step 1: Aggregate data
        aggregated_data = self.aggregate_data_by_grid(
            df, dimensions, agg_functions=agg_functions
        )
        
        # Step 2: Merge with grid cells
        cells_with_data = self.merge_data_with_grid(cells_gdf, aggregated_data)
        
        # Determine primary dimension for visualization
        if primary_dimension is None:
            primary_dimension = f"{dimensions[0]}_mean"
            if primary_dimension not in cells_with_data.columns:
                # Use first available aggregated column
                data_cols = [col for col in cells_with_data.columns 
                           if any(dim in col for dim in dimensions)]
                if data_cols:
                    primary_dimension = data_cols[0]
                else:
                    raise ValueError("No valid data columns found for visualization")
        
        # Step 3: Prepare visualization data
        cells_prepared = self.prepare_visualization_data(
            cells_with_data, 
            primary_dimension,
            use_log_scale=use_log_scale
        )
        
        # Determine colorbar title
        if colorbar_title is None:
            colorbar_title = primary_dimension.replace('_', ' ').title()
        
        # Step 4: Create layers
        grid_polygons = self.create_grid_polygons(
            cells_prepared,
            color_column=primary_dimension,
            colormap=colormap,
            colorbar_title=colorbar_title,
            plot_width=width - 100,  # Account for margins
            plot_height=height - 100
        )
        
        tiles, coastline = self.create_base_layers(tile_source=tile_source)
        
        # Step 5: Combine layers
        plot_layers = [tiles, grid_polygons]
        if coastline is not None:
            plot_layers.append(coastline)
        
        plot = plot_layers[0]
        for layer in plot_layers[1:]:
            plot = plot * layer
        
        # Step 6: Apply final styling
        final_plot = plot.opts(
            title=title,
            width=width,
            height=height
        )
        
        logger.info("Visualization created successfully")
        return final_plot
    
    def create_visualization_from_bounds(self,
                                       df: pd.DataFrame,
                                       bounds: Dict[str, List[float]],
                                       dimensions: Union[str, List[str]],
                                       grid_resolution: Optional[Dict[str, float]] = None,
                                       primary_dimension: Optional[str] = None,
                                       agg_functions: Optional[Union[str, List[str], Dict[str, List[str]]]] = None,
                                       title: str = "Grid Data Visualization",
                                       colormap: str = 'viridis',
                                       **kwargs) -> Any:
        """
        Create a complete grid visualization from coordinate bounds.
        
        This method handles the entire workflow from bounds to visualization:
        1. Creates boundary from bounds
        2. Generates grid cells
        3. Maps data points to grid cells
        4. Aggregates data by grid
        5. Creates visualization
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing data with lat/lon coordinates.
            Must have 'latitude' and 'longitude' columns.
        bounds : dict
            Dictionary with 'latitude' and 'longitude' keys, each containing [min, max]
        dimensions : str or list of str
            Dimension(s) to aggregate and visualize.
        grid_resolution : dict, optional
            Dictionary with 'x_distance' and 'y_distance' in meters.
            Default is 500m x 500m.
        primary_dimension : str, optional
            Primary dimension for color mapping.
        agg_functions : various, optional
            Aggregation functions to apply.
        title : str, default="Grid Data Visualization"
            Plot title.
        colormap : str, default='viridis'
            Colormap for values.
        **kwargs : dict
            Additional parameters passed to create_visualization.
            
        Returns:
        --------
        Any
            Complete GeoViews plot object.
        """
        logger.info("Creating visualization from bounds")
        
        # Validate input dataframe has coordinates
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("DataFrame must contain 'latitude' and 'longitude' columns")
        
        # Step 1: Create grid from bounds
        cells_gdf, utm_num = self.create_grid_from_bounds(bounds, grid_resolution)
        
        # Step 2: Convert data points to UTM projection
        logger.info("Converting data points to UTM projection")
        data_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="epsg:4326"
        )
        data_gdf_utm = data_gdf.to_crs(utm_num)
        
        # Step 3: Spatial join to assign points to grid cells
        logger.info("Assigning data points to grid cells")
        data_with_grid = gpd.sjoin(
            data_gdf_utm,
            cells_gdf,
            how='inner',
            predicate='within'
        )
        
        # Rename grid indices to match expected column names
        data_with_grid = data_with_grid.rename(columns={
            'x_right': 'grid_x',
            'y_right': 'grid_y'
        })
        
        # Drop the geometry column to convert back to regular DataFrame
        df_with_grid = pd.DataFrame(data_with_grid.drop(columns=['geometry', 'index_right']))
        
        logger.info(f"Points assigned to grid: {len(df_with_grid)} out of {len(df)}")
        
        # Step 4: Create visualization using the standard method
        return self.create_visualization(
            df_with_grid,
            cells_gdf,
            dimensions=dimensions,
            primary_dimension=primary_dimension,
            agg_functions=agg_functions,
            title=title,
            colormap=colormap,
            **kwargs
        )