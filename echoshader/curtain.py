from typing import List, Union
import numpy as np
import plotly.graph_objects as go
import xarray as xr


def curtain_plot(
        MVBS_ds: xr.Dataset,
        cmap: Union[str, List[str]] = "jet",
        clim: tuple = None,
        ratio: float = 0.001,
        width: int = 800,
        height: int = 600
) -> go.Figure:
    """
    Create and display a 3D curtain plot using Plotly.

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        Dataset containing Sv data and coordinates
    cmap : Union[str, List[str]], optional
        Colormap name or list of colors. Default is "jet"
    clim : tuple, optional
        Color limits (min, max). Default is data range
    ratio : float, optional
        Depth spacing between samples. Default is 0.001
    width : int, optional
        Figure width. Default is 800
    height : int, optional
        Figure height. Default is 600

    Returns
    -------
    go.Figure
        Plotly Figure object containing the curtain plot
    """
    # Extract and prepare data
    data = MVBS_ds.Sv.values[1:].T  # Match original data handling
    lon = MVBS_ds.longitude.values[1:]
    lat = MVBS_ds.latitude.values[1:]

    nsamples, ntraces = data.shape

    # Create coordinate grids
    depth_levels = np.arange(nsamples) * ratio
    x_grid, z_grid = np.meshgrid(lon, depth_levels)
    y_grid, _ = np.meshgrid(lat, depth_levels)

    # Create colormap
    if isinstance(cmap, list):
        colorscale = [[i / (len(cmap) - 1), color] for i, color in enumerate(cmap)]
    else:
        colorscale = cmap

    # Create 3D surface plot
    surface = go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        surfacecolor=data,
        colorscale=colorscale,
        cmin=clim[0] if clim else data.min(),
        cmax=clim[1] if clim else data.max(),
        colorbar=dict(title="Sv (dB)"),
        showscale=True,
    )

    # Create path line
    path_line = go.Scatter3d(
        x=lon,
        y=lat,
        z=np.zeros_like(lon),
        mode="lines",
        line=dict(color="white", width=4),
        name="Vessel Path"
    )

    # Create figure
    fig = go.Figure(data=[surface, path_line])

    # Configure layout
    fig.update_layout(
        width=width,
        height=height,
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Depth (m)",
            zaxis=dict(autorange="reversed"),
            camera=dict(
                eye=dict(x=0.5, y=-2, z=0.5),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode="manual",
            aspectratio=dict(x=2, y=1, z=0.5)
        ),
        margin=dict(r=20, l=10, b=10, t=10)
    )

    return fig