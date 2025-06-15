from typing import List, Union

import numpy
import plotly.graph_objects as go
import pyvista
import xarray


def curtain_plot_plotly(
    MVBS_ds: xarray.Dataset,
    cmap: Union[str, List[str]] = "jet",
    clim: tuple = None,
    ratio: float = 0.001,
    width: int = 800,
    height: int = 600,
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
    depth_levels = numpy.arange(nsamples) * ratio
    x_grid, z_grid = numpy.meshgrid(lon, depth_levels)
    y_grid, _ = numpy.meshgrid(lat, depth_levels)

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
        z=numpy.zeros_like(lon),
        mode="lines",
        line=dict(color="white", width=4),
        name="Vessel Path",
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
            camera=dict(eye=dict(x=0.5, y=-2, z=0.5), up=dict(x=0, y=0, z=1)),
            aspectmode="manual",
            aspectratio=dict(x=2, y=1, z=0.5),
        ),
        margin=dict(r=20, l=10, b=10, t=10),
    )

    return fig


def curtain_plot_pyvista(
    MVBS_ds: xarray.Dataset,
    cmap: Union[str, List[str]] = "jet",
    clim: tuple = None,
    ratio: float = 0.001,
):
    """
    Create and display a 2D curtain plot from a given xarray dataset.


    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        A dataset containing the data for the curtain plot.
    cmap : str or List[str], optional
        Colormap(s) to use for the curtain plot. Default is 'jet'.
    clim : tuple, optional
        Color limits (min, max) for the colormap. Default is None, which automatically
        determines the limits based on data.
    ratio : float, optional
        The Z spacing (interval) between adjacent slices of the curtain plot. Default is 0.001.
    Returns
    -------
    pyvista.Plotter
        The 2D curtain plot as a PyVista Plotter object.
    Notes
    -----
    This function creates a 2D curtain plot from the given dataset `MVBS_ds`, and the depth
    (echo_range) information is draped along the given latitude and longitude coordinates.
    The `MVBS_ds` dataset should contain a variable named 'Sv' representing the sonar data.
    The latitude and longitude coordinates must be present for each trace in the dataset.
    Example
    -------
        curtain = curtain_plot(MVBS_ds, cmap='jet', clim=(-70, -30), ratio=0.01)
        curtain_panel = panel.panel(
            curtain.ren_win,
            height=600,
            width=400,
            orientation_widget=True,
        )
    """

    data = MVBS_ds.Sv.values[1:].T

    lon = MVBS_ds.longitude.values[1:]
    lat = MVBS_ds.latitude.values[1:]
    path = numpy.array([lon, lat, numpy.full(len(lon), 0)]).T

    assert len(path) in data.shape, "Make sure coordinates are present for every trace."

    # Grab the number of samples (in Z dir) and number of traces/soundings
    nsamples, ntraces = data.shape

    # Define the Z spacing of your 2D section
    z_spacing = ratio

    # Create structured points draping down from the path
    points = numpy.repeat(path, nsamples, axis=0)
    # repeat the Z locations across
    tp = numpy.arange(0, z_spacing * nsamples, z_spacing)
    tp = path[:, 2][:, None] - tp
    points[:, -1] = tp.ravel()

    grid = pyvista.StructuredGrid()
    grid.points = points
    grid.dimensions = nsamples, ntraces, 1

    # Add the data array - note the ordering!
    grid["values"] = data.ravel(order="F")

    pyvista.global_theme.background = "gray"

    curtain = pyvista.Plotter()
    curtain.add_mesh(grid, cmap=cmap, clim=clim)
    curtain.add_mesh(pyvista.PolyData(path), color="white")

    curtain.show_grid()
    curtain.show_axes()

    curtain.view_xy()

    return curtain
