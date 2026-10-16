import numpy
import plotly.graph_objects as go
import pyvista
import xarray


def curtain(
    MVBS_ds: xarray.Dataset,
    channel: str,
    ratio: float = 0.001,
    engine: str = "plotly",
    cmap: str | list[str] = "jet",
    clim: tuple[float, float] | None = None,
):
    """
    Create a 3-D curtain visualization.

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        Dataset containing Sv and geographic coordinates.
    channel : str
        Acoustic channel to plot.
    ratio : float, optional
        Vertical spacing between samples. Default is 0.001.
    engine : {"plotly", "pyvista"}, optional
        Rendering backend. Default is ``"plotly"``.
    cmap : str or list[str], optional
        Colormap used by the selected backend. Default is ``"jet"``.
    clim : tuple[float, float], optional
        Color limits. If None, limits are inferred from the data.

    Returns
    -------
    plotly.graph_objects.Figure or pyvista.Plotter
        Curtain visualization produced by the selected backend.
    """
    data = MVBS_ds.sel(channel=channel)

    if engine == "plotly":
        return _curtain_plotly(
            MVBS_ds=data,
            cmap=cmap,
            clim=clim,
            ratio=ratio,
        )

    if engine == "pyvista":
        return _curtain_pyvista(
            MVBS_ds=data,
            cmap=cmap,
            clim=clim,
            ratio=ratio,
        )

    raise ValueError(f"Unsupported backend: {engine!r}. Expected 'plotly' or 'pyvista'.")


def _curtain_plotly(
    MVBS_ds: xarray.Dataset,
    cmap: str | list[str] = "jet",
    clim: tuple[float, float] | None = None,
    ratio: float = 0.001,
) -> go.Figure:
    """Create a 3-D curtain plot using Plotly."""

    data = MVBS_ds.Sv.values[1:].T
    lon = MVBS_ds.longitude.values[1:]
    lat = MVBS_ds.latitude.values[1:]

    nsamples, _ = data.shape

    depth_levels = numpy.arange(nsamples) * ratio

    x_grid, z_grid = numpy.meshgrid(
        lon,
        depth_levels,
    )

    y_grid, _ = numpy.meshgrid(
        lat,
        depth_levels,
    )

    if isinstance(cmap, list):
        colorscale = [[i / (len(cmap) - 1), color] for i, color in enumerate(cmap)]
    else:
        colorscale = cmap

    surface = go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        surfacecolor=data,
        colorscale=colorscale,
        cmin=clim[0] if clim else data.min(),
        cmax=clim[1] if clim else data.max(),
        colorbar={"title": "Sv (dB)"},
        showscale=True,
    )

    path_line = go.Scatter3d(
        x=lon,
        y=lat,
        z=numpy.zeros_like(lon),
        mode="lines",
        line={
            "color": "white",
            "width": 4,
        },
        name="Vessel Path",
    )

    figure = go.Figure(
        data=[
            surface,
            path_line,
        ]
    )

    figure.update_layout(
        scene={
            "xaxis_title": "Longitude",
            "yaxis_title": "Latitude",
            "zaxis_title": "Depth (m)",
            "zaxis": {
                "autorange": "reversed",
            },
            "camera": {
                "eye": {
                    "x": 0.5,
                    "y": -2,
                    "z": 0.5,
                },
                "up": {
                    "x": 0,
                    "y": 0,
                    "z": 1,
                },
            },
            "aspectmode": "manual",
            "aspectratio": {
                "x": 2,
                "y": 1,
                "z": 0.5,
            },
        },
        margin={
            "r": 20,
            "l": 10,
            "b": 10,
            "t": 10,
        },
    )

    return figure


def _curtain_pyvista(
    MVBS_ds: xarray.Dataset,
    cmap: str | list[str] = "jet",
    clim: tuple[float, float] | None = None,
    ratio: float = 0.001,
) -> pyvista.Plotter:
    """Create a 3-D curtain plot using PyVista."""

    data = MVBS_ds.Sv.values[1:].T

    lon = MVBS_ds.longitude.values[1:]
    lat = MVBS_ds.latitude.values[1:]

    path = numpy.array(
        [
            lon,
            lat,
            numpy.zeros(len(lon)),
        ]
    ).T

    if len(path) not in data.shape:
        raise ValueError("Coordinates must be present for every trace.")

    nsamples, ntraces = data.shape

    points = numpy.repeat(
        path,
        nsamples,
        axis=0,
    )

    z_positions = numpy.arange(
        0,
        ratio * nsamples,
        ratio,
    )

    z_positions = path[:, 2][:, None] - z_positions

    points[:, -1] = z_positions.ravel()

    grid = pyvista.StructuredGrid()

    grid.points = points
    grid.dimensions = (
        nsamples,
        ntraces,
        1,
    )

    grid["values"] = data.ravel(order="F")

    plotter = pyvista.Plotter()

    plotter.add_mesh(
        grid,
        cmap=cmap,
        clim=clim,
    )

    plotter.add_mesh(
        pyvista.PolyData(path),
        color="white",
    )

    plotter.show_grid()
    plotter.show_axes()
    plotter.view_xy()

    return plotter
