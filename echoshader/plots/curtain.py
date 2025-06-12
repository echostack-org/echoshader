from typing import List, Union

import numpy
import pyvista
import xarray


def curtain_plot(
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
