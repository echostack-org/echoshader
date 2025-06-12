import holoviews
import numpy

from ..utils import gram_opts


def get_box_stream(source_pic, bounds: tuple = None):
    """
    Get a Holoviews BoundsXY stream for a source plot with optional default bounds.

    This function creates a Holoviews BoundsXY stream for the specified `source_pic`,
    which is a Holoviews plot object. The `BoundsXY` stream allows users to interactively
    select a rectangular region on the plot by dragging a box with their mouse.

    Parameters
    ----------
    source_pic : holoviews.element
        The source plot for which the BoundsXY stream will be created.
    bounds : tuple, optional
        The initial default bounds of the selected rectangular region (box).
        If not provided, the default bounds are set to the left, bottom, right,
        and top corners of the `source_pic` plot.

    Returns
    -------
    holoviews.streams.BoundsXY
        A Holoviews BoundsXY stream associated with the `source_pic` plot.
        The stream allows users to interactively select a rectangular region (box)
        on the plot and get the selected bounds as an output.

    Examples
    --------
    # Assuming source_pic is a Holoviews plot object (e.g., hvplot.Image or hvplot.Scatter)
    box_stream = get_box_stream(source_pic)

    # Alternatively, provide custom default bounds
    # custom_bounds = (0, 0, 10, 10)  # (left, bottom, right, top)
    box_stream = get_box_stream(source_pic, bounds=custom_bounds)

    # The box_stream can be used to interactively select a rectangular region (box)
    # on the plot and access the selected bounds.
    # For example, you can access the bounds using:
    # bounds = box_stream.bounds
    # where bounds is a tuple containing the selected (left, bottom, right, top) coordinates.
    """
    box_stream = holoviews.streams.BoundsXY(
        source=source_pic,
        # define left, bottom, right, top corner of source plot as default bounds
        bounds=(
            (
                source_pic.lbrt[0],
                source_pic.lbrt[1],
                source_pic.lbrt[2],
                source_pic.lbrt[3],
            )
            if bounds is None
            else bounds
        ),
    )

    return box_stream


def get_lasso_stream(source_pic: holoviews.element, geometry: numpy.array = None):
    """
    Get a Holoviews Lasso stream for a source plot with optional default geometry.

    This function creates a Holoviews Lasso stream for the specified `source_pic`,
    which is a Holoviews plot object. The `Lasso` stream allows users to interactively
    draw free-form polygonal shapes (lasso) on the plot using their mouse.

    Parameters
    ----------
    source_pic : holoviews.element
        The source plot for which the Lasso stream will be created.
    geometry : numpy.array, optional
        The initial default geometry of the lasso shape as an Nx2 NumPy array.
        The array should contain a list of (x, y) coordinate pairs forming the
        vertices of the lasso shape. If not provided, a default lasso shape will
        be set using the left, bottom, right, and top corners of the `source_pic` plot.

    Returns
    -------
    holoviews.streams.Lasso
        A Holoviews Lasso stream associated with the `source_pic` plot.
        The stream allows users to interactively draw lasso shapes on the plot
        and get the drawn lasso geometry as an output.

    Examples
    --------
    # Assuming source_pic is a Holoviews plot object (e.g., hvplot.Image or hvplot.Scatter)
    # Create a Lasso stream with default geometry using the plot's corners
    lasso_stream = get_lasso_stream(source_pic)

    # Alternatively, provide custom default geometry
    custom_geometry = np.array([[0, 0], [5, 5], [10, 0]])
    # (x, y) coordinate pairs forming the lasso shape
    lasso_stream = get_lasso_stream(source_pic, geometry=custom_geometry)

    # The lasso_stream can be used to interactively draw lasso shapes on the plot
    # and access the drawn lasso geometry.
    """
    lasso_stream = holoviews.streams.Lasso(
        source=source_pic,
        geometry=(
            numpy.array(
                [
                    [
                        source_pic.lbrt[0],
                        source_pic.lbrt[0],
                        source_pic.lbrt[2],
                        source_pic.lbrt[2],
                    ],
                    [
                        source_pic.lbrt[1],
                        source_pic.lbrt[3],
                        source_pic.lbrt[1],
                        source_pic.lbrt[3],
                    ],
                ]
            )
            if geometry is None
            else geometry
        ),
    )

    return lasso_stream


def get_box_plot(box_stream: holoviews.streams):
    """
    Create a Holoviews DynamicMap for a box plot based on the given box_stream.

    This function generates a Holoviews DynamicMap representing a box plot based on the provided
    `box_stream`, which is a Holoviews BoundsXY stream. The box plot will dynamically update based
    on the user's interaction with the `box_stream`, allowing them to select a rectangular region
    on the plot.

    Parameters
    ----------
    box_stream : holoviews.streams.BoundsXY
        The Holoviews BoundsXY stream used to interactively select the box on the plot.

    Returns
    -------
    holoviews.core.DynamicMap
        A Holoviews DynamicMap representing the box plot. The box plot will be updated dynamically
        based on the user's interaction with the `box_stream`.

    Examples
    --------
    # Assuming box_stream is a Holoviews BoundsXY stream obtained from get_box_stream function
    # Create a DynamicMap for the box plot based on the box_stream
    box_plot = get_box_plot(box_stream)

    # Display the box plot using panel
    Panel.Row(box_plot)

    # When the user interacts with the box_stream (selecting a rectangular region on the plot),
    # the box plot will be dynamically updated to show the selected region as a box.
    """
    box_plot = holoviews.DynamicMap(
        lambda bounds: holoviews.Bounds(bounds).opts(gram_opts),
        streams=[box_stream],
    )

    return box_plot
