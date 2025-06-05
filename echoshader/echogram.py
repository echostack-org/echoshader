from typing import Dict, List, Optional, Union

import holoviews
import numpy
import xarray

from .utils import gram_opts


def single_echogram(
    MVBS_ds: xarray,
    channel: str,
    cmap: Union[str, List[str]],
    value_range: tuple[float, float],
    vert_dim: Optional[str] = "echo_range",
):
    """
    Generate an echogram for a single frequency channel.

    This function takes an xarray.Dataset containing MVBS (Multibeam Backscatter) data,
    extracts the data for a specific frequency channel, and generates an echogram for
    that channel using Holoviews. The echogram is a visual representation of the
    backscatter values (Sv) over time (ping_time) and depth (echo_range).

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        xarray.Dataset containing MVBS data.
    channel : str
        The name of the frequency channel for which the echogram will be generated.
        It should be a valid channel name present in the 'channel' dimension of MVBS_ds.
    cmap : str or List[str]
        The colormap(s) to use for the echogram. It can be a single colormap name or
        a list of colormap names for each frequency channel (if multiple colormaps are used).
        Input list like ['#0000ff', '#00ffff'] to customize colormap.
    value_range : tuple[float, float]
        The minimum and maximum value for the color scale of the echogram.
    vert_dim : str, optional
        The name of the vertical dimension, must be 1D.

    Returns
    -------
    holoviews.element.Image
        An echogram for the specified frequency channel, displaying the backscatter values (Sv)
        over time (ping_time) and depth (echo_range). The echogram is rendered using Holoviews
        with the provided colormap and color scale limits.

    Examples
    --------
    # Assuming MVBS_ds is an xarray.Dataset containing MVBS data
    # Generate an echogram for the channel 'GPT 38 kHz 00907208dd13 5-1 OOI.38|200'
    echogram = echogram_single_frequency(
        MVBS_ds,
        channel='GPT 38 kHz 00907208dd13 5-1 OOI.38|200',
        cmap='jet',
        value_range=(-80,-30)
    )

    # Display the echogram using Panel
    Panel.Row(echogram)
    """
    gram_opts["Image"]["cmap"] = cmap

    gram_opts["Image"]["clim"] = value_range

    gram_opts["Image"]["title"] = channel

    if ~gram_opts["Image"]["invert_yaxis"]:
        gram_opts["Image"]["invert_yaxis"] = True

    echogram = (
        holoviews.Dataset(MVBS_ds.sel(channel=channel))
        .to(holoviews.Image, vdims=["Sv"], kdims=["ping_time", vert_dim])
        .opts(gram_opts)
    )

    return echogram


def convert_to_color(
    MVBS_ds: xarray, channel_sel: str, th_bottom: float, th_top: float
):
    """
    Convert backscatter data to a color array based on threshold values.

    This function takes an xarray.Dataset containing MVBS (Multibeam Backscatter) data,
    extracts the data for a specific `channel_sel`, and converts the backscatter values (Sv)
    to a color array based on specified threshold values. Values above `th_top` and below
    `th_bottom` are masked (NaN), and the remaining values are scaled
    to a range between 0 and 1, representing colors from minimum to maximum.

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        xarray.Dataset containing MVBS data.
    channel_sel : str
        The name of the frequency channel for which the color array will be generated.
        It should be a valid channel name present in the 'channel' dimension of MVBS_ds.
    th_bottom : float
        The lower threshold value for backscatter data.
    th_top : float
        The upper threshold value for backscatter data.

    Returns
    -------
    numpy.ndarray
        A color array representing backscatter data of the specified `channel_sel`.
        Values are scaled between 0 and 1, with NaN values for backscatter data below `th_bottom`.

    Examples
    --------
    # Assuming MVBS_ds is an xarray.Dataset containing MVBS data
    # Convert backscatter data of 'GPT 38 kHz 00907208dd13 5-1 OOI.38|200' to a color array
    color_array = convert_to_color(
        MVBS_ds,
        channel_sel='GPT 38 kHz 00907208dd13 5-1 OOI.38|200',
        th_bottom=-80.0,
        th_top=-40.0
    )
    """
    da_color = MVBS_ds.sel(channel=channel_sel)
    da_color = da_color.where(
        da_color <= th_top, other=th_top
    )  # set to ceiling at the top
    da_color = da_color.where(
        da_color >= th_bottom, other=th_bottom
    )  # threshold at the bottom
    da_color = da_color.expand_dims("channel")
    da_color = (da_color - th_bottom) / (th_top - th_bottom)
    da_color = numpy.squeeze(da_color.Sv.data).transpose()
    return da_color


def tricolor_echogram(
    MVBS_ds: xarray,
    vmin: float,
    vmax: float,
    rgb_map: Dict[str, str] = {},
    vert_dim: Optional[str] = "echo_range",
):
    """
    Create a tricolor echogram for multiple frequency channels.

    This function generates a tricolor echogram from an xarray.Dataset containing MVBS
    (Multibeam Backscatter) data, where each color channel represents a different frequency
    channel's backscatter values. The function allows custom mapping of frequency channels to
    RGB color channels using the `rgb_map` dictionary.

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        xarray.Dataset containing MVBS data.
    vmin : float
        The minimum value for the color scale of the echogram.
    vmax : float
        The maximum value for the color scale of the echogram.
    rgb_map : Dict[str, str], optional
        A dictionary specifying the mapping of frequency channels to RGB color channels.
        The keys are the frequency channel names, and the values are the corresponding
        RGB channel names. If not provided, the function will assign the first three frequency
        channels to the "R", "G", and "B" channels, respectively.
    vert_dim : str, optional
        Name of vertical dimension. Default is echo_range.

    Returns
    -------
    holoviews.element.RGB
        A tricolor echogram, where each color channel represents the backscatter values (Sv) of
        a different frequency channel. The echogram is rendered using Holoviews with the provided
        colormap and color scale limits.

    Examples
    --------
    # Assuming MVBS_ds is an xarray.Dataset containing MVBS data
    # Create a tricolor echogram for the first three frequency channels
    tricolor_plot = tricolor_echogram(MVBS_ds, vmin=-80.0, vmax=-40.0)

    # Alternatively, provide a custom mapping of frequency channels to RGB channels
    rgb_mapping = {
        'GPT 38 kHz 00907208dd13 5-1 OOI.38|200': 'R',
        'GPT 50 kHz 00907208dd13 5-1 OOI.50|200': 'G',
        'GPT 200 kHz 00907208dd13 5-1 OOI.200|200': 'B',
    }
    tricolor_plot = tricolor_echogram(MVBS_ds, vmin=-80.0, vmax=-40.0, rgb_map=rgb_mapping)

    # Display the tricolor echogram using Panel
    Panel.Row(tricolor_plot)
    """

    if ~gram_opts["RGB"]["invert_yaxis"]:
        gram_opts["RGB"]["invert_yaxis"] = True

    if rgb_map == {}:
        rgb_map[MVBS_ds.channel.values[0]] = "R"
        rgb_map[MVBS_ds.channel.values[1]] = "G"
        rgb_map[MVBS_ds.channel.values[2]] = "B"

    rgb_ch = {"R": None, "G": None, "B": None}

    for ch, color in rgb_map.items():
        rgb_ch[color] = convert_to_color(
            MVBS_ds, channel_sel=ch, th_bottom=vmin, th_top=vmax
        )

    rgb = holoviews.RGB(
        (
            MVBS_ds.ping_time.data,
            MVBS_ds[vert_dim].data,
            rgb_ch["R"],
            rgb_ch["G"],
            rgb_ch["B"],
        )
    ).opts(gram_opts)

    return rgb
