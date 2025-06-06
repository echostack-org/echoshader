from typing import Dict, List, Optional, Tuple, Union

import holoviews
import numpy as np
import xarray

from .utils import gram_opts


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
    da_color = np.squeeze(da_color.Sv.data).transpose()
    return da_color


def tricolor_echogram(
    MVBS_ds: xarray,
    vmin: float,
    vmax: float,
    rgb_map: Dict[str, str] = {},
    vert_dim: Optional[str] = "echo_range",
):

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


# def single_echogram(MVBS_ds: xarray,channel: str,cmap: Union[str, List[str]],value_range: tuple[float, float],vert_dim: Optional[str] = "echo_range",):

#     gram_opts["Image"]["cmap"] = cmap
#     gram_opts["Image"]["clim"] = value_range
#     gram_opts["Image"]["title"] = channel

#     if ~gram_opts["Image"]["invert_yaxis"]:
#         gram_opts["Image"]["invert_yaxis"] = True

#     echogram = (
#         holoviews.Dataset(MVBS_ds.sel(channel=channel))
#         .to(holoviews.Image, vdims=["Sv"], kdims=["ping_time", vert_dim])
#         .opts(gram_opts)
#     )

#     return echogram


def create_echogram(
    MVBS_ds: xarray.Dataset,
    channels: Union[str, List[str]] = None,
    cmap: Union[str, List[str]] = "viridis",
    value_range: tuple[float, float] = None,
    vert_dim: str = "echo_range",
    mode: str = "auto",
    **kwargs,
):
    # Normalize inputs
    if channels is None:
        channels = list(MVBS_ds.channel.values)
    elif isinstance(channels, str):
        channels = [channels]

    # Ensure colormap is a list matching channels
    if isinstance(cmap, str):
        cmaps = [cmap] * len(channels)
    elif isinstance(cmap, list):
        cmaps = cmap + [cmap[-1]] * (len(channels) - len(cmap))  # Extend if needed
    else:
        cmaps = ["viridis"] * len(channels)

    # Auto-detect mode if needed
    if mode == "auto":
        mode = (
            "single"
            if len(channels) == 1
            else "rgb" if len(channels) == 3 else "layout"
        )

    # Validate mode
    if mode == "rgb" and len(channels) != 3:
        raise ValueError(f"RGB mode requires exactly 3 channels, got {len(channels)}")

    # Set up common options
    if mode == "rgb":
        opts_dict = dict(gram_opts.get("RGB", {}))
    else:
        opts_dict = dict(gram_opts.get("Image", {}))

    if "invert_yaxis" not in opts_dict:
        opts_dict["invert_yaxis"] = True

    # Create visualization based on mode
    if mode in ["single", "layout"]:
        # Create individual echograms
        echograms = []
        for i, (channel, channel_cmap) in enumerate(zip(channels, cmaps)):
            opts_dict["clim"] = value_range
            opts_dict["cmap"] = channel_cmap
            opts_dict["title"] = channel

            echogram = (
                holoviews.Dataset(MVBS_ds.sel(channel=channel))
                .to(holoviews.Image, vdims=["Sv"], kdims=["ping_time", vert_dim])
                .opts(**opts_dict)
            )

            echograms.append(echogram)

            if mode == "single":
                break  # Only use first channel

        # Return single or layout
        if mode == "single":
            return echograms[0]
        else:
            cols = kwargs.get("layout_cols", 1)
            return holoviews.Layout(echograms).cols(cols)

    elif mode == "rgb":
        # Get RGB mapping
        rgb_mapping = kwargs.get(
            "rgb_mapping", {channels[0]: "R", channels[1]: "G", channels[2]: "B"}
        )

        # Create normalized color arrays for each channel
        rgb_arrays = {}
        for channel, color in rgb_mapping.items():
            # Extract and normalize data
            data = MVBS_ds.sel(channel=channel).Sv.values
            data = np.clip(data, value_range[0], value_range[1])
            data = (data - value_range[0]) / (value_range[1] - value_range[0])
            rgb_arrays[color] = data.T

        return holoviews.RGB(
            (
                MVBS_ds.ping_time.data,
                MVBS_ds[vert_dim].data,
                rgb_arrays["R"],
                rgb_arrays["G"],
                rgb_arrays["B"],
            )
        ).opts(**opts_dict)
    else:
        raise ValueError(f"Unknown mode: {mode}")
