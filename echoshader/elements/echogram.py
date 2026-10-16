import holoviews
import numpy
import xarray


def _is_regular(values):
    values = numpy.asarray(values)

    if values.size < 3:
        return True

    if numpy.issubdtype(values.dtype, numpy.datetime64):
        values = values.astype("datetime64[ns]").astype("int64")

    diffs = numpy.diff(values)

    return numpy.allclose(
        diffs,
        diffs[0],
        rtol=1e-3,
        atol=0,
    )


def echogram(
    ds: xarray.Dataset,
    channel: str | list[str] | None = None,
    var_name: str = "Sv",
    x_dim: str = "ping_time",
    vert_dim: str = "echo_range",
):
    """
    Create single- or multi-channel echograms.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable to visualize.
    channel : str or list[str]
        Channel or channels to plot.
    var_name : str, optional
        Name of the variable to visualize. Default is ``"Sv"``.
    x_dim : str, optional
        Name of the horizontal dimension. Default is ``"ping_time"``.
    vert_dim : str, optional
        Name of the vertical coordinate. Default is ``"echo_range"``.

    Returns
    -------
    holoviews.Image, holoviews.QuadMesh or holoviews.Layout
        A single echogram or a vertical layout of echograms.
    """
    if channel is None:
        channels = [str(value) for value in ds["channel"].values.tolist()]
    elif isinstance(channel, str):
        return single_echogram(
            ds=ds,
            channel=channel,
            var_name=var_name,
            x_dim=x_dim,
            vert_dim=vert_dim,
        )
    else:
        channels = channel

    echograms = [
        single_echogram(
            ds=ds,
            channel=ch,
            var_name=var_name,
            x_dim=x_dim,
            vert_dim=vert_dim,
        )
        for ch in channels
    ]

    return holoviews.Layout(echograms).cols(1)


def single_echogram(
    ds: xarray.Dataset,
    channel: str,
    var_name: str = "Sv",
    x_dim: str = "ping_time",
    vert_dim: str = "echo_range",
):
    """Create an echogram for one acoustic channel."""
    data = ds[var_name].sel(channel=channel)
    vertical = ds[vert_dim]
    x_values = ds[x_dim].values
    x_regular = _is_regular(x_values)

    if "channel" in vertical.dims:
        vertical = vertical.sel(channel=channel)

    sample_dim = next(dim for dim in data.dims if dim != x_dim)

    if vertical.ndim == 1:
        valid = numpy.isfinite(vertical.values)
        data = data.transpose(sample_dim, x_dim)

        if x_regular:
            plot = holoviews.Image(
                (
                    x_values,
                    vertical.values[valid],
                    data.values[valid, :],
                ),
                kdims=[x_dim, vert_dim],
                vdims=[var_name],
            )
        else:
            plot = holoviews.QuadMesh(
                (
                    x_values,
                    vertical.values[valid],
                    data.values[valid, :],
                ),
                kdims=[x_dim, vert_dim],
                vdims=[var_name],
            )

        return plot.relabel(str(channel)).opts(invert_yaxis=True)

    vertical = vertical.transpose(
        sample_dim,
        x_dim,
    )
    data = data.transpose(
        sample_dim,
        x_dim,
    )

    reference = vertical.isel({x_dim: 0}).values

    vertical_regular = numpy.allclose(
        vertical.values,
        reference[:, None],
        equal_nan=True,
    )

    if vertical_regular:
        valid = numpy.isfinite(reference)

        if x_regular:
            plot = holoviews.Image(
                (
                    x_values,
                    reference[valid],
                    data.values[valid, :],
                ),
                kdims=[x_dim, vert_dim],
                vdims=[var_name],
            )
        else:
            plot = holoviews.QuadMesh(
                (
                    x_values,
                    reference[valid],
                    data.values[valid, :],
                ),
                kdims=[x_dim, vert_dim],
                vdims=[var_name],
            )
    else:
        x_grid = numpy.broadcast_to(
            x_values[numpy.newaxis, :],
            vertical.shape,
        )

        plot = holoviews.QuadMesh(
            (
                x_grid,
                vertical.values,
                data.values,
            ),
            kdims=[x_dim, vert_dim],
            vdims=[var_name],
        )

    return plot.relabel(str(channel)).opts(invert_yaxis=True)


def tricolor_echogram(
    ds: xarray.Dataset,
    channel: list[str],
    vmin: float,
    vmax: float,
    var_name: str = "Sv",
    x_dim: str = "ping_time",
    vert_dim: str = "echo_range",
):
    """
    Create a tricolor RGB echogram.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable to visualize.
    channel : list[str]
        Exactly three channels used for the red, green, and blue components.
    vmin : float
        Minimum value used for normalization.
    vmax : float
        Maximum value used for normalization.
    var_name : str, optional
        Name of the variable to visualize. Default is ``"Sv"``.
    x_dim : str, optional
        Name of the horizontal dimension. Default is ``"ping_time"``.
    vert_dim : str, optional
        Name of the vertical dimension. Default is ``"echo_range"``.

    Returns
    -------
    holoviews.RGB
        Tricolor echogram.

    Notes
    -----
    ``vmin`` and ``vmax`` are data-normalization limits rather than
    ordinary display limits.
    """
    if len(channel) != 3:
        raise ValueError("Exactly 3 channels are required for a tricolor echogram.")

    rgb = [
        _convert_to_color(
            ds=ds,
            channel=ch,
            vmin=vmin,
            vmax=vmax,
            var_name=var_name,
            x_dim=x_dim,
            vert_dim=vert_dim,
        )
        for ch in channel
    ]

    return holoviews.RGB(
        (
            ds[x_dim].data,
            ds[vert_dim].data,
            rgb[0],
            rgb[1],
            rgb[2],
        )
    ).opts(
        invert_yaxis=True,
    )


def _convert_to_color(
    ds: xarray.Dataset,
    channel: str,
    vmin: float,
    vmax: float,
    var_name: str = "Sv",
    x_dim: str = "ping_time",
    vert_dim: str = "echo_range",
):
    """Normalize one channel of a variable to [0, 1] for RGB display."""
    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin.")

    data = ds[var_name].sel(
        channel=channel,
    )

    data = data.clip(
        min=vmin,
        max=vmax,
    )

    normalized = (data - vmin) / (vmax - vmin)

    return numpy.asarray(
        normalized.transpose(
            vert_dim,
            x_dim,
        ).data
    )
