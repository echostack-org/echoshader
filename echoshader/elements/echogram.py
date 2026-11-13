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
    *,
    rasterized: bool = False,
    dynamic: bool = True,
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
            rasterized=rasterized,
            dynamic=dynamic,
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
            rasterized=rasterized,
            dynamic=dynamic,
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
    *,
    rasterized: bool = False,
    dynamic: bool = True,
):
    """Create an echogram for one acoustic channel."""
    data = ds[var_name].sel(channel=channel)
    vertical = ds[vert_dim]
    x_values = ds[x_dim].values
    x_regular = _is_regular(x_values)

    if "channel" in vertical.dims:
        vertical = vertical.sel(channel=channel)

    horizontal_dim = ds[x_dim].dims[0]
    sample_dim = next(dim for dim in data.dims if dim != horizontal_dim)

    if vertical.ndim == 1:
        valid = numpy.isfinite(vertical.values)
        data = data.transpose(sample_dim, horizontal_dim)

        if x_regular and not rasterized:
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
            plot = _datetime_safe_quadmesh(
                (
                    x_values,
                    vertical.values[valid],
                    data.values[valid, :],
                ),
                kdims=[x_dim, vert_dim],
                vdims=[var_name],
            )

        if rasterized:
            from .raster import rasterize_echogram

            plot = rasterize_echogram(plot, dynamic=dynamic)
            if numpy.issubdtype(ds[x_dim].dtype, numpy.datetime64):
                plot = plot.opts(hooks=[_format_time_axis])
        return plot.relabel(str(channel)).opts(invert_yaxis=True)

    vertical = vertical.transpose(
        sample_dim,
        horizontal_dim,
    )
    data = data.transpose(
        sample_dim,
        horizontal_dim,
    )

    reference = vertical.isel({horizontal_dim: 0}).values

    vertical_regular = numpy.allclose(
        vertical.values,
        reference[:, None],
        equal_nan=True,
    )

    if vertical_regular:
        valid = numpy.isfinite(reference)

        if x_regular and not rasterized:
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
            plot = _datetime_safe_quadmesh(
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

        plot = _datetime_safe_quadmesh(
            (
                x_grid,
                vertical.values,
                data.values,
            ),
            kdims=[x_dim, vert_dim],
            vdims=[var_name],
        )

    if rasterized:
        from .raster import rasterize_echogram

        plot = rasterize_echogram(plot, dynamic=dynamic)
        if numpy.issubdtype(ds[x_dim].dtype, numpy.datetime64):
            plot = plot.opts(hooks=[_format_time_axis])
    return plot.relabel(str(channel)).opts(invert_yaxis=True)


def tricolor_echogram(
    ds: xarray.Dataset,
    channel: list[str],
    vmin: float,
    vmax: float,
    var_name: str = "Sv",
    x_dim: str = "ping_time",
    vert_dim: str = "echo_range",
    *,
    rasterized: bool = False,
    dynamic: bool = True,
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
    from echoshader.validation import validate_dataset

    validation = validate_dataset(
        ds,
        tricolor_channels=channel,
        vert_dim=vert_dim,
        var_name=var_name,
        allow_variable_geometry=rasterized,
    )
    if not validation.tricolor_ready:
        raise ValueError(f"Tricolor is not available: {validation.tricolor_reason}")

    if rasterized:
        from .raster import rasterize_rgb

        mesh_ds = ds
        if numpy.issubdtype(ds[x_dim].dtype, numpy.datetime64):
            mesh_ds = ds.assign_coords(
                {x_dim: ds[x_dim].astype("datetime64[ns]").astype("int64") / 1e6}
            )
        meshes = [single_echogram(mesh_ds, ch, var_name, x_dim, vert_dim) for ch in channel]
        result = rasterize_rgb(meshes, vmin, vmax, dynamic=dynamic)
        if numpy.issubdtype(ds[x_dim].dtype, numpy.datetime64):
            result = result.opts(hooks=[_format_time_axis])
        return result.opts(invert_yaxis=True)

    # A native Sv array is indexed by range_sample, even when the requested
    # vertical coordinate is physical echo_range/depth.
    sample_dim = validation.vertical_dim
    vertical = ds[vert_dim]
    if "channel" in vertical.dims:
        vertical = vertical.sel(channel=channel[0], drop=True)
    if "ping_time" in vertical.dims:
        vertical = vertical.isel(ping_time=0, drop=True)
    axis = numpy.asarray(vertical.values)
    valid = numpy.isfinite(axis)
    if valid.sum() < 2:
        raise ValueError(f"No usable vertical coordinates for {vert_dim!r}.")

    rgb = [
        _convert_to_color(
            ds=ds,
            channel=ch,
            vmin=vmin,
            vmax=vmax,
            var_name=var_name,
            x_dim="ping_time",
            vert_dim=sample_dim,
        )[valid, :]
        for ch in channel
    ]

    if x_dim != "ping_time":
        return _tricolor_coordinate_mesh(ds, rgb, axis[valid], x_dim, vert_dim)

    return holoviews.RGB(
        (
            ds[x_dim].data,
            axis[valid],
            rgb[0],
            rgb[1],
            rgb[2],
        ),
        kdims=[x_dim, vert_dim],
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


def _tricolor_coordinate_mesh(ds, rgb, vertical, x_dim, vert_dim):
    """Use actual auxiliary coordinates, preserving ping order and RGB values."""
    x = ds[x_dim]
    if x.dims != ("ping_time",) or not numpy.isfinite(x.values).all():
        raise ValueError(f"{x_dim} must contain a finite coordinate for every ping")
    if x.size < 2 or numpy.ptp(x.values) == 0:
        raise ValueError(f"{x_dim} requires at least two distinct positions")
    colors = numpy.stack(rgb, axis=-1)
    valid = numpy.isfinite(colors)
    finite = valid.any(axis=-1).ravel()
    colors = numpy.where(valid, colors, 0.0)
    colors = numpy.rint(numpy.clip(colors, 0, 1) * 255).astype("uint8")
    hexcolors = numpy.array([f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors.reshape(-1, 3)])
    indices = numpy.arange(len(hexcolors)).reshape(colors.shape[:2])

    def apply_rgb(plot, element):
        # QuadMesh may choose quads or polygons. Its cell index identifies the
        # exact source sample in either case; do not sort longitude or average.
        source = plot.handles["source"]
        cell = numpy.asarray(source.data["rgb_index"], dtype=int)
        # Restore acquisition draw order if the backend reverses a descending axis.
        # This orders source cell indices, never longitude or acoustic values.
        order = numpy.argsort(cell % x.size, kind="stable")
        cell = cell[order]
        ordered = {key: numpy.asarray(value)[order] for key, value in source.data.items()}
        source.data = dict(ordered, rgb_color=hexcolors[cell], rgb_alpha=finite[cell].astype(float))
        renderer = plot.handles["glyph_renderer"]
        renderer.glyph.fill_color = {"field": "rgb_color"}
        renderer.glyph.fill_alpha = {"field": "rgb_alpha"}
        renderer.glyph.line_color = None

    return _datetime_safe_quadmesh(
        (x.values, vertical, indices), kdims=[x_dim, vert_dim], vdims=["rgb_index"]
    ).opts(invert_yaxis=True, colorbar=False, hooks=[apply_rgb])


def _datetime_safe_quadmesh(data, **kwargs):
    """Render datetime meshes numerically, retaining UTC tick labels."""
    x, y, values = data
    x = numpy.asarray(x)
    is_datetime = numpy.issubdtype(x.dtype, numpy.datetime64)

    if is_datetime:
        valid = ~numpy.isnat(x)
        x = x.astype("datetime64[ms]").astype("float64")
        x = numpy.where(valid, x, numpy.nan)

    mesh = holoviews.QuadMesh((x, y, values), **kwargs)

    if is_datetime:
        mesh = mesh.opts(hooks=[_format_time_axis])

    return mesh


def _format_time_axis(plot, element):
    from bokeh.models import DatetimeTicker

    for axis in plot.state.xaxis:
        axis.ticker = DatetimeTicker()
        axis.ticker = DatetimeTicker(desired_num_ticks=12)
        axis.axis_label = "Time (UTC)"
