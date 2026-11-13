"""Shared Datashader aggregation for native meshes and recorded sample centres.

Transform display values only: scalar Sv is averaged in linear backscatter;
RGB averages sample-normalized colors.
"""

import logging
from time import perf_counter

import datashader as dsh
import holoviews as hv
import numpy as np
import xarray as xr
from holoviews.operation.datashader import rasterize

logger = logging.getLogger(__name__)


def _coordinates(plot):
    return tuple(plot.dimension_values(i, expanded=False, flat=False) for i in range(3))


def _map_values(plot, transform):
    """Preserve the input element."""
    name = plot.vdims[0].name
    if hasattr(plot.data, "assign"):
        return plot.clone(plot.data.assign(**{name: transform(plot.data[name])}))
    x, y, values = _coordinates(plot)
    return plot.clone((x, y, transform(values)))


def rasterize_echogram(
    plot, width=1000, height=300, dynamic=True, aggregation="linear_mean", **kwargs
):
    if aggregation not in ("mean", "linear_mean"):
        raise ValueError("aggregation must be linear_mean (Sv) or mean.")
    prepared = (
        _map_values(plot, lambda values: 10.0 ** (values / 10.0))
        if aggregation == "linear_mean"
        else plot
    )
    started = perf_counter()
    image = rasterize(
        prepared,
        aggregator=(dsh.mean() if isinstance(prepared, hv.Image) else dsh.mean(plot.vdims[0].name)),
        width=width,
        height=height,
        dynamic=dynamic,
        **kwargs,
    )
    stage = "graph setup" if dynamic else "raster computation"
    logger.info(
        "Datashader %s: variable=%s size=%dx%d time=%.2fs",
        stage,
        plot.vdims[0].name,
        width,
        height,
        perf_counter() - started,
    )
    if aggregation == "mean":
        return image

    def to_db(frame):
        with np.errstate(divide="ignore", invalid="ignore"):
            return _map_values(frame, lambda values: 10 * np.log10(values))

    return image.apply(to_db) if dynamic else to_db(image)


def rasterize_rgb(plots, vmin, vmax, width=1000, height=300, dynamic=True, **kwargs):
    if len(plots) != 3:
        raise ValueError("RGB requires exactly three input elements.")
    if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
        raise ValueError("RGB limits must be finite with vmax > vmin")
    rasters = []
    total_started = perf_counter()
    for plot in plots:
        colors = _map_values(plot, lambda values: ((values - vmin) / (vmax - vmin)).clip(0, 1))
        started = perf_counter()
        raster = rasterize(
            colors,
            aggregator=(
                dsh.mean() if isinstance(colors, hv.Image) else dsh.mean(plot.vdims[0].name)
            ),
            width=width,
            height=height,
            dynamic=dynamic,
            **kwargs,
        )
        stage = "graph setup" if dynamic else "raster computation"
        logger.info(
            "Datashader RGB %s: channel=%s variable=%s size=%dx%d time=%.2fs",
            stage,
            getattr(plot, "label", "") or "unknown",
            plot.vdims[0].name,
            width,
            height,
            perf_counter() - started,
        )
        rasters.append(raster)

    def combine(overlay):
        images = list(overlay.values())
        x, y, _ = _coordinates(images[0])
        colors = [_coordinates(image)[2] for image in images]
        valid = [np.isfinite(color) for color in colors]
        alpha = np.logical_or.reduce(valid).astype(float)
        rgb = [np.where(mask, color, 0.0) for color, mask in zip(colors, valid)]
        horizontal, vertical = [dim.name for dim in images[0].kdims]
        data = xr.Dataset(
            {name: ((vertical, horizontal), values) for name, values in zip("RGBA", [*rgb, alpha])},
            coords={horizontal: x, vertical: y},
        )
        return hv.RGB(data, kdims=images[0].kdims, vdims=["R", "G", "B", "A"])

    overlay = rasters[0] * rasters[1] * rasters[2]
    result = overlay.apply(combine, per_element=False) if dynamic else combine(overlay)
    logger.info(
        "Datashader RGB %s total: size=%dx%d time=%.2fs",
        "graph setup" if dynamic else "rasterization",
        width,
        height,
        perf_counter() - total_started,
    )
    return result
