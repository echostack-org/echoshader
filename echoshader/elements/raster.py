"""Small Datashader adapters around the existing HoloViews elements.

Scalar pixels average linear backscatter (then return dB). RGB pixels average
normalized display colors, with a common missing-data mask for all three colors.
These operations affect display only.
"""

import datashader as dsh
import holoviews as hv
import numpy as np
from holoviews.operation.datashader import rasterize


def _coordinates(plot):
    return (
        plot.dimension_values(0, expanded=False, flat=False),
        plot.dimension_values(1, expanded=False, flat=False),
        plot.dimension_values(2, flat=False),
    )


def rasterize_echogram(plot, width=1000, height=300, dynamic=True):
    x, y, values = _coordinates(plot)
    linear = plot.clone((x, y, 10.0 ** (values / 10.0)))
    image = rasterize(
        linear, aggregator=dsh.mean(), width=width, height=height, dynamic=dynamic
    )

    def to_db(im):
        x, y, values = _coordinates(im)
        with np.errstate(divide="ignore", invalid="ignore"):
            return im.clone((x, y, 10.0 * np.log10(values)))

    return image.apply(to_db) if dynamic else to_db(image)


def rasterize_rgb(meshes, vmin, vmax, width=1000, height=300, dynamic=True):
    """Reuse three scalar meshes, combine their raster images into a normal RGB."""
    if vmax <= vmin:
        raise ValueError("vmax must exceed vmin")
    coords = [_coordinates(mesh) for mesh in meshes]
    valid = np.logical_and.reduce([np.isfinite(c[2]) for c in coords])
    rasters = []
    for mesh, (x, y, values) in zip(meshes, coords):
        color = np.where(valid, np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0), np.nan)
        rasters.append(
            rasterize(
                mesh.clone((x, y, color)),
                aggregator=dsh.mean(),
                width=width,
                height=height,
                dynamic=dynamic,
            )
        )

    def combine(overlay):
        images = list(overlay.values())
        x, y, _ = _coordinates(images[0])
        colors = [_coordinates(im)[2] for im in images]
        alpha = np.logical_and.reduce([np.isfinite(c) for c in colors]).astype(float)
        return hv.RGB(
            (x, y, *[np.nan_to_num(c) for c in colors], alpha),
            kdims=images[0].kdims,
            vdims=["R", "G", "B", "A"],
        )

    overlay = rasters[0] * rasters[1] * rasters[2]
    return overlay.apply(combine, per_element=False) if dynamic else combine(overlay)
