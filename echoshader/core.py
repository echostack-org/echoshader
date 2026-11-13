import xarray

from .elements.curtain import curtain as curtain_element
from .elements.echogram import (
    echogram as echogram_element,
)
from .elements.echogram import (
    tricolor_echogram as tricolor_echogram_element,
)
from .elements.hist import hist as hist_element
from .elements.table import table as table_element
from .elements.track import track as track_element
from .validation import validate_dataset


@xarray.register_dataset_accessor("eshader")
class Echoshader:
    """Echoshader xarray Dataset accessor."""

    def __init__(self, xarray_obj: xarray.Dataset):
        self._obj = xarray_obj
        self.validation = validate_dataset(xarray_obj)

    def echogram(
        self,
        channel: str | list[str] | None = None,
        vert_dim: str = "echo_range",
        **kwargs,
    ):
        """Create single- or multi-channel echograms."""
        return echogram_element(
            self._obj,
            channel=channel,
            vert_dim=vert_dim,
            **kwargs,
        )

    def tricolor_echogram(
        self,
        channel: list[str],
        vmin: float,
        vmax: float,
        vert_dim: str = "echo_range",
        **kwargs,
    ):
        """Create a tricolor RGB echogram."""

        return tricolor_echogram_element(
            self._obj,
            channel=channel,
            vmin=vmin,
            vmax=vmax,
            vert_dim=vert_dim,
            **kwargs,
        )

    def track(
        self,
        tile: str | None = "OSM",
    ):
        """Create a vessel-track visualization."""
        return track_element(
            self._obj,
            tile=tile,
        )

    def curtain(
        self,
        channel: str,
        ratio: float = 0.001,
        engine: str = "plotly",
        cmap: str | list[str] = "jet",
        clim: tuple[float, float] | None = None,
    ):
        """Create a 3-D curtain visualization."""
        return curtain_element(
            self._obj,
            channel=channel,
            ratio=ratio,
            engine=engine,
            cmap=cmap,
            clim=clim,
        )

    def hist(
        self,
        bins: int = 24,
        overlay: bool = True,
    ):
        """Create histograms of acoustic backscatter."""
        return hist_element(
            self._obj,
            bins=bins,
            overlay=overlay,
        )

    def table(self):
        """Create a summary table for the acoustic dataset."""
        return table_element(self._obj)
