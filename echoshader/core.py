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


@xarray.register_dataset_accessor("eshader")
class Echoshader:
    """Echoshader xarray Dataset accessor."""

    def __init__(self, xarray_obj: xarray.Dataset):
        self._obj = xarray_obj
        self._check_input()

    def _check_input(self):
        """Validate the minimum dataset structure required by Echoshader."""
        if "Sv" not in self._obj.variables:
            raise ValueError("Dataset must contain a variable named 'Sv'.")

        expected_dims = (
            "channel",
            "ping_time",
            ("range_sample", "depth", "echo_range"),
        )
        actual_dims = self._obj["Sv"].dims

        if len(actual_dims) != 3:
            raise ValueError("'Sv' must have exactly 3 dimensions.")

        for i, (actual, expected) in enumerate(zip(actual_dims, expected_dims)):
            if isinstance(expected, tuple):
                if actual not in expected:
                    raise ValueError(
                        f"'Sv' dimension at index {i} "
                        f"must be one of {expected}, "
                        f"but got '{actual}'."
                    )
            elif actual != expected:
                raise ValueError(
                    f"'Sv' dimension at index {i} must be '{expected}', but got '{actual}'."
                )

    def echogram(
        self,
        channel: str | list[str] | None = None,
        vert_dim: str = "echo_range",
    ):
        """Create single- or multi-channel echograms."""
        return echogram_element(
            self._obj,
            channel=channel,
            vert_dim=vert_dim,
        )

    def tricolor_echogram(
        self,
        channel: list[str],
        vmin: float,
        vmax: float,
        vert_dim: str = "echo_range",
    ):
        """Create a tricolor RGB echogram."""
        return tricolor_echogram_element(
            self._obj,
            channel=channel,
            vmin=vmin,
            vmax=vmax,
            vert_dim=vert_dim,
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
