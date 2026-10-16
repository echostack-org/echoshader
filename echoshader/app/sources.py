from pathlib import Path

import pandas
import panel as pn
import param
import xarray

from .context import AppContext


def load_zarr(
    path: str | Path,
    **kwargs,
):
    """Load a Zarr dataset."""
    return xarray.open_zarr(
        path,
        **kwargs,
    )


def load_netcdf(
    path: str | Path,
    **kwargs,
):
    """Load a NetCDF dataset."""
    return xarray.open_dataset(
        path,
        **kwargs,
    )


def load_csv(
    path: str | Path,
    **kwargs,
):
    """Load a CSV file."""
    return pandas.read_csv(
        path,
        **kwargs,
    )


SOURCES = {
    "zarr": load_zarr,
    "netcdf": load_netcdf,
    "csv": load_csv,
}


class DataSource(param.Parameterized):
    """Reactive data source for an Echoshader application."""

    data = param.Parameter(
        default=None,
        allow_None=True,
    )

    def __init__(
        self,
        loader,
        context: AppContext,
        path=None,
        args=None,
        refresh=None,
        exports=None,
        **params,
    ):
        super().__init__(**params)

        self.loader = loader
        self.context = context

        self.path_template = path
        self.args = args or {}

        self.refresh_interval = refresh
        self.exports = exports or {}

        self._callback = None

    def _resolve_path(self):
        """Resolve the configured path against app context."""

        if self.path_template is None:
            return None

        return self.context.resolve(self.path_template)

    def _export_context(self):
        """Export configured values from loaded data to app context."""

        if self.data is None:
            return

        values = {}

        for name, export_config in self.exports.items():
            # export dataset attribute
            if "attr" in export_config:
                attr_name = export_config["attr"]

                if not hasattr(
                    self.data,
                    "attrs",
                ):
                    continue

                value = self.data.attrs.get(
                    attr_name,
                )

                if value is not None:
                    values[name] = value

            # export dataset coordinate
            elif "coord" in export_config:
                coord_name = export_config["coord"]

                if not isinstance(
                    self.data,
                    xarray.Dataset,
                ):
                    continue

                if coord_name not in self.data.coords:
                    continue

                value = self.data[coord_name].values

                if value.size == 1:
                    value = value.item()

                values[name] = value

        if values:
            self.context.update(
                **values,
            )

    def reload(self):
        """Reload the source."""

        path = self._resolve_path()

        if path is None:
            new_data = self.loader(
                **self.args,
            )

        else:
            new_data = self.loader(
                path,
                **self.args,
            )

        self.data = new_data

        self._export_context()

    def start_refresh(self):
        """Start periodic source refresh."""

        if self.refresh_interval is None:
            return

        if self.refresh_interval <= 0:
            raise ValueError("Source refresh interval must be greater than zero.")

        self._callback = pn.state.add_periodic_callback(
            self.reload,
            period=int(self.refresh_interval * 1000),
        )


def build_source(
    source_config: dict,
    context: AppContext,
):
    """Build one configured data source."""

    source_type = source_config["type"]

    if source_type not in SOURCES:
        raise ValueError(f"Unknown data source type: {source_type!r}.")

    return DataSource(
        loader=SOURCES[source_type],
        context=context,
        path=source_config.get(
            "path",
        ),
        args=source_config.get(
            "args",
            {},
        ),
        refresh=source_config.get(
            "refresh",
        ),
        exports=source_config.get(
            "exports",
            {},
        ),
    )
