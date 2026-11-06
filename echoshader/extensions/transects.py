"""Generic file-based transect overview, independent of CPS processing state."""

from pathlib import Path

import panel as pn
import xarray as xr

from ..app.data import bounded_preview, source_revision
from ..app.navigation import attach_navigation, navigation_group
from ..app.refresh import watch_changes
from ..elements.echogram import tricolor_echogram
from ..elements.transect_map import transect_map


def build_tabs(*, ds, root, config, max_pings, max_range_samples):
    root = Path(root).resolve()

    def overview(*_):
        content = pn.Column(sizing_mode="stretch_width")
        notice = pn.pane.Alert(visible=False)
        plots = {}
        navigation = navigation_group()
        variable = config.get("variable", "Sv")
        vertical = config.get("vert_dim", "echo_range")
        axis = config.get("x_dim", "ping_time")

        def snapshot():
            values = config.get("files") or sorted(root.glob(config.get("pattern", "*.zarr")))
            paths = [Path(value) if Path(value).is_absolute() else root / value for value in values]
            return tuple((path, source_revision(path)) for path in paths)

        def update(revision):
            if not revision:
                content.objects = [
                    pn.pane.Alert("Waiting for transect files…", alert_type="warning")
                ]
                return
            rows, tracks = [], []
            active = {path for path, _ in revision}
            for path in list(plots):
                if path not in active:
                    plots.pop(path)
            for path, stamp in revision:
                if path not in plots or plots[path][0] != stamp:
                    with (
                        xr.open_zarr(path)
                        if path.suffix == ".zarr"
                        else xr.open_dataset(path) as data
                    ):
                        number = str(data.attrs.get("transect_number", path.stem))
                        track = (
                            (number, data.longitude.values, data.latitude.values)
                            if {"longitude", "latitude"}.issubset(data)
                            else None
                        )
                        channels = config.get("channels", list(data.channel.values[:3]))
                        names = list(
                            dict.fromkeys(
                                [variable, vertical, *([axis] if axis != "ping_time" else [])]
                            )
                        )
                        preview = bounded_preview(
                            data[names].sel(channel=channels),
                            variable,
                            max_pings,
                            max_range_samples,
                        ).load()
                    plot = tricolor_echogram(
                        preview,
                        channel=channels,
                        vmin=config.get("vmin", -100),
                        vmax=config.get("vmax", -30),
                        var_name=variable,
                        x_dim=axis,
                        vert_dim=vertical,
                        rasterized=True,
                    ).opts(responsive=True, height=150, toolbar=None)
                    plots[path] = (
                        stamp,
                        pn.Row(
                            pn.pane.Markdown(f"**{number}**", width=55),
                            pn.pane.HoloViews(
                                attach_navigation(plot, navigation),
                                linked_axes=True,
                                sizing_mode="stretch_width",
                            ),
                            sizing_mode="stretch_width",
                            margin=0,
                        ),
                        track,
                    )
                if plots[path][2] is not None:
                    tracks.append(plots[path][2])
                rows.append(plots[path][1])
            content.objects = [
                pn.Row(
                    transect_map(tracks, config, [t[0] for t in tracks]),
                    pn.Column(*rows, sizing_mode="stretch_width"),
                    sizing_mode="stretch_width",
                )
            ]
            notice.visible = False

        def error(exc):
            notice.object = (
                f"Could not update transects; retaining the last view and retrying: {exc}"
            )
            notice.alert_type = "warning"
            notice.visible = True

        check = watch_changes(
            snapshot, update, seconds=float(config.get("refresh_seconds", 60)), on_error=error
        )
        check()
        return pn.Column(notice, content, sizing_mode="stretch_width")

    return None, [("Transects", overview)]
