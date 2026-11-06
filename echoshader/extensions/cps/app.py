"""Read CPS products for Echoshader's optional recent-transects tab."""

import sqlite3
from contextlib import ExitStack, closing
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import xarray as xr

from echoshader.app.data import bounded_preview, source_revision
from echoshader.app.navigation import navigation_group
from echoshader.app.refresh import file_revision, watch_changes
from echoshader.elements.curve import curve
from echoshader.elements.echogram import echogram, tricolor_echogram
from echoshader.elements.transect_map import transect_map as _transect_map


def _catalogue(database, count=None, completed_only=False):
    """Read small ledger tables; missing startup tables mean no data yet."""
    empty_transects = pd.DataFrame(
        columns=["transect_part", "transect_number", "start_time", "end_time", "status"]
    )
    empty_files = pd.DataFrame(
        columns=["cps_filename", "first_ping_time", "last_ping_time", "status"]
    )
    if not database.exists():
        return empty_transects, empty_files
    try:
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as con:
            query = "SELECT * FROM transects"
            if completed_only:
                query += " WHERE status='completed'"
            query += " ORDER BY end_time DESC"
            params = ()
            if count is not None:
                query += " LIMIT ?"
                params = (count,)
            transects = pd.read_sql_query(query, con, params=params)
            files = pd.read_sql_query(
                "SELECT * FROM sv_cps WHERE status='completed' ORDER BY first_ping_time",
                con,
            )
    except (sqlite3.OperationalError, pd.errors.DatabaseError) as exc:
        if "no such table" not in str(exc).lower():
            raise
        return empty_transects, empty_files
    for column in ("first_ping_time", "last_ping_time"):
        files[column] = pd.to_datetime(files[column], utc=True)
    return transects, files


def _file_revisions(root, names):
    revisions = []
    for name in names:
        try:
            revision = source_revision(root / "CPS_Sv" / name)
        except FileNotFoundError:
            revision = None
        revisions.append(revision)
    return tuple(revisions)


def _bottom_paths(root, names):
    return tuple(
        root / (Path(name).name.removesuffix("_CPS.zarr") + "_bottom_line.csv") for name in names
    )


def _bottom_samples(paths, data):
    """Match saved detector results to display pings; retain gaps as NaN."""
    frames = [pd.read_csv(path, usecols=["time", "depth"]) for path in paths if path.exists()]
    if not frames:
        return np.full(data.sizes["ping_time"], np.nan)
    bottom = pd.concat(frames, ignore_index=True)
    bottom["time"] = pd.to_datetime(bottom["time"], utc=True, errors="coerce")
    bottom["depth"] = pd.to_numeric(bottom["depth"], errors="coerce")
    bottom.loc[(bottom.depth < 0) | ~np.isfinite(bottom.depth), "depth"] = np.nan
    bottom = bottom.dropna(subset=["time"]).drop_duplicates("time", keep="last")
    # The CSV and CPS Zarr use the same detection ping timestamps. Exact matching
    # avoids extrapolating through missing detections or between separate files.
    times = pd.to_datetime(data.ping_time.values, utc=True)
    return bottom.set_index("time").depth.reindex(times).to_numpy(dtype=float)


def _combine(parts, start, end):
    """Assemble the requested interval consistently for previews and file view."""
    data = xr.concat(parts, dim="ping_time", data_vars="minimal", coords="minimal", compat="equals")
    data = data.sortby("ping_time").sel(
        ping_time=slice(start.tz_convert(None), end.tz_convert(None))
    )
    _, indices = np.unique(data.ping_time.values, return_index=True)
    return data.isel(ping_time=np.sort(indices))


def build_tabs(*, ds, root, config, max_pings, max_range_samples):
    root = Path(root).resolve()
    fixed_xlim = config.get("xlim")
    if fixed_xlim is not None:
        fixed_xlim = tuple(float(value) for value in fixed_xlim)
        if (
            len(fixed_xlim) != 2
            or not np.isfinite(fixed_xlim).all()
            or fixed_xlim[0] >= fixed_xlim[1]
        ):
            raise ValueError("xlim must contain two finite increasing longitudes")
    database = root / config.get("processing_db", "processing.db")
    bottom_root = root / config.get("seafloor_path", "CPS_Seafloor_CSVs")

    def read_catalogue():
        return _catalogue(database, int(config.get("count", 10)))

    def revision():
        transects, files = read_catalogue()
        return (
            transects.to_json(),
            files.to_json(),
            _file_revisions(root, tuple(files.cps_filename)),
            tuple(file_revision(path) for path in _bottom_paths(bottom_root, files.cps_filename)),
        )

    @lru_cache(maxsize=1)
    def available_fields(current_revision):
        """Read one completed file's metadata; do not load acoustic arrays."""
        _, files = read_catalogue()
        if files.empty:
            return [], {}
        with xr.open_zarr(root / "CPS_Sv" / files.iloc[-1].cps_filename) as data:
            requested = config.get("variables")
            variables = [
                name
                for name, array in data.data_vars.items()
                if {"channel", "ping_time"}.issubset(array.dims)
                and array.ndim == 3
                and (name in requested if requested is not None else name.startswith("Sv"))
            ]
            channels = {}
            for index, channel in enumerate(data.channel.values):
                name = str(channel)
                label = name
                if "frequency_nominal" in data:
                    frequency = float(data.frequency_nominal.isel(channel=index))
                    if np.isfinite(frequency):
                        label = f"{frequency / 1000:g} kHz · {name}"
                channels[label] = name
            return variables, channels

    @lru_cache(maxsize=10)
    def read_preview(selected_names, start, end, revisions, variable, channels):
        """Cache bounded previews by variable, channels, interval and source revision."""
        with ExitStack() as stack:
            parts = [
                stack.enter_context(xr.open_zarr(root / "CPS_Sv" / name)) for name in selected_names
            ]
            if not parts:
                raise ValueError("No completed Sv data available")
            names = (
                variable,
                "depth",
                "echo_range",
                "longitude",
                "latitude",
                "frequency_nominal",
            )
            parts = [part[[name for name in names if name in part]] for part in parts]
            data = _combine(parts, start, end)
            if not data.sizes.get("ping_time", 0):
                raise ValueError("No pings in this transect interval")
            if variable not in data:
                raise ValueError(f"{variable} is unavailable in this transect's CPS files")
            navigation = None
            if (
                "latitude" in data
                and "longitude" in data
                and data.latitude.dims == ("ping_time",)
                and data.longitude.dims == ("ping_time",)
            ):
                navigation = (
                    np.asarray(data.longitude.values, dtype=float),
                    np.asarray(data.latitude.values, dtype=float),
                )
            if "longitude" not in data or not np.isfinite(data.longitude.values).any():
                raise ValueError("No finite longitude values for this transect")
            data = data.sel(channel=list(dict.fromkeys(channels)))
            data = bounded_preview(data, variable, max_pings, max_range_samples).load()
        return data, navigation

    def tab(interval, clim):
        refresh = pn.widgets.Button(name="Refresh transects", button_type="default", width=150)
        notice = pn.pane.Alert(visible=False)
        failed = [False]
        preview_plots = {}
        navigation_hook = navigation_group()
        variable = pn.widgets.Select(label="Variable", options=[])
        mode = pn.widgets.Select(
            label="Display",
            options=["RGB", "Single channel"],
            value=config.get("mode", "RGB"),
        )
        channel = pn.widgets.Select(label="Channel", options=[], visible=False)
        rgb = [
            pn.widgets.Select(label=f"{color} channel", options=[])
            for color in ("Red", "Green", "Blue")
        ]
        show_bottom = pn.widgets.Checkbox(
            label="Show seafloor",
            value=bool(config.get("show_seafloor", True))
            and config.get("vert_dim", "depth") == "depth",
            disabled=config.get("vert_dim", "depth") != "depth",
        )
        syncing = [False]

        def sync_fields():
            try:
                variables, channels = available_fields(revision())
            except (
                ValueError,
                KeyError,
                OSError,
                sqlite3.Error,
                pd.errors.DatabaseError,
            ) as exc:
                notice.object = f"Waiting for CPS metadata; will retry: {exc}"
                notice.alert_type = "warning"
                notice.visible = True
                return
            syncing[0] = True
            try:
                previous = variable.value
                variable.options = variables
                variable.value = (
                    previous if previous in variables else (variables[0] if variables else None)
                )
                names = list(channels.values())
                for index, widget in enumerate([channel, *rgb]):
                    previous = widget.value
                    widget.options = channels
                    if previous in names:
                        widget.value = previous
                    elif names:
                        target = (18000, 38000, 70000)[max(0, index - 1)]
                        labels = list(channels)
                        frequencies = [
                            float(label.split(" kHz")[0]) * 1000 if " kHz" in label else np.nan
                            for label in labels
                        ]
                        chosen = (
                            int(np.nanargmin(np.abs(np.asarray(frequencies) - target)))
                            if np.isfinite(frequencies).any()
                            else min(max(0, index - 1), len(names) - 1)
                        )
                        widget.value = names[chosen]
                    else:
                        widget.value = None
            finally:
                syncing[0] = False

        def show_channels(event=None):
            channel.visible = mode.value == "Single channel"
            for widget in rgb:
                widget.visible = mode.value == "RGB"

        mode.param.watch(show_channels, "value")
        sync_fields()
        show_channels()

        @lru_cache(maxsize=1)
        def render_cached(
            limits,
            current_revision,
            variable_name,
            display_mode,
            channels,
            overlay_bottom,
        ):
            failed[0] = False
            transects, files = read_catalogue()
            if transects.empty:
                return pn.pane.Alert("Waiting for completed transects…", alert_type="warning")
            rows, plots, bounds = [], [], []
            tracks = []
            active = {str(value).zfill(3) for value in transects.transect_part}
            for number in list(preview_plots):
                if number not in active:
                    preview_plots.pop(number)
            for _, row in transects.iterrows():
                number = str(row.transect_part).zfill(3)
                start, end = (
                    pd.to_datetime(row.start_time, utc=True),
                    pd.to_datetime(row.end_time, utc=True),
                )
                selected = files[(files.last_ping_time >= start) & (files.first_ping_time <= end)]
                info = pn.pane.Markdown(f"**{number}**", width=48, margin=(4, 2, 0, 0))
                try:
                    selected_names = tuple(selected.cps_filename)
                    revisions = _file_revisions(root, selected_names)
                    key = (
                        selected_names,
                        start,
                        end,
                        revisions,
                        tuple(limits),
                        variable_name,
                        display_mode,
                        channels,
                    )
                    bottom_paths = _bottom_paths(bottom_root, selected_names)
                    key = (
                        *key,
                        overlay_bottom,
                        tuple(file_revision(path) for path in bottom_paths)
                        if overlay_bottom
                        else (),
                    )
                    cached = preview_plots.get(number)
                    if cached is None or cached[0] != key:
                        data, navigation = read_preview(
                            selected_names,
                            start,
                            end,
                            revisions,
                            variable_name,
                            channels,
                        )
                        bound = (
                            float(data.longitude.min()),
                            float(data.longitude.max()),
                        )
                        if fixed_xlim is not None and (
                            bound[1] < fixed_xlim[0] or bound[0] > fixed_xlim[1]
                        ):
                            raise ValueError(
                                f"Transect longitude {bound[0]:.4f} to {bound[1]:.4f} is outside "
                                f"xlim {fixed_xlim}. Remove xlim from app.yaml to show all transects."
                            )
                        options = dict(
                            var_name=variable_name,
                            x_dim="longitude",
                            vert_dim=config.get("vert_dim", "depth"),
                            rasterized=True,
                        )
                        if display_mode == "RGB":
                            plot = tricolor_echogram(
                                data,
                                channel=list(channels),
                                vmin=limits[0],
                                vmax=limits[1],
                                **options,
                            )
                        else:
                            plot = echogram(data, channel=channels[0], **options).opts(
                                clim=tuple(limits), cmap=config.get("cmap", "Viridis")
                            )
                        if overlay_bottom:
                            depths = _bottom_samples(bottom_paths, data)
                            if np.isfinite(depths).any():
                                plot = plot * curve(
                                    (data.longitude.values, depths),
                                    x="longitude",
                                    y="depth",
                                ).opts(
                                    color="black",
                                    line_width=2,
                                    tools=["hover"],
                                    show_legend=False,
                                )
                        plot = plot.opts(
                            responsive=True,
                            height=int(config.get("plot_height", 150)),
                            title="",
                            toolbar=None,
                            xaxis=None,
                            xlabel="",
                            ylabel="Depth (m)",
                            margin=(0, 0, 0, 0),
                            shared_axes=True,
                            hooks=[navigation_hook],
                        )
                        bound = (
                            float(data.longitude.min()),
                            float(data.longitude.max()),
                        )
                        pane = pn.pane.HoloViews(
                            plot,
                            linked_axes=True,
                            sizing_mode="stretch_width",
                            margin=0,
                        )
                        preview_plots[number] = (key, plot, bound, navigation, pane)
                    _, plot, bound, navigation, pane = preview_plots[number]
                    if navigation is not None:
                        tracks.append((number, *navigation))
                    plots.append(plot)
                    bounds.append(bound)
                    rows.append((info, pane))
                except (ValueError, KeyError, OSError) as exc:
                    failed[0] = True
                    rows.append((info, pn.pane.Alert(str(exc), alert_type="warning")))
            if bounds:
                lo, hi = min(b[0] for b in bounds), max(b[1] for b in bounds)
                pad = max((hi - lo) * 0.01, 1e-6)
                limits = fixed_xlim if fixed_xlim is not None else (lo - pad, hi + pad)
                for plot in plots:
                    plot.opts(xlim=limits)
            if plots:
                plots[-1].opts(xaxis="bottom", xlabel="Longitude")
            echograms = pn.Column(
                *[
                    pn.Row(
                        info,
                        (
                            pn.pane.HoloViews(
                                plot,
                                linked_axes=True,
                                sizing_mode="stretch_width",
                                margin=0,
                            )
                            if hasattr(plot, "opts")
                            else plot
                        ),
                        sizing_mode="stretch_width",
                        margin=0,
                    )
                    for info, plot in rows
                ],
                sizing_mode="stretch_width",
                margin=0,
                styles={"gap": "0px"},
            )

            return pn.Row(
                _transect_map(
                    tracks,
                    config,
                    [str(r.transect_part).zfill(3) for _, r in transects.iterrows()],
                ),
                echograms,
                sizing_mode="stretch_width",
                margin=0,
                styles={"gap": "16px", "align-items": "center"},
            )

        def render(
            limits,
            refresh_count,
            variable_name,
            display_mode,
            selected_channel,
            red,
            green,
            blue,
            overlay_bottom,
        ):
            if syncing[0]:
                return pn.Spacer(height=0)
            channels = (red, green, blue) if display_mode == "RGB" else (selected_channel,)
            if variable_name is None or any(value is None for value in channels):
                return pn.pane.Alert(
                    "Waiting for acoustic variables and channels…", alert_type="warning"
                )
            try:
                if failed[0]:
                    render_cached.cache_clear()
                return render_cached(
                    tuple(limits),
                    revision(),
                    variable_name,
                    display_mode,
                    channels,
                    overlay_bottom,
                )
            except (
                ValueError,
                KeyError,
                OSError,
                sqlite3.Error,
                pd.errors.DatabaseError,
            ) as exc:
                return pn.pane.Alert(
                    f"Waiting for CPS data; will retry: {exc}", alert_type="warning"
                )

        def changed(current):
            sync_fields()
            refresh.clicks += 1
            notice.visible = False
            return not failed[0]

        def error(exc):
            notice.object = (
                f"Could not check CPS updates; retaining the last view and retrying: {exc}"
            )
            notice.alert_type = "warning"
            notice.visible = True

        check = watch_changes(
            revision,
            changed,
            seconds=float(config.get("refresh_seconds", 60)),
            on_error=error,
        )
        content = pn.panel(
            pn.bind(
                render,
                clim,
                refresh.param.clicks,
                variable,
                mode,
                channel,
                *rgb,
                show_bottom,
            )
        )
        check()
        return pn.Column(
            pn.Row("**Transects**", refresh),
            pn.FlexBox(variable, mode, channel, *rgb, show_bottom),
            notice,
            content,
            sizing_mode="stretch_width",
            margin=0,
            styles={"gap": "0px"},
        )

    return None, [("Recent transects", tab)]


def build_files(*, root, processing_db="processing.db", seafloor_path="CPS_Seafloor_CSVs"):
    """Offer completed transects and CPS files without reading acoustic arrays."""
    root = Path(root).resolve()
    transects, files = _catalogue(root / processing_db, completed_only=True)

    bottom_root = root / seafloor_path

    def attach_bottom(data, names):
        data["seafloor_depth"] = (
            "ping_time",
            _bottom_samples(_bottom_paths(bottom_root, names), data),
        )
        data.seafloor_depth.attrs.update(units="m", long_name="Detected seafloor depth")
        return data

    def open_file(name):
        data = xr.open_zarr(root / "CPS_Sv" / name)
        try:
            return attach_bottom(data, [name])
        except Exception:
            data.close()
            raise

    def open_transect(row):
        start, end = pd.Timestamp(row.start_time), pd.Timestamp(row.end_time)
        first = pd.to_datetime(files.first_ping_time, utc=True)
        last = pd.to_datetime(files.last_ping_time, utc=True)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        names = files.loc[(first <= end) & (last >= start), "cps_filename"]
        parts = []
        try:
            parts = [xr.open_zarr(root / "CPS_Sv" / name) for name in names]
            if not parts:
                raise ValueError("No completed CPS files cover this transect")
            data = _combine(
                [part.drop_vars("source_filenames", errors="ignore") for part in parts],
                start,
                end,
            )
            attach_bottom(data, names)
            data.attrs = dict(data.attrs, transect_number=str(row.transect_number))
            data.set_close(lambda: [part.close() for part in parts])
            return data
        except Exception:
            for part in parts:
                part.close()
            raise

    entries = {}
    for row in transects.itertuples():
        label = f"Transect {str(row.transect_number).zfill(3)} · {row.start_time}"
        loader = lambda row=row: open_transect(row)
        first = pd.to_datetime(files.first_ping_time, utc=True)
        last = pd.to_datetime(files.last_ping_time, utc=True)
        start, end = (
            pd.to_datetime(row.start_time, utc=True),
            pd.to_datetime(row.end_time, utc=True),
        )
        names = tuple(files.loc[(first <= end) & (last >= start), "cps_filename"])
        loader.revision = (
            row.start_time,
            row.end_time,
            names,
            _file_revisions(root, names),
            tuple(file_revision(path) for path in _bottom_paths(bottom_root, names)),
        )
        entries[label] = loader
    if entries:
        entries = {"Latest completed transect": next(iter(entries.values())), **entries}
    for row in files.itertuples():
        loader = lambda name=row.cps_filename: open_file(name)
        loader.revision = (
            _file_revisions(root, (row.cps_filename,)),
            tuple(file_revision(path) for path in _bottom_paths(bottom_root, (row.cps_filename,))),
        )
        entries[f"CPS · {row.cps_filename}"] = loader
    return entries
