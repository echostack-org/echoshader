"""Reusable, read-only HoloViz application shell for acoustic datasets.

Uses Panel for layout/reactivity, HoloViews for selections and overlays,
Echoshader for acoustic elements, and xarray for lazily sliced Zarr data.
This module does NOT import Echodataflow, Prefect, or a YAML builder.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import echopype.colormap  # noqa: F401 -- registers Echopype colormaps
import geoviews as gv
import holoviews as hv
import matplotlib as mpl
import numpy as np
import pandas as pd
import panel as pn
import param
import xarray as xr
from bokeh.models import (
    CustomJSTickFormatter,
    DatetimeTicker,
    DatetimeTickFormatter,
)

from echoshader.elements.curve import curve
from echoshader.elements.echogram import echogram, tricolor_echogram
from echoshader.elements.track import track as track_element
from echoshader.validation import validate_dataset

from .data import AcousticReadCache, SourceChangedError
from .lazy import lazy_view
from .navigation import attach_navigation, navigation_group

hv.extension("bokeh")
pn.extension("tabulator")


def _format_app_time_axis(plot, element):
    for axis in plot.state.xaxis:
        axis.ticker = DatetimeTicker(desired_num_ticks=10)
        axis.formatter = DatetimeTickFormatter(
            seconds="%H:%M:%S",
            minsec="%H:%M:%S",
            minutes="%H:%M:%S",
        )
        axis.major_label_orientation = np.pi / 4
        axis.axis_label = "Time (UTC)"


class AppliedSettings(param.Parameterized):
    interval = param.Tuple()
    clim = param.Tuple()
    cmap = param.String()
    variable = param.String()
    rgb_variable = param.String()
    rgb_red = param.String()
    rgb_green = param.String()
    rgb_blue = param.String()


@dataclass
class AcousticSource:
    data: xr.Dataset
    name: str = "Acoustic survey"
    path: Path | None = None
    bottom: pd.DataFrame | None = None
    track: pd.DataFrame | None = None
    max_pings: int = 500
    max_range_samples: int = 1200
    map_width: int = 650
    map_height: int = 400
    variables: tuple[str, ...] = ("Sv",)
    rasterized: bool = True
    cache_mb: int = 256
    initial_tab: str = "Multi-frequency"


def _with_seafloor(plot, data, *, bottom=None, rasterized=True):
    """Overlay stored depth detections; preserve missing samples as gaps."""
    times = pd.to_datetime(data.ping_time.values, utc=True)
    if "seafloor_depth" in data:
        depths = np.asarray(data.seafloor_depth.values, dtype=float)
    elif bottom is not None and {"time", "depth"}.issubset(bottom):
        values = bottom.copy()
        values["time"] = pd.to_datetime(values["time"], utc=True)
        values = values.drop_duplicates("time", keep="last").set_index("time")
        depths = pd.to_numeric(values.depth, errors="coerce").reindex(times).to_numpy(dtype=float)
    else:
        return plot
    if not np.isfinite(depths).any():
        return plot
    # Datashader uses milliseconds since epoch; explicitly normalize ns so this
    # also works with pandas versions whose default datetime unit is microseconds.
    x = times.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    if rasterized:
        x = x.astype("int64") / 1e6
    return plot * curve((x, depths), x="ping_time", y="depth").opts(
        color="black", line_width=2, tools=["hover"], show_legend=False
    )


def _channel_for_frequency(ds, freq):
    names = [str(v) for v in ds.channel.values]
    if "frequency_nominal" not in ds:
        return names[0]
    freqs = np.asarray(ds.frequency_nominal.values, dtype=float).reshape(-1)
    finite = np.flatnonzero(np.isfinite(freqs))
    return names[int(finite[np.argmin(abs(freqs[finite] - freq))])] if len(finite) else names[0]


def _parse_track(source: AcousticSource):
    """Return metadata track if time, latitude, longitude are available."""
    if source.track is not None:
        table = source.track.copy()
    elif {"ping_time", "longitude", "latitude"}.issubset(source.data.variables):
        # Track coordinates only; no acoustic array computed.
        table = pd.DataFrame(
            {
                k: np.asarray(source.data[k].values).reshape(-1)
                for k in ("ping_time", "longitude", "latitude")
            }
        )
    else:
        return None
    required = {"ping_time", "longitude", "latitude"}
    if not required.issubset(table):
        return None
    table = table[list(required)].copy()
    table["ping_time"] = pd.to_datetime(
        table["ping_time"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    for col in ("longitude", "latitude"):
        table[col] = pd.to_numeric(table[col], errors="coerce")
    return table.dropna().sort_values("ping_time").reset_index(drop=True)


def build_acoustic_app(
    source: AcousticSource,
    *,
    extra_tabs: list[tuple[str, object]] | None = None,
    title: str = "Echoshader",
):
    """Build generic interactive app; optional tabs are composed by caller."""

    longitude_formatter = CustomJSTickFormatter(
        code="""
            return (tick / 6378137 * 180 / Math.PI).toFixed(2) + "°";
        """
    )

    latitude_formatter = CustomJSTickFormatter(
        code="""
            const lat = (2 * Math.atan(Math.exp(tick / 6378137)) - Math.PI / 2);
            return (lat * 180 / Math.PI).toFixed(2) + "°";
        """
    )

    ds = source.data
    if not source.variables or isinstance(source.variables, str):
        raise ValueError("variables must be a non-empty sequence of acoustic variable names")
    variables = tuple(source.variables)
    if len(set(variables)) != len(variables):
        raise ValueError("Acoustic variable names must be unique")
    dataset_validation = validate_dataset(ds, var_name=variables[0])
    for variable in variables[1:]:
        other = validate_dataset(ds, var_name=variable)
        if other.vertical_dim != dataset_validation.vertical_dim:
            raise ValueError("Displayed acoustic variables must share the same vertical dimension")
    has_bottom = source.bottom is not None or "seafloor_depth" in ds
    n = ds.sizes["ping_time"]
    if not n:
        raise ValueError("No pings available")
    max_pings = min(max(1, int(source.max_pings)), n)
    times = ds.ping_time.values
    initial = (max(0, n - max_pings), n)
    channels = [str(ch) for ch in ds.channel.values]
    freqs = (
        np.asarray(ds.frequency_nominal.values, dtype=float) if "frequency_nominal" in ds else None
    )
    # Missing nominal frequencies must not collapse all channels into the first.
    ordered = [channels[i] for i in np.argsort(freqs)] if freqs is not None else channels
    available = (
        {ch: float(freq) / 1000 for ch, freq in zip(channels, freqs) if np.isfinite(freq)}
        if freqs is not None
        else {}
    )

    # native Panel controls: throttled updates avoid re-rendering while dragging
    timeline = pn.widgets.IntRangeSlider(
        label="Ping interval (entire survey)",
        start=0,
        end=n,
        value=initial,
        step=1,
        sizing_mode="stretch_width",
    )
    color = pn.widgets.RangeSlider(
        label="Sv color range (dB)",
        start=-120,
        end=-20,
        value=(-100, -30),
        step=1,
        sizing_mode="stretch_width",
    )
    cmap = pn.widgets.Select(
        label="Colormap",
        options=[
            "ep.ek500",
            "viridis",
            "jet",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "Greys",
        ],
        value="ep.ek500",
        width=165,
    )
    variable_widget = pn.widgets.Select(
        label="Acoustic variable",
        options=list(variables),
        value=variables[0],
        width=220,
    )
    rgb_variable_widget = pn.widgets.Select(
        label="RGB variable",
        options=list(variables),
        value=variables[0],
        width=220,
    )
    rgb_widgets = [
        pn.widgets.Select(
            label=f"{label} channel",
            options=channels,
            value=(
                _channel_for_frequency(ds, freq)
                if freqs is not None and np.isfinite(freqs).any()
                else channels[min(i, len(channels) - 1)]
            ),
        )
        for i, (label, freq) in enumerate(zip(("Red", "Green", "Blue"), (120000, 38000, 18000)))
    ]

    def bounded(window):
        a, b = map(int, window)
        a = min(max(0, a), n - 1)
        b = min(n, max(a + 1, b))
        if b - a > max_pings:
            b = min(n, a + max_pings)
        return a, b

    # single reactive selection shared by all plots, widgets and the map
    # slider updates are throttled; map actions update state immediately
    class WindowState(param.Parameterized):
        window = param.Tuple(default=initial)

    selection = WindowState()

    # Draft controls do not redraw plots until one of these buttons is clicked.
    applied = AppliedSettings(
        interval=tuple(selection.window),
        clim=tuple(color.value),
        cmap=cmap.value,
        variable=variable_widget.value,
        rgb_variable=rgb_variable_widget.value,
        rgb_red=rgb_widgets[0].value,
        rgb_green=rgb_widgets[1].value,
        rgb_blue=rgb_widgets[2].value,
    )
    apply_multifrequency = pn.widgets.Button(name="Apply", button_type="primary", disabled=True)
    apply_tricolor = pn.widgets.Button(name="Apply", button_type="primary", disabled=True)
    multi_spinner = pn.indicators.LoadingSpinner(value=False, width=24, height=24)
    rgb_spinner = pn.indicators.LoadingSpinner(value=False, width=24, height=24)
    apply_buttons = (apply_multifrequency, apply_tricolor)

    def enable_single_apply(*events):
        for button in apply_buttons:
            button.disabled = False

    for widget in [timeline, color, cmap, variable_widget, rgb_variable_widget, *rgb_widgets]:
        widget.param.watch(enable_single_apply, "value")
    selection.param.watch(enable_single_apply, "window")

    def apply_single_settings(event):
        applied.param.update(
            interval=tuple(selection.window),
            clim=tuple(color.value),
            cmap=cmap.value,
            variable=variable_widget.value,
            rgb_variable=rgb_variable_widget.value,
            rgb_red=rgb_widgets[0].value,
            rgb_green=rgb_widgets[1].value,
            rgb_blue=rgb_widgets[2].value,
        )
        for button in apply_buttons:
            button.disabled = True

    apply_multifrequency.on_click(apply_single_settings)
    apply_tricolor.on_click(apply_single_settings)

    time_index = pd.DatetimeIndex(times)

    start_input = pn.widgets.DatetimeInput(
        label="Start time (UTC)",
        value=time_index[initial[0]].to_pydatetime(),
        start=time_index[0].to_pydatetime(),
        end=time_index[-1].to_pydatetime(),
        width=240,
    )

    end_input = pn.widgets.DatetimeInput(
        label="End time (UTC)",
        value=time_index[initial[1] - 1].to_pydatetime(),
        start=time_index[0].to_pydatetime(),
        end=time_index[-1].to_pydatetime(),
        width=240,
    )

    # keep the displayed slider interval within the configured limit
    updating = False

    def enforce_window(event):
        nonlocal updating
        if updating:
            return

        a, b = map(int, event.new)
        a = min(max(0, a), n - 1)
        b = min(n, max(a + 1, b))

        if b - a > max_pings:
            previous = tuple(map(int, event.old or initial))
            if a != previous[0] and b == previous[1]:
                b = min(n, a + max_pings)
            else:
                a = max(0, b - max_pings)

        value = (a, b)
        if value != tuple(timeline.value):
            updating = True
            try:
                timeline.value = value
            finally:
                updating = False

    timeline.param.watch(enforce_window, "value")

    def move_selection(value):
        """Synchronize the slider before publishing the shared interval."""
        nonlocal updating
        value = bounded(value)

        updating = True
        try:
            if tuple(timeline.value) != value:
                timeline.value = value
        finally:
            updating = False

        if tuple(selection.window) != value:
            selection.window = value

    def slider_committed(event):
        # Commit the corrected interval displayed by the slider.
        move_selection(tuple(timeline.value))

    timeline.param.watch(slider_committed, "value_throttled")

    syncing_dates = False

    def update_from_dates(event):
        if syncing_dates or start_input.value is None or end_input.value is None:
            return

        start = pd.Timestamp(start_input.value)
        end = pd.Timestamp(end_input.value)

        if start > end:
            return

        a = int(time_index.searchsorted(start, side="left"))
        b = int(time_index.searchsorted(end, side="right"))

        a = min(max(a, 0), n - 1)
        b = min(max(b, a + 1), n)

        move_selection((a, b))

    start_input.param.watch(update_from_dates, "value")
    end_input.param.watch(update_from_dates, "value")

    def sync_date_inputs(event):
        nonlocal syncing_dates
        a, b = bounded(event.new)

        syncing_dates = True
        try:
            start_input.value = time_index[a].to_pydatetime()
            end_input.value = time_index[b - 1].to_pydatetime()
        finally:
            syncing_dates = False

    selection.param.watch(sync_date_inputs, "window")

    reads = AcousticReadCache(ds, source.path, max_bytes=source.cache_mb * 1024**2)

    def window_dataset(bounds, variable, selected_channels):
        a, b = bounded(bounds)
        vertical = next(
            (
                v
                for v in (("depth", "echo_range") if has_bottom else ("echo_range", "depth"))
                if v in ds
            ),
            dataset_validation.vertical_dim,
        )
        dim = dataset_validation.vertical_dim
        step = (
            max(1, int(np.ceil(ds.sizes[dim] / source.max_range_samples)))
            if source.max_range_samples > 0
            else 1
        )
        parts = [
            reads.read(variable, ch, a, b, vertical, step).expand_dims(channel=[ch])
            for ch in selected_channels
        ]
        # Auxiliary coordinates (e.g. range_sample on a depth grid) can vary
        # by channel. Preserve them along channel instead of overriding them.
        window = xr.concat(
            parts, dim="channel", data_vars="all", coords="different", compat="equals"
        )
        return window

    def plot_xrange(bounds):
        bounds = xrange(bounds)
        return tuple(pd.Timestamp(v).value / 1e6 for v in bounds) if source.rasterized else bounds

    def xrange(bounds):
        a, b = bounded(bounds)
        start, end = (
            pd.Timestamp(times[a]).to_pydatetime(),
            pd.Timestamp(times[b - 1]).to_pydatetime(),
        )
        return (start, end if end > start else start + dt.timedelta(seconds=1))

    def echogram_view(bounds, clim, palette, *, variable, channel, label):
        try:
            w = window_dataset(tuple(bounds), variable, [channel])
        except SourceChangedError as exc:
            return pn.pane.Alert(str(exc), alert_type="warning")
        if variable not in w:
            return pn.pane.Alert(f"{variable} is not available", alert_type="info")
        vert_dim = next(
            (
                name
                for name in (
                    ("depth",)
                    if dataset_validation.vertical_dim == "depth" or (has_bottom and "depth" in w)
                    else ("echo_range", "depth")
                )
                if name in w
            ),
            dataset_validation.vertical_dim,
        )

        palette = mpl.colormaps["ep.ek500"] if palette == "ep.ek500" else palette

        # reuse the standalone element, which handles regular and irregular grids
        plot = echogram(
            w,
            channel=channel,
            var_name=variable,
            vert_dim=vert_dim,
            rasterized=source.rasterized,
            dynamic=False,
        ).opts(
            hooks=[_format_app_time_axis],
            responsive=True,
            height=300,
            cmap=palette,
            clim=tuple(clim),
            xlim=plot_xrange(bounds),
            invert_yaxis=True,
            colorbar=True,
            default_tools=[],
            tools=[
                "hover",
                "pan",
                "wheel_zoom",
                "box_zoom",
                "reset",
                "save",
            ],
            title=label,
        )
        if vert_dim == "depth":
            a, b = bounded(bounds)
            plot = _with_seafloor(
                plot,
                ds.isel(ping_time=slice(a, b)),
                bottom=source.bottom,
                rasterized=source.rasterized,
            )
        return pn.pane.HoloViews(
            plot,
            linked_axes=False,
        )

    def linked_echogram(bounds, clim, palette, variable, ch, navigation):
        pane = echogram_view(
            bounds,
            clim,
            palette,
            variable=variable,
            channel=ch,
            label=f"{variable} — {available.get(ch, ch)}{' kHz' if ch in available else ''}",
        )
        if isinstance(pane, pn.pane.HoloViews):
            pane.object = attach_navigation(pane.object, navigation)
            pane.linked_axes = True
        return pane

    def multifrequency(bounds, clim, palette, variable):
        navigation = navigation_group()
        return pn.Column(
            *[linked_echogram(bounds, clim, palette, variable, ch, navigation) for ch in ordered],
            sizing_mode="stretch_width",
        )

    def tricolor(bounds, clim, variable, red, green, blue):
        try:
            w = window_dataset(tuple(bounds), variable, list(dict.fromkeys([red, green, blue])))
        except SourceChangedError as exc:
            return pn.pane.Alert(str(exc), alert_type="warning")
        # Keep geometry variables: w[["Sv"]] discards native echo_range/depth.
        dim = next(
            (
                name
                for name in (
                    ("depth",)
                    if dataset_validation.vertical_dim == "depth" or (has_bottom and "depth" in w)
                    else ("echo_range", "depth")
                )
                if name in w
            ),
            dataset_validation.vertical_dim,
        )
        status = validate_dataset(
            w,
            tricolor_channels=[red, green, blue],
            vert_dim=dim,
            var_name=variable,
            allow_variable_geometry=source.rasterized,
        )
        if not status.tricolor_ready:
            return pn.pane.Alert(status.tricolor_reason, alert_type="warning")
        plot = tricolor_echogram(
            w,
            channel=[red, green, blue],
            vmin=float(clim[0]),
            vmax=float(clim[1]),
            var_name=variable,
            vert_dim=dim,
            rasterized=source.rasterized,
            dynamic=False,
        ).opts(
            hooks=[_format_app_time_axis],
            responsive=True,
            height=450,
            framewise=False,
            xlim=plot_xrange(bounds),
            default_tools=[],
            tools=[
                "hover",
                "pan",
                "wheel_zoom",
                "box_zoom",
                "reset",
                "save",
            ],
        )

        if dim == "depth":
            a, b = bounded(bounds)
            plot = _with_seafloor(
                plot,
                ds.isel(ping_time=slice(a, b)),
                bottom=source.bottom,
                rasterized=source.rasterized,
            )
        return attach_navigation(plot, navigation_group())

    async def multifrequency_async(bounds, clim, palette, variable):
        multi_spinner.value = True
        apply_multifrequency.disabled = True
        try:
            return await asyncio.to_thread(multifrequency, bounds, clim, palette, variable)
        finally:
            multi_spinner.value = False

    async def tricolor_async(bounds, clim, variable, red, green, blue):
        rgb_spinner.value = True
        apply_tricolor.disabled = True
        try:
            return await asyncio.to_thread(tricolor, bounds, clim, variable, red, green, blue)
        finally:
            rgb_spinner.value = False

    def selection_label(bounds):
        a, b = bounded(bounds)
        return f"**Selected:** {a:,}–{b - 1:,} ({b - a:,} pings) · {str(times[a])[:19]} → {str(times[b - 1])[:19]} UTC"

    track_data = _parse_track(source)
    map_widget = pn.pane.Alert(
        "No latitude/longitude coordinates available.",
        alert_type="info",
    )
    if track_data is not None and len(track_data):
        # reuse the standalone Echoshader track element for ping positions
        # render its positions with HoloViews in Web Mercator coordinates:
        # this avoids the GeoViews/Cartopy projection error in tile overlays
        track_ds = ds
        if source.track is not None:
            indexed = track_data.set_index("ping_time").reindex(pd.to_datetime(times))
            track_ds = ds.assign(
                longitude=("ping_time", indexed.longitude.to_numpy()),
                latitude=("ping_time", indexed.latitude.to_numpy()),
            )
        track_plot = track_element(track_ds, tile=None)
        candidates = track_plot.traverse(lambda obj: obj, [gv.Points])
        if not candidates:
            raise ValueError("Track element has no selectable Points layer")
        track_points = candidates[0]
        rows = pd.DataFrame(
            {
                "longitude": track_points.dimension_values("Longitude"),
                "latitude": track_points.dimension_values("Latitude"),
                "ping_index": track_points.dimension_values("Ping Index"),
            }
        )
        latitude = np.clip(rows.latitude.to_numpy(dtype=float), -85.0, 85.0)
        rows["x"] = 6378137.0 * np.deg2rad(rows.longitude.to_numpy(dtype=float))
        rows["y"] = 6378137.0 * np.log(np.tan(np.pi / 4 + np.deg2rad(latitude) / 2))

        points = hv.Points(rows, kdims=["x", "y"], vdims=["ping_index", "longitude", "latitude"])
        selectable = points.opts(
            size=6,
            color="gray",
            alpha=0.7,
            selection_color="gray",
            selection_alpha=0.7,
            nonselection_color="gray",
            nonselection_alpha=0.7,
            tools=[
                "hover",
                "box_select",
                "tap",
                "pan",
                "wheel_zoom",
                "reset",
            ],
        )
        path = hv.Path([rows[["x", "y"]].to_numpy()]).opts(
            color="black",
            line_width=1,
        )
        tiles = hv.Tiles("https://tile.openstreetmap.org/{Z}/{X}/{Y}.png")
        x0, x1 = rows.x.min(), rows.x.max()
        y0, y1 = rows.y.min(), rows.y.max()
        xpad = max((x1 - x0) * 0.1, 200)
        ypad = max((y1 - y0) * 0.1, 200)
        base = (tiles * path * selectable).opts(
            responsive=True,
            min_height=220,
            data_aspect=None,
            xlim=(x0 - xpad, x1 + xpad),
            ylim=(y0 - ypad, y1 + ypad),
            show_grid=False,
            xlabel="Longitude",
            ylabel="Latitude",
            xaxis="bottom",
            yaxis="left",
            xformatter=longitude_formatter,
            yformatter=latitude_formatter,
        )
        selected_stream = hv.streams.Selection1D(source=selectable)

        def select_indices(event):
            indices = np.asarray(
                event.new if event.new is not None else [],
                dtype=int,
            )
            if not len(indices):
                return

            indices = indices[(indices >= 0) & (indices < len(rows))]
            if not len(indices):
                return

            ping_indices = rows.iloc[indices].ping_index.to_numpy(dtype=int)
            a = int(ping_indices.min())
            b = int(ping_indices.max()) + 1

            if b - a > max_pings:
                midpoint = (a + b) // 2
                a = max(
                    0,
                    min(n - max_pings, midpoint - max_pings // 2),
                )
                b = a + max_pings

            bounds = (a, b)
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(lambda: move_selection(bounds))
            else:
                move_selection(bounds)

        selected_stream.param.watch(select_indices, "index")

        def selected_points(window):
            a, b = bounded(window)
            selected = rows.loc[rows.ping_index.between(a, b - 1)]
            return hv.Points(
                selected,
                kdims=["x", "y"],
            ).opts(
                color="dodgerblue",
                alpha=1.0,
                size=6,
                selection_color="dodgerblue",
                selection_alpha=1.0,
                nonselection_color="dodgerblue",
                nonselection_alpha=1.0,
                tools=[],
            )

        highlight = hv.DynamicMap(
            selected_points,
            streams={"window": selection.param.window},
        )

        map_plot = (base * highlight).opts(
            responsive=True,
            height=source.map_height,
            data_aspect=None,
            framewise=False,
            xlabel="Longitude (°)",
            ylabel="Latitude (°)",
            xformatter=longitude_formatter,
            yformatter=latitude_formatter,
            xaxis="bottom",
            yaxis="left",
        )

        map_widget = pn.pane.HoloViews(
            map_plot,
            linked_axes=False,
            sizing_mode="stretch_width",
            height=source.map_height,
        )

    meta = [
        f"### {source.name}",
        f"**Pings:** {n:,}",
        f"**Coverage:** {str(times[0])[:19]} → {str(times[-1])[:19]} UTC",
        f"**Channels:** {len(channels)}",
    ]
    if source.path is not None:
        meta.append(f"**File:** `{source.path.name}`")
    meta.extend(
        ["### Frequencies"]
        + [
            f"- {available[ch]:g} kHz: `{ch}`" if ch in available else f"- `{ch}`"
            for ch in channels
        ]
    )

    # expose all source metadata without filling the sidebar with long histories
    attributes = {k: v for k, v in ds.attrs.items() if k.lower() != "history"}
    details = []
    if attributes:
        details.append(
            (
                "Dataset attributes",
                pn.pane.Markdown(
                    "\n\n".join(
                        f"**{key}**\n\n```text\n{value}\n```" for key, value in attributes.items()
                    )
                ),
            )
        )
    history = next((v for k, v in ds.attrs.items() if k.lower() == "history"), None)
    if history is not None:
        details.append(("History", pn.pane.Str(str(history))))

    interval = applied.param.interval
    clim = applied.param.clim
    optional_tabs = []
    for tab_name, content in extra_tabs or []:
        # Optional recipes may provide a native Panel object or a factory
        # that receives the two shared reactive Parameter references.
        optional_tabs.append(
            (
                tab_name,
                lazy_view(
                    lambda content=content: (
                        content(interval, clim) if callable(content) else content
                    )
                ),
            )
        )

    for widget in [rgb_variable_widget, *rgb_widgets]:
        widget.width = 230
        widget.sizing_mode = "fixed"

    visualization_controls = pn.FlexBox(
        pn.Column(color, width=270),
        pn.Column(cmap, width=180),
        variable_widget,
        apply_multifrequency,
        multi_spinner,
        sizing_mode="stretch_width",
    )

    tabs = pn.Tabs(
        (
            "Multi-frequency",
            pn.Column(
                visualization_controls,
                lazy_view(
                    pn.bind(
                        multifrequency_async,
                        interval,
                        clim,
                        applied.param.cmap,
                        applied.param.variable,
                    )
                ),
            ),
        ),
        (
            "Tricolor",
            pn.Column(
                pn.FlexBox(
                    rgb_variable_widget,
                    *rgb_widgets,
                    apply_tricolor,
                    rgb_spinner,
                    flex_wrap="wrap",
                    sizing_mode="stretch_width",
                ),
                lazy_view(
                    pn.bind(
                        tricolor_async,
                        interval,
                        clim,
                        applied.param.rgb_variable,
                        applied.param.rgb_red,
                        applied.param.rgb_green,
                        applied.param.rgb_blue,
                    )
                ),
            ),
        ),
        *optional_tabs,
        dynamic=True,
        sizing_mode="stretch_width",
    )
    tab_names = ["Multi-frequency", "Tricolor"] + [name for name, _ in optional_tabs]
    if source.initial_tab in tab_names:
        tabs.active = tab_names.index(source.initial_tab)
    sidebar = pn.Column(
        pn.pane.Markdown("\n\n".join(meta)),
        (
            pn.Accordion(*details, active=[], sizing_mode="stretch_width")
            if details
            else pn.Spacer(height=0)
        ),
        sizing_mode="stretch_width",
    )
    navigator = pn.Column(
        "### Select acoustic interval",
        pn.bind(selection_label, selection.param.window),
        timeline,
        start_input,
        end_input,
        sizing_mode="stretch_width",
    )
    map_panel = pn.Column(
        "### Ship track",
        pn.Row(
            map_widget,
            sizing_mode="stretch_width",
            align="center",
        ),
        sizing_mode="stretch_width",
        margin=0,
    )
    navigator.min_height = 340
    map_panel.min_height = source.map_height + 80
    for block in [navigator, map_panel, *sidebar.objects]:
        block.styles = {**block.styles, "flex-shrink": "0"}
    # Keep navigation with file information, leaving the main area for plots.
    sidebar.objects = [navigator, map_panel, *sidebar.objects]
    template = pn.template.FastListTemplate(
        title="Echoshader",
        theme_toggle=False,
        sidebar_width=330,
        sidebar=[sidebar],
        main=[tabs],
    )
    return template
