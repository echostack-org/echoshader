"""RGB region editor, adapted from Safwannn89's PR #226.

The input EVR is read-only. Changes are saved to a separate Regions2D CSV.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import DatetimeTicker, DatetimeTickFormatter
from holoviews.streams import PolyDraw, PolyEdit

from echoshader.elements.echogram import tricolor_echogram
from echoshader.validation import resolve_rgb_channels


@dataclass
class Region:
    region_id: int
    name: str
    xs: np.ndarray  # milliseconds UTC as float, Bokeh's date coordinate
    ys: np.ndarray  # physical depth, not sample indices

    def copy(self):
        return Region(self.region_id, self.name, self.xs.copy(), self.ys.copy())


def _read_region(row):
    times = pd.to_datetime(row["time"], utc=True).tz_localize(None)
    depths = np.asarray(row["depth"], dtype=float)
    xs = times.to_numpy(dtype="datetime64[ms]").astype("int64").astype(float)
    if len(xs) < 3 or len(xs) != len(depths):
        raise ValueError("Region has invalid vertex count")
    if not np.isfinite(xs).all() or not np.isfinite(depths).all():
        raise ValueError("Region has non-finite vertices")
    return Region(int(row["region_id"]), str(row.get("region_name", "")), xs, depths.copy())


def _read_stream(stream):
    data = getattr(stream, "data", None) or {}
    if len(data.get("xs", [])) != 1 or len(data.get("ys", [])) != 1:
        raise ValueError("There must be exactly one edited/drawn polygon")
    xs = np.asarray(data["xs"][0], dtype=float)
    ys = np.asarray(data["ys"][0], dtype=float)
    if len(xs) < 3 or len(xs) != len(ys):
        raise ValueError("Polygon must have at least three vertices")
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        raise ValueError("Polygon vertices must be finite")
    return xs.copy(), ys.copy()


def _time_axis(plot, element):
    for axis in plot.state.xaxis:
        axis.ticker = DatetimeTicker(desired_num_ticks=9)
        axis.formatter = DatetimeTickFormatter(
            seconds="%H:%M:%S",
            minsec="%H:%M:%S",
            minutes="%H:%M",
            hourmin="%H:%M",
            hours="%H:%M",
            days="%b %d %H:%M",
            months="%Y-%m",
            years="%Y",
        )
        axis.major_label_orientation = 0.6
        axis.axis_label = "Time (UTC)"


def _poly(region, vertical, color, *, selected=False):
    # Each visual receives independent coordinate arrays, never mutable source arrays.
    return hv.Polygons(
        [
            {
                "ping_time": region.xs.copy(),
                vertical: region.ys.copy(),
                "region_id": str(region.region_id),
            }
        ],
        kdims=["ping_time", vertical],
        vdims=["region_id"],
    ).opts(
        line_color=color,
        color=color,
        fill_color=color,
        fill_alpha=0.08 if selected else 0.0,
        line_width=3 if selected else 1.5,
    )


def _overlay(plot, layer):
    return layer if plot is None else plot * layer


def build_region_editor(*, ds, root, config, max_pings=500, max_range_samples=1200):
    """YAML extension factory. Returns an independent Panel page for RGB regions.

    `max_range_samples` is accepted for compatibility but intentionally unused:
    this editor never subsamples vertical acoustic data.
    """
    import echoregions as er  # optional dependency

    evr = Path(config["regions"])
    if not evr.is_absolute():
        evr = Path(root) / evr
    regions_obj = er.read_evr(str(evr))
    # Keep the complete original object alive and untouched for future export.
    original = {}
    for _, row in regions_obj.data.iterrows():
        region = _read_region(row)
        if region.region_id in original:
            raise ValueError(f"Duplicate region ID: {region.region_id}")
        original[region.region_id] = region
    if not original:
        return None, [("RGB regions", pn.pane.Alert("No regions in EVR"))]

    # Deep copies of every original region; deletions affect this collection only.
    working = {rid: region.copy() for rid, region in original.items()}
    removed = set()
    next_id = [max(original) + 1]

    # The original EVR stays read-only; Modified is persisted separately.
    output = Path(config.get("modified_path", "regions/modified/regions_modified.csv"))
    if not output.is_absolute():
        output = Path(root) / output

    if output.exists():
        saved = er.read_regions_csv(output).data
        working = {_read_region(row).region_id: _read_region(row) for _, row in saved.iterrows()}
        removed = set(original) - set(working)
        next_id[0] = max(set(original) | set(working)) + 1

    def persist(updated, deleted):
        # Build a Regions2D-compatible table without changing regions_obj.data.
        original_rows = regions_obj.data.set_index("region_id", drop=False)
        rows = []
        for rid, region in sorted(updated.items()):
            if rid in deleted:
                continue
            if rid in original_rows.index:
                row = original_rows.loc[rid].copy(deep=True)
                if (
                    np.array_equal(region.xs, original[rid].xs)
                    and np.array_equal(region.ys, original[rid].ys)
                    and region.name == original[rid].name
                ):
                    rows.append(row)  # Preserve untouched EVR metadata and timestamp precision.
                    continue
            else:
                row = pd.Series({col: None for col in regions_obj.data.columns})
            times = pd.to_datetime(region.xs, unit="ms").to_numpy(dtype="datetime64[ns]")
            row["region_id"] = rid
            row["region_name"] = region.name
            row["time"] = times
            row["depth"] = region.ys.copy()
            row["region_point_count"] = len(times)
            row["region_bbox_left"] = pd.Timestamp(times.min())
            row["region_bbox_right"] = pd.Timestamp(times.max())
            row["region_bbox_top"] = float(region.ys.min())
            row["region_bbox_bottom"] = float(region.ys.max())
            rows.append(row)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(output.stem + ".tmp.csv")
        try:
            er.read_regions_csv(pd.DataFrame(rows, columns=regions_obj.data.columns)).to_csv(
                temp, index=False
            )
            # Verify that Echoregions can reopen the CSV before replacing a previous save.
            if len(er.read_regions_csv(temp).data) != len(rows):
                raise ValueError("Echoregions CSV round-trip failed")
            os.replace(temp, output)
        finally:
            temp.unlink(missing_ok=True)

    red, green, blue = resolve_rgb_channels(
        ds, config.get("rgb_frequencies", [120000, 38000, 18000])
    )
    vmin, vmax = map(float, config.get("rgb_clim", [-90, -30]))
    if vmax <= vmin:
        raise ValueError("rgb_clim must have a larger maximum")

    rendering = config.get("rendering", {})
    rasterized = bool(rendering.get("rasterized", False))
    window_size = min(len(ds.ping_time), max(2, int(rendering.get("max_pings", max_pings))))
    ds_times = np.asarray(ds.ping_time.values, dtype="datetime64[ns]")
    if len(ds_times) < 2:
        raise ValueError("A transect requires at least two time bins")
    max_start = max(0, len(ds_times) - window_size)
    vertical = "depth" if "depth" in ds else "echo_range"
    if vertical not in ds:
        raise ValueError("Missing physical depth/echo_range coordinate")
    ymin = float(ds[vertical].min())
    ymax = float(ds[vertical].max())

    display = config.get("regions_display", {})
    viewing = pn.widgets.RadioButtonGroup(
        name="Active collection", options=["Original", "Modified"], value="Modified"
    )
    show_original = pn.widgets.Checkbox(
        name="Show Original overlay", value=display.get("show_original", True)
    )
    show_modified = pn.widgets.Checkbox(
        name="Show Modified overlay", value=display.get("show_modified", True)
    )
    original_color = pn.widgets.ColorPicker(
        name="Original color", value=display.get("original_color", "#000000")
    )
    modified_color = pn.widgets.ColorPicker(
        name="Modified color", value=display.get("modified_color", "#00D4FF")
    )

    # This ID is independent of the active collection; switching never resets it.
    selected = pn.widgets.Select(name="Selected region", options={}, width=350)
    prev_region = pn.widgets.Button(name="◀ Region", width=105)
    next_region = pn.widgets.Button(name="Region ▶", width=105)
    mode = pn.widgets.RadioButtonGroup(
        name="Action", options=["Browse", "Edit vertices", "Draw new"], value="Browse"
    )
    name_input = pn.widgets.TextInput(
        name="New region name", value="New region", width=240, visible=False
    )
    replace_ids = pn.widgets.MultiChoice(
        name="Remove these working regions after drawing", options={}, visible=False
    )
    save = pn.widgets.Button(name="Apply to Modified", button_type="primary")
    cancel = pn.widgets.Button(name="Cancel editing")
    delete = pn.widgets.Button(name="Remove from Modified", button_type="warning")
    restore = pn.widgets.Button(name="Restore in Modified", button_type="light")
    status = pn.pane.Markdown("**Browse:** navigate and inspect. Original EVR is never changed.")

    window = pn.widgets.IntSlider(
        name="Acoustic window start (time-bin index)",
        start=0,
        end=max_start,
        step=1,
        value=0,
        width=650,
    )
    back = pn.widgets.Button(name="◀ Transect")
    forward = pn.widgets.Button(name="Transect ▶")
    window_caption = pn.pane.Markdown("")
    pending = {"stream": None, "mode": None, "region_id": None}
    suppress = [False]

    def all_ids():
        return sorted(set(original) | set(working))

    def available_ids():
        # Include deleted originals in navigation so they can be inspected/restored.
        ids = all_ids()
        start, end = ds_times[0], ds_times[-1]
        visible = []
        for rid in ids:
            r = working.get(rid) or original.get(rid)
            if r is None:
                continue
            x0 = np.datetime64(int(r.xs.min()), "ms").astype("datetime64[ns]")
            x1 = np.datetime64(int(r.xs.max()), "ms").astype("datetime64[ns]")
            if x0 <= end and x1 >= start:
                visible.append(rid)
        return visible

    def update_selector(keep=None):
        ids = available_ids()
        opts = {}
        for rid in ids:
            region = working.get(rid) or original.get(rid)
            marker = " (removed)" if rid in removed else ""
            opts[f"{rid} — {region.name}{marker}"] = rid
        suppress[0] = True
        try:
            selected.options = opts
            selected.value = keep if keep in ids else (ids[0] if ids else None)
            replace_ids.options = {
                f"{rid} — {(working.get(rid) or original[rid]).name}": rid
                for rid in ids
                if rid in working and rid not in removed
            }
            replace_ids.value = [
                rid for rid in replace_ids.value if rid in replace_ids.options.values()
            ]
        finally:
            suppress[0] = False

    update_selector()

    def jump_to_region(rid):
        if rid is None:
            return
        region = working.get(rid) or original.get(rid)
        if region is None:
            return
        t0 = np.datetime64(int(np.min(region.xs)), "ms").astype("datetime64[ns]")
        t1 = np.datetime64(int(np.max(region.xs)), "ms").astype("datetime64[ns]")
        start = max(0, int(np.searchsorted(ds_times, t0)) - 15)
        stop = min(len(ds_times), int(np.searchsorted(ds_times, t1, side="right")) + 15)
        if stop - start > window_size:
            mid = t0 + (t1 - t0) / 2
            start = max(0, int(np.searchsorted(ds_times, mid)) - window_size // 2)
            status.object = "**Region exceeds window:** showing its center; move along the transect to inspect the rest."
        window.value = min(max_start, start)

    def on_region(event):
        if suppress[0]:
            return
        mode.value = "Browse"
        jump_to_region(event.new)

    selected.param.watch(on_region, "value")

    def step_region(delta):
        ids = available_ids()
        if ids:
            index = ids.index(selected.value) if selected.value in ids else 0
            selected.value = ids[(index + delta) % len(ids)]

    prev_region.on_click(lambda event: step_region(-1))
    next_region.on_click(lambda event: step_region(1))

    def step_window(delta):
        window.value = max(0, min(max_start, window.value + delta * max(1, window_size // 2)))

    back.on_click(lambda event: step_window(-1))
    forward.on_click(lambda event: step_window(1))

    def on_mode(event):
        drawing = event.new == "Draw new"
        name_input.visible = drawing
        replace_ids.visible = drawing
        if event.new != "Browse" and viewing.value == "Original":
            viewing.value = "Modified"
        pending.update(stream=None, mode=None, region_id=None)
        if event.new == "Browse":
            status.object = "**Browse:** view regions and move along the transect; no editing."
        elif event.new == "Edit vertices":
            status.object = "**Edit vertices:** select/hold the working polygon and drag its vertices; click Apply to Modified."
        else:
            status.object = "**Draw new:** use the polygon-draw toolbar tool; double-click to finish, then Apply to Modified."

    mode.param.watch(on_mode, "value")

    def on_collection(event):
        # Never reset `selected`: the selected ID survives collection switches.
        if event.new == "Original" and mode.value != "Browse":
            mode.value = "Browse"

    viewing.param.watch(on_collection, "value")

    def on_delete(event):
        rid = selected.value
        if rid is None or rid not in working or rid in removed:
            return
        try:
            persist(working, removed | {rid})
        except (OSError, ValueError, TypeError) as exc:
            status.object = f"**Remove failed:** {exc}"
            return
        removed.add(rid)
        mode.value = "Browse"
        update_selector(keep=rid)
        status.object = f"Working region {rid} removed and saved. Original unchanged."
        refresh_plot()

    def on_restore(event):
        rid = selected.value
        if rid is None:
            return
        updated = working.copy()
        deleted = removed - {rid}
        if rid not in original:
            return
        updated[rid] = original[rid].copy()
        try:
            persist(updated, deleted)
        except (OSError, ValueError, TypeError) as exc:
            status.object = f"**Restore failed:** {exc}"
            return
        working.clear()
        working.update(updated)
        removed.clear()
        removed.update(deleted)
        mode.value = "Browse"
        update_selector(keep=rid)
        status.object = f"Working region {rid} restored and saved. Original unchanged."
        refresh_plot()

    def on_cancel(event):
        mode.value = "Browse"
        pending.update(stream=None, mode=None, region_id=None)
        status.object = "Editing cancelled; no working-copy changes applied."

    def on_save(event):
        if pending["stream"] is None or pending["mode"] != mode.value:
            status.object = "**No active edit.** Select Edit vertices or Draw new."
            return
        try:
            xs, ys = _read_stream(pending["stream"])
        except (ValueError, TypeError) as exc:
            status.object = f"**Not applied:** {exc}"
            return
        updated = working.copy()
        deleted = removed.copy()
        if mode.value == "Edit vertices":
            rid = selected.value
            if rid not in working or rid in removed or pending["region_id"] != rid:
                status.object = "**Not applied:** selected working region is unavailable."
                return
            updated[rid] = Region(rid, working[rid].name, xs, ys)
            msg = f"Working region {rid} updated"
        elif mode.value == "Draw new":
            rid = next_id[0]
            updated[rid] = Region(rid, name_input.value.strip() or f"New region {rid}", xs, ys)
            replaced = [int(r) for r in replace_ids.value if r in working and r != rid]
            deleted.update(replaced)
            msg = f"New working region {rid} created; {len(replaced)} regions removed"
        else:
            return
        try:
            persist(updated, deleted)
        except (OSError, ValueError, TypeError) as exc:
            status.object = f"**Save failed:** {exc}"
            return
        working.clear()
        working.update(updated)
        removed.clear()
        removed.update(deleted)
        next_id[0] = max(next_id[0], rid + 1)
        replace_ids.value = []
        pending.update(stream=None, mode=None, region_id=None)
        mode.value = "Browse"
        update_selector(keep=rid)
        jump_to_region(rid)
        status.object = f"**{msg} and saved to {output.name}.** Original EVR untouched."
        refresh_plot()

    save.on_click(on_save)
    delete.on_click(on_delete)
    restore.on_click(on_restore)
    cancel.on_click(on_cancel)

    def display_name():
        rid = selected.value
        r = working.get(rid) or original.get(rid)
        return f"Region {rid} — {r.name}" if r is not None else "No region selected"

    @pn.depends(
        selected.param.value,
        window.param.value,
        mode.param.value,
        viewing.param.value,
        show_original.param.value,
        show_modified.param.value,
        original_color.param.value,
        modified_color.param.value,
    )
    def display(rid, window_start, action, active, orig_on, mod_on, orig_col, mod_col):
        window_stop = min(len(ds_times), window_start + window_size)
        window_caption.object = (
            f"**UTC:** {pd.Timestamp(ds_times[window_start])} — "
            f"{pd.Timestamp(ds_times[window_stop - 1])} · "
            f"{window_stop - window_start} native time bins · {ds.sizes[vertical]} depth samples"
        )
        subset = (
            ds.sel(channel=[red, green, blue])
            .isel(ping_time=slice(window_start, window_stop))
            .compute()
        )
        # No acoustic slicing in the depth direction, for any rendering mode.
        rgb = tricolor_echogram(
            subset,
            channel=[red, green, blue],
            vmin=vmin,
            vmax=vmax,
            vert_dim=vertical,
            rasterized=rasterized,
            dynamic=False,
        )
        overlay = rgb
        # Passive layers are NOT connected to editing streams.
        if orig_on:
            for key, r in original.items():
                if key == rid and active == "Original" and action != "Draw new":
                    continue
                if r.xs.max() < ds_times[window_start].astype("datetime64[ms]").astype(
                    float
                ) or r.xs.min() > ds_times[window_stop - 1].astype("datetime64[ms]").astype(float):
                    continue
                overlay = _overlay(overlay, _poly(r, vertical, orig_col, selected=key == rid))
        if mod_on:
            for key, r in working.items():
                if key in removed:
                    continue
                if key == rid and active == "Modified" and action == "Edit vertices":
                    continue
                if r.xs.max() < ds_times[window_start].astype("datetime64[ms]").astype(
                    float
                ) or r.xs.min() > ds_times[window_stop - 1].astype("datetime64[ms]").astype(float):
                    continue
                overlay = _overlay(overlay, _poly(r, vertical, mod_col, selected=key == rid))
        pending.update(stream=None, mode=None, region_id=None)
        if (
            action == "Edit vertices"
            and active == "Modified"
            and rid in working
            and rid not in removed
        ):
            editable = _poly(working[rid], vertical, mod_col, selected=True).opts(
                active_tools=["poly_edit"]
            )
            stream = PolyEdit(
                source=editable,
                shared=False,
                vertex_style={"size": 10, "color": "white", "line_color": mod_col},
            )
            pending.update(stream=stream, mode=action, region_id=rid)
            overlay = overlay * editable
        elif action == "Draw new" and active == "Modified":
            drawing = hv.Polygons([], kdims=["ping_time", vertical], vdims=["region_id"]).opts(
                line_color=mod_col,
                color=mod_col,
                fill_alpha=0.12,
                active_tools=["poly_draw"],
            )
            stream = PolyDraw(source=drawing, drag=True, num_objects=1, show_vertices=True)
            pending.update(stream=stream, mode=action, region_id=None)
            overlay = overlay * drawing
        # Current region remains visible in Original mode, even when Modified removed it.
        if orig_on and rid in original and active == "Original" and action != "Draw new":
            overlay = overlay * _poly(original[rid], vertical, orig_col, selected=True)
        plot = overlay.opts(
            width=1100,
            height=550,
            responsive=True,
            invert_yaxis=True,
            ylim=(ymin, ymax),
            xlabel="Time (UTC)",
            ylabel=f"{vertical} (m)",
            tools=["pan", "wheel_zoom", "box_zoom", "reset"],
            hooks=[_time_axis],
        )
        return pn.Column(
            pn.pane.Markdown(f"**{display_name()}** · Active: **{active}** · Tool: **{action}**"),
            pn.pane.HoloViews(plot, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    # .param.trigger refreshes the rendering after working dict modifications.
    def refresh_plot():
        window.param.trigger("value")

    # Initial view jumps to the first overlapping region only once.
    jump_to_region(selected.value)
    controls = pn.Column(
        "### Region layers",
        viewing,
        pn.Row(show_original, show_modified),
        pn.Row(original_color, modified_color),
        "### Region navigation",
        selected,
        pn.Row(prev_region, next_region),
        "### Transect navigation",
        pn.Row(back, forward),
        window,
        window_caption,
        "### Working-copy tools",
        mode,
        name_input,
        replace_ids,
        save,
        cancel,
        delete,
        restore,
        status,
        sizing_mode="stretch_width",
    )
    app = pn.Column(
        "## RGB Region Editor",
        display,
        sizing_mode="stretch_width",
    )
    return controls, [("RGB regions", app)]
