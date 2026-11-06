"""Change-aware SPASSO maps and recorded navigation, loaded per visible tab."""

from pathlib import Path

import cartopy
import holoviews as hv
import pandas as pd
import panel as pn
from holoviews.streams import Pipe

from ...app.lazy import lazy_view
from ...app.refresh import file_revision, watch_changes
from ...elements.map import CRS, geographic_bounds
from ...elements.ship_track import add_navigation, navigation_layer
from .data import NavigationReader, find_latest_product_file
from .plotting import create_currents_plot, create_product_plot


def create_spasso_app(config):
    dashboard = config["dashboard"]
    root = Path(dashboard.get("spasso_path", "."))
    if dashboard.get("cartopy_path"):
        directory = Path(dashboard["cartopy_path"])
        directory.mkdir(parents=True, exist_ok=True)
        cartopy.config["data_dir"] = str(directory)
    reader = NavigationReader(config)
    pipe = Pipe(data=pd.DataFrame())
    track = hv.DynamicMap(
        lambda data: navigation_layer(
            data, history_hours=float(dashboard.get("navigation_history_hours", 6))
        ),
        streams=[pipe],
    )
    status = pn.pane.Markdown("Waiting for ship navigation…")
    notice = pn.pane.Alert(visible=False)
    size = dashboard.get("plot", {})
    width, height = int(size.get("width", 520)), int(size.get("height", 400))
    groups = {}
    for name, options in config.get("products", {}).items():
        groups.setdefault(options.get("tab", "Products"), []).append((name, options, False))
    currents = config.get("currents")
    if currents is not None:
        groups.setdefault(currents.get("tab", "Currents / Diagnostics"), []).insert(
            0, ("currents", currents, True)
        )
    # Store only rendered products for opened tabs, not a history of full maps.
    slots, revisions = {}, {}
    state = {"bounds": None, "navigation": None}

    def product_path(options, current):
        patterns = (
            [options.get("pattern", "????????_Copernicus_PHY.nc")]
            if current
            else options.get("patterns", [])
        )
        return find_latest_product_file(root, patterns)

    def snapshot():
        products = tuple(
            (name, file_revision(path) if (path := product_path(options, current)) else None)
            for group in groups.values()
            for name, options, current in group
        )
        return (reader.revision(), products)

    def update(revision):
        nav_revision, _ = revision
        if nav_revision != state["navigation"] or state["bounds"] is None:
            navigation = reader.read()
            if navigation.empty:
                raise ValueError("Navigation contains no valid positions")
            latest = navigation.iloc[-1]
            lon, lat = float(latest.longitude), float(latest.latitude)
            bounds = state["bounds"]
            # Recenter only after leaving the central half of the map window.
            if bounds is None or not (
                bounds[0] + (bounds[1] - bounds[0]) / 4
                <= lon
                <= bounds[1] - (bounds[1] - bounds[0]) / 4
                and bounds[2] + (bounds[3] - bounds[2]) / 4
                <= lat
                <= bounds[3] - (bounds[3] - bounds[2]) / 4
            ):
                state["bounds"] = geographic_bounds(lat, lon, dashboard.get("map_radius_km", 500))
            pipe.send(navigation)
            status.object = (
                f"**Ship position:** {lat:.5f}, {lon:.5f}  \n"
                f"**Navigation time:** {latest.timestamp_utc:%Y-%m-%d %H:%M:%S UTC}"
            )
            state["navigation"] = nav_revision
        complete = True
        for name, (slot, options, current) in slots.items():
            path = product_path(options, current)
            key = (file_revision(path) if path else None, state["bounds"])
            if revisions.get(name) == key:
                continue
            if path is None:
                slot.objects = [
                    pn.pane.Alert(
                        f"Waiting for {options.get('title', name)}…", alert_type="warning"
                    )
                ]
                revisions[name] = key
                continue
            common = dict(spasso_dir=root, bounds=state["bounds"], width=width, height=height)
            plot = (
                create_currents_plot(currents_config=options, **common)
                if current
                else create_product_plot(product_name=name, product_config=options, **common)
            )
            if isinstance(plot, pn.viewable.Viewable):
                # Plotting errors are already reported by the product reader.
                # Retain an existing good plot; retry without committing its revision.
                if name not in revisions:
                    slot.objects = [plot]
                complete = False
                notice.object = f"Could not update {options.get('title', name)}; retaining the last view and retrying.\n\n{getattr(plot, 'object', '')}"
                notice.visible = True
                continue
            if file_revision(path) != key[0]:
                complete = False
                continue
            plot = add_navigation(plot, track).opts(
                projection=CRS,
                framewise=False,
                backend_opts={"x_range.bounds": "auto", "y_range.bounds": "auto"},
            )
            slot.objects = [pn.pane.HoloViews(plot, linked_axes=True)]
            revisions[name] = key
        if complete:
            notice.visible = False
        return complete

    def error(exc):
        notice.object = (
            "Waiting for ship-navigation Parquet files…"
            if isinstance(exc, FileNotFoundError)
            else f"SPASSO update failed; retaining the last view and retrying: {exc}"
        )
        notice.alert_type = "warning"
        notice.visible = True

    check = watch_changes(
        snapshot, update, seconds=float(dashboard.get("refresh_seconds", 60)), on_error=error
    )

    def product_tab(group):
        panes = []
        for name, options, current in groups[group]:
            if name not in slots:
                slots[name] = (pn.Column("Waiting for data…", width=width), options, current)
            panes.append(slots[name][0])
        check(force=True)
        return pn.FlexBox(
            *panes, flex_wrap="wrap", sizing_mode="stretch_width", styles={"gap": "12px"}
        )

    tabs = pn.Tabs(
        *[(name, lazy_view(lambda name=name: product_tab(name))) for name in groups],
        dynamic=True,
        sizing_mode="stretch_width",
    )
    check()
    return pn.Column(
        f"# {dashboard.get('title', 'SPASSO')}", status, notice, tabs, sizing_mode="stretch_width"
    )
