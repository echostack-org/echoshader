"""Numbered transect tracks on an OpenStreetMap background."""

import numpy as np
import panel as pn


def transect_map(tracks, config, numbers):
    """Small Bokeh map using actual GPS positions; no acoustic reprocessing."""
    from bokeh.models import ColumnDataSource, HoverTool, LabelSet, WMTSTileSource
    from bokeh.palettes import Category10
    from bokeh.plotting import figure

    valid_tracks = []
    for number, lon, lat in tracks:
        valid = np.isfinite(lon) & np.isfinite(lat) & (abs(lon) <= 180) & (abs(lat) < 85.051129)
        if not valid.any():
            continue
        # NaNs break the line, avoiding connections across missing GPS samples.
        x = np.where(valid, 6378137.0 * np.deg2rad(lon), np.nan)
        safe_lat = np.where(valid, lat, 0.0)
        y = np.where(
            valid, 6378137.0 * np.log(np.tan(np.pi / 4 + np.deg2rad(safe_lat) / 2)), np.nan
        )
        valid_tracks.append((number, lon, lat, x, y, valid))
    width = int(config.get("map_width", 540))
    if not valid_tracks:
        return pn.Column(
            "**Transect locations**",
            pn.pane.Alert(
                "No usable latitude/longitude in the transect files.", alert_type="warning"
            ),
            width=width,
            margin=0,
        )
    xs = np.concatenate([t[3][t[5]] for t in valid_tracks])
    ys = np.concatenate([t[4][t[5]] for t in valid_tracks])
    dx, dy = max(np.ptp(xs), 1000.0), max(np.ptp(ys), 1000.0)
    plot = figure(
        width=width,
        height=int(config.get("map_height", 750)),
        x_axis_type="mercator",
        y_axis_type="mercator",
        x_range=(float(xs.min() - 0.15 * dx), float(xs.max() + max(0.15 * dx, 50000.0))),
        y_range=(float(ys.min() - 0.15 * dy), float(ys.max() + 0.15 * dy)),
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        title="Transect locations",
    )
    plot.add_tile(
        WMTSTileSource(
            url="https://tile.openstreetmap.org/{Z}/{X}/{Y}.png",
            attribution="© OpenStreetMap contributors",
        )
    )
    labels = {"x": [], "y": [], "number": [], "color": []}
    hover_renderers = []
    for i, (number, lon, lat, x, y, valid) in enumerate(valid_tracks):
        color = Category10[10][i % 10]
        source = ColumnDataSource(
            {"x": x, "y": y, "longitude": lon, "latitude": lat, "number": [number] * len(x)}
        )
        plot.line("x", "y", source=source, line_color=color, line_width=3)
        hover_renderers.append(
            plot.scatter("x", "y", source=source, size=4, fill_color=color, line_color=None)
        )
        middle = np.flatnonzero(valid)[np.count_nonzero(valid) // 2]
        for key, value in [
            ("x", x[middle]),
            ("y", y[middle]),
            ("number", number),
            ("color", color),
        ]:
            labels[key].append(value)
    plot.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="number",
            text_color="color",
            source=ColumnDataSource(labels),
            x_offset=7,
            y_offset=5,
            text_font_size="11pt",
            text_font_style="bold",
            background_fill_color="white",
            background_fill_alpha=0.8,
        )
    )
    plot.add_tools(
        HoverTool(
            renderers=hover_renderers,
            tooltips=[
                ("Transect", "@number"),
                ("Longitude", "@longitude{0.0000}"),
                ("Latitude", "@latitude{0.0000}"),
            ],
        )
    )
    available = {t[0] for t in valid_tracks}
    missing = [n for n in numbers if n not in available]
    content = [pn.pane.Bokeh(plot)]
    if missing:
        content.append(pn.pane.Markdown("GPS unavailable: " + ", ".join(missing)))
    return pn.Column(*content, width=width, margin=0)


def track_revision(tracks, numbers):
    """Stable map-input fingerprint; acoustic values and color settings are irrelevant."""
    from hashlib import blake2b

    return (
        tuple(numbers),
        tuple(
            (
                number,
                tuple(
                    (
                        np.asarray(values).shape,
                        np.asarray(values).dtype.str,
                        blake2b(np.asarray(values).tobytes(), digest_size=16).digest(),
                    )
                    for values in (longitude, latitude)
                ),
            )
            for number, longitude, latitude in tracks
        ),
    )
