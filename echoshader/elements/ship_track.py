"""Standalone full/recent ship-track and latest-position layers."""

import geoviews as gv
import pandas as pd
import panel as pn

from .map import CRS


def navigation_layer(
    data,
    history_hours: float = 6,
):
    """
    Create complete and recent ship-track layers.

    The complete available track is shown as a thin dashed line.
    The most recent ``history_hours`` are shown as a thicker solid line.
    """

    if data is None or data.empty:
        return gv.Overlay([])

    navigation = data.sort_values("timestamp_utc").reset_index(drop=True)

    latest = navigation.iloc[-1]

    ship_lon = float(latest["longitude"])

    ship_lat = float(latest["latitude"])

    # Complete ship track
    all_coordinates = navigation[["longitude", "latitude"]].to_numpy()

    full_track = gv.Path(
        [all_coordinates],
        crs=CRS,
    ).opts(
        line_width=1,
        line_dash="dashed",
        color="deepskyblue",
        alpha=0.6,
    )

    # Recent ship track
    latest_time = navigation["timestamp_utc"].max()

    cutoff = latest_time - pd.Timedelta(hours=history_hours)

    recent = navigation.loc[navigation["timestamp_utc"] >= cutoff].copy()

    # Include the final historical point so that the recent track
    # connects cleanly to the complete track
    older = navigation.loc[navigation["timestamp_utc"] < cutoff]

    if not older.empty:
        recent = pd.concat(
            [
                older.tail(1),
                recent,
            ],
            ignore_index=True,
        )

    recent_coordinates = recent[["longitude", "latitude"]].to_numpy()

    recent_track = gv.Path(
        [recent_coordinates],
        crs=CRS,
    ).opts(
        line_width=3,
        color="deepskyblue",
    )

    # Current ship position
    ship = gv.Points(
        [
            (
                ship_lon,
                ship_lat,
            )
        ],
        kdims=[
            "longitude",
            "latitude",
        ],
        crs=CRS,
    ).opts(
        size=12,
        marker="triangle",
        color="deepskyblue",
        tools=["hover"],
    )

    return full_track * recent_track * ship


def add_navigation(
    plot,
    dynamic_navigation,
):
    """Overlay live navigation on a geographic plot."""

    if isinstance(plot, pn.viewable.Viewable):
        return plot

    return (plot * dynamic_navigation).opts(framewise=False)


