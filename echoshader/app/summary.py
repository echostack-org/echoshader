import numpy
import panel as pn

from .selection import apply_selection


def _selected_dataset(
    data,
    selection,
    ping_dim: str = "ping_time",
):
    """Return the currently selected dataset."""

    if selection is None:
        return data

    return apply_selection(
        data,
        selection,
        projection=ping_dim,
        drop=True,
    )


def _format_datetime(
    value,
):
    """Format a numpy datetime value compactly."""

    if value is None:
        return "—"

    value = numpy.datetime64(
        value,
        "s",
    )

    return str(value).replace(
        "T",
        " ",
    )


def _format_duration(
    start,
    end,
):
    """Format a datetime interval."""

    if start is None or end is None:
        return "—"

    seconds = int(
        (
            numpy.datetime64(
                end,
                "s",
            )
            - numpy.datetime64(
                start,
                "s",
            )
        )
        / numpy.timedelta64(
            1,
            "s",
        )
    )

    hours, remainder = divmod(
        max(
            seconds,
            0,
        ),
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    if hours:
        return f"{hours} h {minutes} min {seconds} s"

    if minutes:
        return f"{minutes} min {seconds} s"

    return f"{seconds} s"


def selection_summary(
    data,
    selection=None,
    ping_dim: str = "ping_time",
):
    """Build a compact summary of the current data selection."""

    selected = _selected_dataset(
        data,
        selection,
        ping_dim=ping_dim,
    )

    total_pings = data.sizes.get(
        ping_dim,
        0,
    )

    selected_pings = selected.sizes.get(
        ping_dim,
        0,
    )

    if ping_dim in selected and selected_pings:
        times = selected[ping_dim].values

        start = times[0]

        end = times[-1]

    else:
        start = None
        end = None

    if selection is None:
        title = "Entire dataset"

    else:
        title = "Current selection"

    markdown = f"""
### Selection

**{title}**

**Pings:** {selected_pings:,} / {total_pings:,}

**Start:** {_format_datetime(start)}

**End:** {_format_datetime(end)}

**Duration:** {_format_duration(start, end)}
"""

    return pn.pane.Markdown(
        markdown,
        sizing_mode="stretch_width",
        margin=(
            5,
            10,
        ),
    )


def selection_metadata(
    data,
    selection=None,
    var_name: str = "Sv",
    ping_dim: str = "ping_time",
):
    """Build compact metadata for the current selection."""

    selected = _selected_dataset(
        data,
        selection,
        ping_dim=ping_dim,
    )

    if "channel" in selected:
        channels = [str(value) for value in selected["channel"].values.tolist()]

    else:
        channels = []

    if var_name in selected:
        variable = selected[var_name]

        dimensions = " × ".join(variable.dims)

        shape = " × ".join(str(variable.sizes[dim]) for dim in variable.dims)

    else:
        dimensions = "—"
        shape = "—"

    channel_text = "\n".join(f"- {channel}" for channel in channels)

    if not channel_text:
        channel_text = "—"

    markdown = f"""
### Metadata

**Variable:** `{var_name}`

**Dimensions:** `{dimensions}`

**Shape:** `{shape}`

**Channels**

{channel_text}
"""

    return pn.pane.Markdown(
        markdown,
        sizing_mode="stretch_width",
        margin=(
            5,
            10,
        ),
    )
