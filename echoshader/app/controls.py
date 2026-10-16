import panel as pn

CONTROLS = {
    "range_slider": pn.widgets.RangeSlider,
    "select": pn.widgets.Select,
    "checkbox": pn.widgets.Checkbox,
}


def build_control(
    control_config: dict,
):
    """Build one Panel control from configuration."""

    control_type = control_config["type"]

    if control_type not in CONTROLS:
        raise ValueError(f"Unknown control type: {control_type!r}.")

    control_class = CONTROLS[control_type]

    args = control_config.get(
        "args",
        {},
    )

    return control_class(
        **args,
    )
