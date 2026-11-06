"""Defer constructing a view until its tab becomes visible."""

import panel as pn


def lazy_view(factory):
    return pn.param.ParamFunction(
        factory,
        lazy=True,
        defer_load=True,
        loading_indicator=True,
        sizing_mode="stretch_width",
    )
