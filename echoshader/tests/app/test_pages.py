import sys
import types

import numpy as np
import panel as pn
import xarray as xr

from echoshader.app.builder import build_app
from echoshader.app.holoviz import AcousticSource, build_acoustic_app


def dataset():
    return xr.Dataset(
        {"Sv": (("channel", "ping_time", "depth"), np.full((3, 4, 5), -60.0))},
        coords={
            "channel": ["a", "b", "c"],
            "depth": np.arange(5.0),
            "ping_time": np.datetime64("2024-01-01") + np.arange(4) * np.timedelta64(1, "s"),
            "range_sample": (("channel", "depth"), np.arange(15).reshape(3, 5)),
        },
    )


def test_home_does_not_open_source(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Home opened an acoustic source")

    monkeypatch.setattr(xr, "open_zarr", fail)
    app = build_app({"data": {"survey": {"type": "zarr", "path": "/not/opened.zarr"}}})
    assert app.title == "Echoshader"
    assert app.theme_toggle is False
    assert app.main[0].active == 0
    assert app.sidebar_width == 330


def test_file_page_and_catalogue_switch(monkeypatch):
    calls = []
    module = types.ModuleType("test_catalogue")

    def catalogue(**kwargs):
        return {
            "Latest completed transect": lambda: dataset(),
            "Other file": lambda: calls.append("other") or dataset(),
        }

    module.catalogue = catalogue
    monkeypatch.setitem(sys.modules, module.__name__, module)
    app = build_app(
        {"data": {"survey": {}}, "file_catalogue": {"factory": "test_catalogue:catalogue"}},
        datasets={"survey": dataset()},
    )
    pages = app.main[0]
    pages.active = 1
    page = pages[1]
    panel = page.eval(page.object)
    assert len(panel) == 1
    assert app.sidebar_width == 330
    sidebar = app.sidebar[0]
    menu = sidebar[0]
    menu.value = "Other file"
    assert calls == ["other"]
    assert "Alert" not in type(panel[0]).__name__
    # Independent pages hide the file's sidebar controls.
    pages.active = 0
    assert not sidebar.visible


def test_rgb_preserves_channel_specific_auxiliary_coordinates():
    app = build_acoustic_app(AcousticSource(dataset()))
    tabs = app.main[0]
    rgb = tabs[1][1]
    result = rgb.eval(rgb.object)
    assert not isinstance(result, pn.pane.Alert)


def test_independent_extension_does_not_open_file(monkeypatch):
    module = types.ModuleType("test_independent_extension")

    def factory(**kwargs):
        assert not kwargs["ds"].variables
        return None, [("Transects", lambda *_: pn.pane.Markdown("Overview"))]

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def fail(*args, **kwargs):
        raise AssertionError("Independent overview opened the single file")

    monkeypatch.setattr(xr, "open_zarr", fail)
    app = build_app(
        {
            "data": {"survey": {"type": "zarr", "path": "/missing.zarr"}},
            "extensions": {
                "recent_transects": {
                    "factory": "test_independent_extension:factory",
                    "requires_source": False,
                }
            },
        }
    )
    pane = app.main[0][2]
    content = pane.eval(pane.object)
    assert isinstance(content, pn.Column)
