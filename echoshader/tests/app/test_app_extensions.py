"""Check page-level extension loading without starting a server."""

import sys
import types

import numpy as np
import panel as pn
import pytest
import xarray as xr

from echoshader.app import builder


def dataset():
    return xr.Dataset(
        {"Sv": (("channel", "ping_time", "depth"), np.zeros((1, 2, 2)))},
        coords={
            "channel": ["70"],
            "depth": [1.0, 2.0],
            "ping_time": np.array(
                ["2024-08-05T13:20:00", "2024-08-05T13:20:01"], dtype="datetime64[ns]"
            ),
        },
    )


def test_yaml_loads_explicit_extension_factory(monkeypatch, tmp_path):
    source = dataset()
    panel = pn.pane.Markdown("External extension")
    calls = []
    module = types.ModuleType("test_external_acoustic_extension")

    def factory(**kwargs):
        calls.append(kwargs)
        return None, [("Recent transects", panel)]

    module.make_tabs = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    app = builder.build_app(
        {
            "extensions": {
                "recent": {
                    "factory": module.__name__ + ":make_tabs",
                    "root": str(tmp_path),
                    "count": 10,
                }
            }
        },
        datasets={"survey": source},
    )
    assert not calls  # Extensions are loaded only when their page is opened.
    lazy = app.main[0][2]  # Home, Single file, then the extension page.
    content = lazy.eval(lazy.object)
    assert len(calls) == 1
    assert calls[0]["ds"] is source
    assert calls[0]["root"] == tmp_path.resolve()
    assert calls[0]["config"] == {"root": str(tmp_path), "count": 10}
    assert content[0] is panel


def test_basic_viewer_does_not_load_extensions(monkeypatch):
    calls = []

    def unexpected_factory(path):
        raise AssertionError("Basic viewer tried to load an extension")

    from echoshader.app import pages

    monkeypatch.setattr(pages, "_factory", unexpected_factory)
    monkeypatch.setattr(
        builder, "build_acoustic_app", lambda source, **kw: calls.append((source, kw))
    )
    source = dataset()
    builder._build_file_app(
        {"extensions": {"recent": {"factory": "unused:factory"}}},
        datasets={"survey": source},
    )
    assert len(calls) == 1
    assert calls[0][0].data is source
    assert calls[0][0].bottom is None
    assert not calls[0][1].get("extra_tabs")


def test_unknown_extension_still_fails_without_factory():
    with pytest.raises(ValueError, match="Unknown extension"):
        builder.build_app({"extensions": {"missing": {}}}, datasets={"survey": dataset()})
