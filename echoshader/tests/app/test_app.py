import numpy as np
import panel as pn
import pytest
import xarray as xr
import yaml

from echoshader.app.builder import build_app
from echoshader.app.config import load_config
from echoshader.app.launcher import launch


def make_dataset():
    return xr.Dataset(
        {
            "Sv": (
                ("channel", "ping_time", "echo_range"),
                np.random.rand(1, 5, 10),
            )
        },
        coords={
            "channel": ["38 kHz"],
            "ping_time": np.arange(5),
            "echo_range": np.arange(10),
        },
    )


def test_load_config(tmp_path):
    config_path = tmp_path / "app_config.yaml"

    config_path.write_text(
        yaml.safe_dump(
            {
                "app": {
                    "title": "Test app",
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["app"]["title"] == "Test app"


def test_load_config_requires_mapping(tmp_path):
    config_path = tmp_path / "app_config.yaml"

    config_path.write_text(
        "- one\n- two\n",
        encoding="utf-8",
    )

    with pytest.raises(
        TypeError,
        match="Application configuration must be a YAML mapping",
    ):
        load_config(config_path)


def test_build_app():
    ds = make_dataset()

    config = {
        "elements": {
            "echogram": {
                "type": "echogram",
                "data": "survey",
                "args": {
                    "channel": "38 kHz",
                    "vert_dim": "echo_range",
                },
            }
        },
        "layout": {
            "type": "column",
            "children": [
                "echogram",
            ],
        },
    }

    app = build_app(
        config,
        datasets={
            "survey": ds,
        },
    )

    assert isinstance(
        app,
        pn.template.FastListTemplate,
    )


def test_launch(monkeypatch, tmp_path):
    config_path = tmp_path / "app_config.yaml"

    config_path.write_text(
        yaml.safe_dump(
            {
                "app": {
                    "title": "Test app",
                },
                "elements": {},
                "layout": {
                    "type": "column",
                    "children": [],
                },
            }
        ),
        encoding="utf-8",
    )

    called = {}

    def fake_serve(app, **kwargs):
        called["app"] = app
        called["kwargs"] = kwargs
        return "server"

    monkeypatch.setattr(
        pn,
        "serve",
        fake_serve,
    )

    result = launch(
        config_path,
        port=5007,
        address="127.0.0.1",
        show=False,
    )

    assert result == "server"
    assert isinstance(
        called["app"],
        pn.template.FastListTemplate,
    )
    assert called["kwargs"]["port"] == 5007
    assert called["kwargs"]["address"] == "127.0.0.1"
    assert called["kwargs"]["show"] is False
