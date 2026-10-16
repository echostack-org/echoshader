from pathlib import Path

import panel as pn

from echoshader.app.builder import build_app
from echoshader.app.config import load_config


def launch(
    config_path: str | Path,
    *,
    port: int = 5006,
    address: str = "0.0.0.0",
    websocket_origin: str | list[str] | None = "*",
    admin: bool = False,
    show: bool = False,
    autoreload: bool = False,
):
    """Build and serve an Echoshader app from a YAML configuration."""

    config = load_config(config_path)
    app = build_app(config)

    return pn.serve(
        app,
        port=port,
        address=address,
        websocket_origin=websocket_origin,
        admin=admin,
        show=show,
        autoreload=autoreload,
    )
