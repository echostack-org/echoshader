"""Launch an Echoshader application from a YAML configuration."""

import argparse
import logging
from pathlib import Path

import panel as pn

from .builder import build_app
from .config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def launch(
    config_path: str | Path,
    *,
    port: int | None = None,
    address: str = "127.0.0.1",
    websocket_origin=None,
    admin: bool = False,
    show: bool = True,
    autoreload: bool = False,
):
    """Build and serve an Echoshader application."""

    path = Path(config_path).expanduser().resolve()
    config = load_config(path)

    settings = config.get("app", {})
    port = int(port if port is not None else settings.get("port", 5006))
    route = str(settings.get("route", "")).strip("/")

    # Each browser session gets its own app and selection state.
    def app_factory():
        return build_app(config, config_dir=path.parent)

    return pn.serve(
        {route: app_factory},
        port=port,
        address=address,
        websocket_origin=websocket_origin
        or [
            f"localhost:{port}",
            f"127.0.0.1:{port}",
        ],
        admin=admin,
        show=show,
        autoreload=autoreload,
    )


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Launch an Echoshader application")
    parser.add_argument("config", type=Path)
    parser.add_argument("--port", type=int, default=None)

    args = parser.parse_args()
    launch(args.config, port=args.port)


if __name__ == "__main__":
    main()
