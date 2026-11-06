"""Adapt the generic app extension interface to the SPASSO dashboard."""

from copy import deepcopy
from pathlib import Path


def build_tabs(*, ds, root, config, max_pings=500, max_range_samples=1200):
    from .app import create_spasso_app

    settings = deepcopy(config)
    dashboard = settings.setdefault("dashboard", {})
    if not isinstance(dashboard, dict):
        raise TypeError("SPASSO requires a dashboard mapping")
    if not isinstance(settings.get("products", {}), dict):
        raise TypeError("SPASSO products must be a mapping")
    for name, default in (("spasso_path", "."), ("navigation_path", "navigation"), ("cartopy_path", "cartopy_cache")):
        path = Path(dashboard.get(name, default)).expanduser()
        dashboard[name] = str(path if path.is_absolute() else Path(root) / path)

    return None, [("SPASSO", lambda *_: create_spasso_app(settings))]
