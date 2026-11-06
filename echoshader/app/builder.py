"""Build the default HoloViz application from YAML configuration.

Elements remain standalone in :mod:`echoshader.elements`.
"""

from __future__ import annotations

import os
from pathlib import Path

from echoshader.validation import validate_dataset

from .data import open_source
from .registry import EXTENSIONS
from .single_viewer import AcousticSource, build_acoustic_app


def _data_path(value: str, config_dir: str | Path | None) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute() and config_dir is not None:
        path = Path(config_dir) / path
    return path.resolve()


def _build_file_app(
    config: dict,
    datasets: dict | None = None,
    context: dict | None = None,
    *,
    config_dir: str | Path | None = None,
):
    """Build only the acoustic viewer; the page shell owns extensions."""

    app = config.get("app", {})
    name = app.get("source", "survey")
    sources = config.get("data", {})
    if name not in sources and (not datasets or name not in datasets):
        raise ValueError(f"Missing data source {name!r}")

    settings = sources.get(name, {})
    path = _data_path(settings["path"], config_dir) if "path" in settings else None
    if datasets and name in datasets:
        ds = datasets[name]
    else:
        if path is None or not path.exists():
            raise FileNotFoundError(f"Acoustic dataset not found: {path}")
        ds = open_source(settings, path)

    display = config.get("display", {})
    variables = display.get("variables", ["Sv"])
    if not isinstance(variables, (list, tuple)) or not variables:
        raise ValueError("display.variables must be a non-empty list of acoustic variable names")
    if not all(isinstance(name, str) and name for name in variables):
        raise ValueError("display.variables must contain non-empty strings")
    # The app validates every selected variable, including shared geometry.
    validate_dataset(ds, var_name=variables[0])

    navigation = config.get("navigation", {})
    display = config.get("display", {})
    max_pings = int(navigation.get("max_pings", 500))
    max_range_samples = int(display.get("max_range_samples", 1200))

    source = AcousticSource(
        data=ds,
        variables=tuple(variables),
        name=str(settings.get("label", name)),
        path=path,
        max_pings=max_pings,
        max_range_samples=max_range_samples,
        rasterized=bool(display.get("rasterized", True)),
        cache_mb=int(display.get("cache_mb", 256)),
        initial_tab=str(app.get("initial_tab", "Multi-frequency")),
        map_width=int(display.get("map_width", 300)),
        map_height=int(display.get("map_height", 230)),
    )
    return build_acoustic_app(source, title=app.get("title", "Echoshader"))


def build_app(config, datasets=None, context=None, *, config_dir=None):
    """Compose a lightweight home, a file viewer, and independent extension pages."""
    from .pages import build_pages

    variables = config.get("display", {}).get("variables", ["Sv"])
    if (
        not isinstance(variables, (list, tuple))
        or not variables
        or not all(isinstance(v, str) and v for v in variables)
    ):
        raise ValueError("display.variables must be a non-empty list of acoustic variable names")
    for name, options in config.get("extensions", {}).items():
        if not (options or {}).get("factory") and name not in EXTENSIONS:
            raise ValueError(f"Unknown extension {name!r}")
    if datasets:
        name = config.get("app", {}).get("source", "survey")
        if name in datasets:
            validate_dataset(datasets[name], var_name=variables[0])
    return build_pages(config, datasets, config_dir=config_dir)
