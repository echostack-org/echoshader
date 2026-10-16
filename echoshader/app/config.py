from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load an Echoshader application configuration."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise TypeError("Application configuration must be a YAML mapping.")

    return config
