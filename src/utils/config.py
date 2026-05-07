"""Config loader: reads YAML and provides a dot-accessible config object."""
import yaml
from pathlib import Path
from typing import Any


class Config:
    """Dot-accessible configuration object."""

    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self):
        return f"Config({vars(self)})"


def load_config(path: str) -> Config:
    """Load a YAML config file and return a Config object."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    print(f"[config] Loaded config from {path} (mode={data.get('mode', 'unknown')})")
    return Config(data)
