from pathlib import Path
from typing import Any

import yaml

RUN_CONFIGS = Path(__file__).parent.parent / "run_configs"


def load_config(path: str = "run_configs/default.yaml") -> dict[str, Any]:
    """Read package-relative YAML, defaulting to the native Inspect run configuration."""
    config = yaml.safe_load((Path(__file__).parent.parent / path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config
