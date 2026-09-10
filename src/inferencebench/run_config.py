from pathlib import Path
from typing import Any

import yaml


def load_config(path: str = "run_configs/default.yaml") -> dict[str, Any]:
    """Read a run configuration relative to the installed task package."""
    return yaml.safe_load((Path(__file__).parent / path).read_text())
