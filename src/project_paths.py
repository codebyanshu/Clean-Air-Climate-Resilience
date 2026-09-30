from __future__ import annotations

from pathlib import Path
from typing import Optional
import yaml


def find_project_root(start: Optional[Path] = None) -> Path:
    """Return the repository root independently of the current working directory."""
    candidate = (start or Path.cwd()).resolve()
    for path in [candidate, *candidate.parents]:
        if (path / "data" / "raw" / "city_day.csv").exists() or (path / "requirements.txt").exists():
            if (path / "data").exists() and (path / "src").exists():
                return path
    return candidate


PROJECT_ROOT = find_project_root()
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
