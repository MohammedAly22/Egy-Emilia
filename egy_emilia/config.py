"""Central config loader. Reads config.yaml into nested SimpleNamespace objects
so code can write `cfg.download.sample_rate` instead of dict lookups."""

from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str | Path = DEFAULT_CONFIG) -> SimpleNamespace:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _to_ns(raw)
    cfg._repo_root = REPO_ROOT          # noqa: SLF001 (handy for resolving rel paths)
    return cfg


def resolve(path_str: str) -> Path:
    """Resolve a config path relative to the repo root (absolute paths pass through)."""
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)
