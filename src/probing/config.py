"""Project configuration and local checkpoint lookup (never downloads)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    """Read `config.yaml` (single source of truth for paths, models and run settings)."""
    return yaml.safe_load((path or ROOT / "config.yaml").read_text())


def local_dir(repo: str, models_dir: str | None = None) -> Path:
    """Newest local snapshot folder of `repo` in the Hugging Face cache.

    Same logic as `scripts/01_tokenization_audit.py`. Raises if the repo is not cached.
    """
    from huggingface_hub import scan_cache_dir
    kw = {"cache_dir": models_dir} if models_dir else {}
    for r in scan_cache_dir(**kw).repos:
        if r.repo_id == repo:
            return Path(max(r.revisions, key=lambda v: v.last_modified).snapshot_path)
    raise FileNotFoundError(f"{repo} not found in the local cache")


def is_aligned(key: str) -> bool:
    """True for the five post-trained checkpoints, False for Base."""
    return key != "base"


def uses_bos(key: str, cfg: dict) -> bool:
    """Main-run BOS policy from `config.yaml` (`<|endoftext|>` first for all six checkpoints)."""
    return bool(cfg["bos"]["aligned" if is_aligned(key) else "base"])
