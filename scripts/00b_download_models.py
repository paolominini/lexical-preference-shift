#!/usr/bin/env python3
"""
00b_download_models.py - download the six checkpoints of one size into the Hugging Face cache.

Only weights in safetensors + config + tokenizer files are fetched (no duplicate pytorch_model.bin / TF / Flax
copies). Resumable: if interrupted, run it again. Checks free disk first.

    python scripts/00b_download_models.py --size 1.4b [--dry-run] [--only base --only sft]
"""
from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
ALLOW = ["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"]


def wanted(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in ALLOW)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default=None)
    ap.add_argument("--only", action="append", default=[], help="model key (base, sft, sft_dpo, sft_ppo, sft_kto, kto)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    size = a.size or cfg["run"]["size"]
    models_dir = cfg["paths"].get("models_dir")      # None = default Hugging Face cache
    from huggingface_hub import HfApi, constants, scan_cache_dir, snapshot_download

    api = HfApi()
    cached = set()
    try:
        cached = {r.repo_id for r in scan_cache_dir(models_dir).repos}
    except Exception:  # noqa: BLE001
        pass
    plan = []
    for key, repo in cfg["models"][size].items():
        if a.only and key not in a.only:
            continue
        try:
            info = api.model_info(repo, files_metadata=True)
            gib = sum((s.size or 0) for s in info.siblings if wanted(s.rfilename)) / GIB
        except Exception as e:  # noqa: BLE001
            print(f"cannot reach the Hub for {repo}: {type(e).__name__}. Check your internet connection.")
            sys.exit(1)
        plan.append((key, repo, gib, repo in cached))

    todo = sum(g for _, _, g, c in plan if not c)
    target = Path(models_dir or constants.HF_HUB_CACHE)
    while not target.exists():          # the cache folder may not exist yet: measure the disk it will be created on
        target = target.parent
    free = shutil.disk_usage(target).free / GIB
    print(f"{'key':8s} {'GiB':>6s}  cached  repo")
    for key, repo, gib, c in plan:
        print(f"{key:8s} {gib:6.2f}  {'yes' if c else 'no':6s}  {repo}")
    print(f"\nto download ~{todo:.1f} GiB into {models_dir or constants.HF_HUB_CACHE}, free disk {free:.0f} GiB")
    if free < todo * 1.2:
        print("NOT ENOUGH DISK SPACE: free some space and retry.")
        sys.exit(1)
    if a.dry_run:
        return
    for key, repo, gib, c in plan:
        print(f"\n== {key}: {repo}")
        path = snapshot_download(repo_id=repo, allow_patterns=ALLOW, cache_dir=models_dir)   # resumes; instant if already complete
        print(f"   ready: {path}")
    print("\nAll done. Now run: python scripts/00_check_env.py --time")


if __name__ == "__main__":
    main()
