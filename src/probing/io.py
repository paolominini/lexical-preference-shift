"""Resumable, append-only writer of scores: parquet parts, never overwritten.

Layout: results/scores_<size>/<model>/part_NNNN.parquet. scripts/04b_gate_and_consolidate.py concatenates them into scores_<size>.parquet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["prefix_id", "category", "model", "model_repo", "word", "polarity_word", "n_tokens", "logp",
           "input_format", "bos", "dtype"]
KEY = ["prefix_id", "model", "word"]


def read_parts(out_dir: Path) -> pd.DataFrame:
    """All rows written so far in `out_dir` (empty frame with the right columns if none)."""
    parts = sorted(Path(out_dir).glob("part_*.parquet"))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


class PartWriter:
    """Writes rows as new numbered parts; knows which keys are done so a run can resume."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        done = read_parts(self.out_dir)
        self.done: set[tuple] = set(map(tuple, done[KEY].itertuples(index=False, name=None)))

    def is_done(self, prefix_id: str, model: str, word: str) -> bool:
        return (prefix_id, model, word) in self.done

    def _next_path(self) -> Path:
        nums = [int(p.stem.split("_")[1]) for p in self.out_dir.glob("part_*.parquet")]
        return self.out_dir / f"part_{max(nums, default=-1) + 1:04d}.parquet"

    def write(self, rows: list[dict]) -> Path:
        """Write `rows` as a new part. Refuses wrong columns, duplicate keys and existing files."""
        df = pd.DataFrame(rows)
        if list(df.columns) != COLUMNS:
            raise ValueError(f"columns must be {COLUMNS}, got {list(df.columns)}")
        keys = list(map(tuple, df[KEY].itertuples(index=False, name=None)))
        if len(set(keys)) != len(keys) or self.done.intersection(keys):
            raise ValueError("duplicate (prefix_id, model, word): refusing to write")
        path = self._next_path()
        with open(path, "xb") as fh:  # 'x' = fail if the file exists, never overwrite
            df.to_parquet(fh, index=False)
        self.done.update(keys)
        return path
