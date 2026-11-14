"""Helpers for the replication at larger sizes: file hashes, run-vs-run comparison, H1 verdict, category-order rho.

Pure functions on DataFrames / paths, so `scripts/12_ingest.py` and `scripts/12_scale.py` only arrange them into tables.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_script(path: Path) -> ModuleType:
    """Import a numbered script (e.g. `scripts/05_derived.py`) as a module, to reuse its functions without copying them.

    The scripts keep their work under `if __name__ == "__main__"`, so importing runs no analysis and writes nothing.
    """
    spec = importlib.util.spec_from_file_location(f"script_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare_column(ref: pd.DataFrame, new: pd.DataFrame, keys: list[str], col: str) -> dict:
    """Row-by-row comparison of `col` between two runs matched on `keys` (one-to-one, same key set required).

    Returns n, max and mean absolute difference, and the Pearson correlation.
    """
    m = ref[keys + [col]].merge(new[keys + [col]], on=keys, how="outer", suffixes=("_ref", "_new"), validate="one_to_one",
                                indicator=True)
    if (m["_merge"] != "both").any():
        raise ValueError(f"key sets differ: {int((m['_merge'] != 'both').sum())} unmatched rows")
    a, b = m[f"{col}_ref"].to_numpy(float), m[f"{col}_new"].to_numpy(float)
    d = np.abs(a - b)
    return dict(n=len(m), max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), pearson_r=float(np.corrcoef(a, b)[0, 1]))


def mean_d_gate(ref: pd.DataFrame, new: pd.DataFrame, models: list[str], tol: float) -> pd.DataFrame:
    """Mean D per model in two runs (prefix-level D tables) and whether |difference| <= tol."""
    rows = []
    for m in models:
        a = ref.loc[ref["model"] == m, "D"]
        b = new.loc[new["model"] == m, "D"]
        if len(a) != len(b) or a.isna().any() or b.isna().any():
            raise ValueError(f"{m}: incomplete D ({len(a)} vs {len(b)} rows)")
        diff = float(b.mean() - a.mean())
        rows.append(dict(model=m, mean_D_ref=float(a.mean()), mean_D_new=float(b.mean()), diff=diff, tol=tol, passed=abs(diff) <= tol))
    return pd.DataFrame(rows)


def h1_verdict(h1: pd.DataFrame) -> pd.Series:
    """H1 verdict per row of an `h1_table` output: 'supported' if mean D > 0, the bootstrap 95% CI excludes 0 and the
    Bonferroni x5 Wilcoxon p < 0.05; else 'not supported'."""
    ok = (h1["mean"] > 0) & (h1["ci_boot_lo"] > 0) & (h1["wilcoxon_p_bonf5"] < ALPHA)
    return ok.map({True: "supported", False: "not supported"})


def category_order_rho(ref: pd.Series, new: pd.Series) -> float:
    """Spearman rho between two per-category values (same categories, matched by index). Descriptive: n = 5, no p."""
    if set(ref.index) != set(new.index):
        raise ValueError("category sets differ")
    return float(stats.spearmanr(ref.sort_index(), new.sort_index()).statistic)
