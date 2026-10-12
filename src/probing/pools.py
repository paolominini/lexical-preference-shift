"""Prefixes, probe-word pools and the (prefix, word) pairs to score."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from probing.config import ROOT

VOCAB_SHEET = "Final_Probing_Set"


@dataclass(frozen=True)
class Pair:
    """One (prefix, word) pair. A prefix of category c is scored only with the pools of c."""
    prefix_id: str
    category: str
    prefix: str
    word: str
    polarity_word: str


def load_prefixes(path: Path | None = None) -> pd.DataFrame:
    """The 500 manually validated prefixes."""
    df = pd.read_csv(path or ROOT / "data/final_prefix_set.csv")
    df["category"] = df["category"].str.strip()
    return df


def load_vocabulary(path: Path | None = None) -> pd.DataFrame:
    """The 41 probe-word rows: columns category, polarity, word."""
    df = pd.read_excel(path or ROOT / "data/Vocabulary_Final.xlsx", sheet_name=VOCAB_SHEET)
    df = df.rename(columns=str.lower)[["category", "polarity", "word"]]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def pools(vocab: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """category -> {"SAFE": [...], "NOT SAFE": [...]} in vocabulary order."""
    out: dict[str, dict[str, list[str]]] = {}
    for (cat, pol), grp in vocab.groupby(["category", "polarity"], sort=False):
        out.setdefault(cat, {})[pol] = list(grp["word"])
    return out


def build_pairs(prefixes: pd.DataFrame, vocab: pd.DataFrame) -> list[Pair]:
    """All (prefix, word) pairs, each prefix with its own category's pools only."""
    by_cat = pools(vocab)
    missing = set(prefixes["category"]) - set(by_cat)
    if missing:
        raise ValueError(f"categories without a pool: {missing}")
    pairs = []
    for pid, cat, pfx in zip(prefixes["final_id"], prefixes["category"], prefixes["prefix"]):
        for pol, words in by_cat[cat].items():
            pairs += [Pair(pid, cat, pfx, w, pol) for w in words]
    return pairs
