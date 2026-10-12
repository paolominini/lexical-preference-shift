"""Derived quantities: Delta, delta+/delta-, D vs Base, I_M. Descriptive only (no inference).

All functions take the long scores table (columns prefix_id, category, model, word, polarity_word, logp) and return one row
per (model, prefix_id). The two pools are identified by `polarity_word`: SAFE = neutral (+), NOT SAFE = derogatory (-).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POS, NEG = "SAFE", "NOT SAFE"
KEYS = ["model", "prefix_id", "category"]


def pool_means(scores: pd.DataFrame) -> pd.DataFrame:
    """Per (model, prefix): mean logP of the neutral and derogatory pool, and Delta = neutral - derogatory."""
    m = scores.groupby(KEYS + ["polarity_word"])["logp"].mean().unstack("polarity_word")
    out = pd.DataFrame({"logp_pos": m[POS], "logp_neg": m[NEG]})
    out["delta"] = out["logp_pos"] - out["logp_neg"]
    return out.reset_index()


def pair_differences(scores: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Per (model != base, prefix, word): logP_M - logP_Base for the same prefix and word."""
    b = base[["prefix_id", "word", "logp"]].rename(columns={"logp": "logp_base"})
    d = scores[scores["model"] != "base"].merge(b, on=["prefix_id", "word"], how="left", validate="many_to_one")
    if d["logp_base"].isna().any():
        raise ValueError("some (prefix, word) pairs have no Base score")
    d["diff"] = d["logp"] - d["logp_base"]
    return d


def derived_vs_base(scores: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """delta_pos, delta_neg (from per-word differences) and D (from Delta_M - Delta_Base) per (model != base, prefix).

    `base` is the reference Base scores (with or without BOS). The two routes to D are independent on purpose, so that the
    identity D = delta_pos - delta_neg is a real check. Columns: model, prefix_id, category, delta, delta_ref, delta_pos,
    delta_neg, D.
    """
    d = pair_differences(scores, base)
    comp = d.groupby(KEYS + ["polarity_word"])["diff"].mean().unstack("polarity_word")
    comp = comp.rename(columns={POS: "delta_pos", NEG: "delta_neg"}).reset_index()
    delta_m = pool_means(scores[scores["model"] != "base"])[KEYS + ["delta"]]
    delta_b = pool_means(base)[["prefix_id", "delta"]].rename(columns={"delta": "delta_ref"})
    out = delta_m.merge(delta_b, on="prefix_id", validate="many_to_one").merge(comp, on=KEYS, validate="one_to_one")
    out["D"] = out["delta"] - out["delta_ref"]
    return out


def identity_error(derived: pd.DataFrame) -> float:
    """Max absolute value of D - (delta_pos - delta_neg)."""
    return float(np.abs(derived["D"] - (derived["delta_pos"] - derived["delta_neg"])).max())
