"""Coherence tables on the consolidated scores (descriptive only; the inferential analysis is in 07-09).

Per checkpoint: mean logP neutral / derogatory, mean Delta, share of prefixes with Delta > 0, share of neutral-vs-derogatory
word pairs with neutral higher. Provisional D (Delta_M - Delta_Base) per checkpoint and category, and the same D
with Base scored without BOS (robustness d). Tables in reports/step_04_tables/. Prints at most ~20 lines.

Usage: .venv/bin/python scripts/04c_sanity_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probing.config import load_config  # noqa: E402

TAB = ROOT / "reports/step_04_tables"


def deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Delta(c|s) per (model, prefix): mean logP of neutral words minus mean logP of derogatory words."""
    m = df.groupby(["model", "prefix_id", "category", "polarity_word"])["logp"].mean().unstack("polarity_word")
    m["delta"] = m["SAFE"] - m["NOT SAFE"]
    return m.reset_index()


def pair_share(df: pd.DataFrame) -> pd.Series:
    """Share of (neutral, derogatory) word pairs, within a prefix, where the neutral word has the higher logP."""
    out = {}
    for model, g in df.groupby("model"):
        n = hit = 0
        for _, h in g.groupby("prefix_id"):
            pos, neg = h.loc[h["polarity_word"] == "SAFE", "logp"].to_numpy(), h.loc[h["polarity_word"] == "NOT SAFE", "logp"].to_numpy()
            n += pos.size * neg.size
            hit += int((pos[:, None] > neg[None, :]).sum())
        out[model] = hit / n
    return pd.Series(out, name="share_pairs_neutral_higher")


def d_vs(delta: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    b = base.set_index("prefix_id")["delta"]
    d = delta[delta["model"] != "base"].copy()
    d["D"] = d["delta"] - d["prefix_id"].map(b)
    return d


def main() -> None:
    cfg = load_config()
    res = ROOT / cfg["paths"]["results"]
    size = cfg["run"]["size"]
    main_df = pd.read_parquet(res / f"scores_{size}.parquet")
    nobos = pd.read_parquet(res / f"scores_{size}_base_nobos.parquet")
    dl, dn = deltas(main_df), deltas(nobos)
    order = list(cfg["models"][size])
    summ = dl.groupby("model").agg(mean_logp_neutral=("SAFE", "mean"), mean_logp_derog=("NOT SAFE", "mean"),
                                   mean_delta=("delta", "mean"), share_prefixes_delta_pos=("delta", lambda x: (x > 0).mean()))
    summ = summ.join(pair_share(main_df)).loc[order]
    nb = dn.set_index("model").loc["base"]
    summ.loc["base_nobos"] = [nb["SAFE"].mean(), nb["NOT SAFE"].mean(), nb["delta"].mean(), (nb["delta"] > 0).mean(),
                              pair_share(nobos)["base"]]
    summ.to_csv(TAB / "summary_by_model.csv")
    base = dl[dl["model"] == "base"]
    d_main, d_nb = d_vs(dl, base), d_vs(dl, dn[dn["model"] == "base"])
    rows = []
    for name, d in (("base_BOS", d_main), ("base_noBOS", d_nb)):
        for cat, g in [("ALL", d)] + list(d.groupby("category")):
            for m, h in g.groupby("model"):
                rows.append(dict(reference=name, category=cat, model=m, n=len(h), mean_D=h["D"].mean(),
                                 share_D_pos=(h["D"] > 0).mean(), wilcoxon_p=wilcoxon(h["D"]).pvalue))
    dtab = pd.DataFrame(rows)
    dtab.to_csv(TAB / "provisional_D_by_model_category.csv", index=False)
    print(summ.round(3).to_string())
    allr = dtab[dtab["category"] == "ALL"].pivot(index="model", columns="reference", values="mean_D").loc[order[1:]]
    print("\nprovisional mean D (Delta_M - Delta_Base), all 500 prefixes:\n", allr.round(3).to_string())
    kto = dtab[(dtab["model"] == "kto") & (dtab["reference"] == "base_BOS")][["category", "mean_D", "share_D_pos", "wilcoxon_p"]]
    print("\nKTO vs Base (BOS) by category:\n", kto.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
