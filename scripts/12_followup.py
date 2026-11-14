"""Follow-up diagnostics of the replication across sizes, descriptive only (no test; the tables of 12_scale.py are untouched).

A. D vs Base by prefix-toxicity tercile, per model and size (Detoxify `toxicity` from results/toxicity.parquet, as in 06).
B. Distance of each aligned model from its own Base, per size: mean |logp_M - logp_Base| over the same (prefix, word) pairs, by pool,
   and the Pearson r of logp_M vs logp_Base.
The 1.4B point is the Mac main run; 2.8b, 6.9b, 12b are the server working copies. Runs only if the reproduction check of 12_ingest.py passed.
Writes:
  reports/step_12_tables/s12f_D_by_toxicity_tercile.csv  mean D [bootstrap CI], share D > 0, difference vs 1.4B [paired CI]
  reports/step_12_tables/s12f_tercile_composition.csv    prefixes per tercile x category and x source polarity
  reports/step_12_tables/s12f_distance_from_base.csv     mean |diff| and signed diff by pool, Pearson r (all pairs and by pool)
  reports/step_12_tables/s12f_gate_checks.csv
Usage: .venv/bin/python scripts/12_followup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probing.config import load_config  # noqa: E402
from probing.derived import NEG, POS, pair_differences  # noqa: E402
from probing.scale import load_script  # noqa: E402
from probing.stats import bootstrap_mean_ci  # noqa: E402

TAB = ROOT / "reports/step_12_tables"
SIZES = ["1.4b", "2.8b", "6.9b", "12b"]
REF = "1.4b"
TERCILES = ["low", "mid", "high"]
d07 = load_script(ROOT / "scripts/07_rq1.py")
LABEL, N_BOOT = d07.LABEL, d07.N_BOOT


def terciles(tox: pd.DataFrame) -> pd.Series:
    """Prefix -> toxicity tercile (equal-count split of the 500 prefixes)."""
    return pd.qcut(tox.set_index("prefix_id")["toxicity"], 3, labels=TERCILES).astype(str)


def d_by_tercile(ds: dict[str, pd.DataFrame], terc: pd.Series, aligned: list[str], seed: int) -> pd.DataFrame:
    """Per size x model x tercile: mean D vs Base [bootstrap CI], share D > 0, and the paired difference vs the 1.4B run [CI]."""
    rows = []
    for size in SIZES:
        for m in aligned:
            d = ds[size].loc[ds[size]["model"] == m].set_index("prefix_id")["D"].sort_index()
            ref = ds[REF].loc[ds[REF]["model"] == m].set_index("prefix_id")["D"].sort_index()
            for t in TERCILES:
                ids = terc.index[terc == t]
                y = d.loc[ids].to_numpy()
                lo, hi = bootstrap_mean_ci(y, N_BOOT, seed)
                row = dict(size=size, model=m, label=LABEL[m], tercile=t, n=len(y), mean_D=y.mean(), ci_boot_lo=lo, ci_boot_hi=hi,
                           share_pos=float((y > 0).mean()))
                if size != REF:
                    diff = y - ref.loc[ids].to_numpy()
                    dlo, dhi = bootstrap_mean_ci(diff, N_BOOT, seed)
                    row.update(diff_vs_1_4b=diff.mean(), diff_ci_lo=dlo, diff_ci_hi=dhi)
                rows.append(row)
    return pd.DataFrame(rows)


def distance_from_base(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per aligned model: mean |logp_M - logp_Base| and mean signed difference by pool and overall, Pearson r of logp_M vs logp_Base.

    Also returns the pair-level differences (for the check against delta_pos / delta_neg)."""
    d = pair_differences(scores, scores[scores["model"] == "base"])
    rows = []
    for m, g in d.groupby("model"):
        row = dict(model=m, label=LABEL[m], n_pairs=len(g), mean_abs_all=g["diff"].abs().mean(), r_all=g["logp"].corr(g["logp_base"]))
        for pool, name in [(POS, "neutral"), (NEG, "derog")]:
            p = g[g["polarity_word"] == pool]
            row.update({f"mean_abs_{name}": p["diff"].abs().mean(), f"mean_signed_{name}": p["diff"].mean(),
                        f"r_{name}": p["logp"].corr(p["logp_base"])})
        rows.append(row)
    return pd.DataFrame(rows), d


def main() -> None:
    cfg = load_config()
    res = ROOT / cfg["paths"]["results"]
    seed = int(cfg["project"]["seed"])
    aligned = list(cfg["models"][REF])[1:]
    if not pd.read_csv(TAB / "reproduction_gate.csv")["passed"].all():
        sys.exit("reproduction check not passed: run 12_ingest.py first; no analysis")
    tox = pd.read_parquet(res / "toxicity.parquet")
    terc = terciles(tox)
    ds = {s: pd.read_parquet(res / f"delta_{s}.parquet") for s in SIZES}
    checks = [dict(check="500 prefixes with toxicity, tercile sizes", value=str(terc.value_counts().reindex(TERCILES).tolist()),
                   pass_=len(terc) == 500 and sorted(terc.value_counts().tolist()) == [166, 167, 167])]

    a = d_by_tercile(ds, terc, aligned, seed)
    edges = tox["toxicity"].quantile([1 / 3, 2 / 3]).to_numpy()
    a["tercile_edges"] = f"low <= {edges[0]:.4f} < mid <= {edges[1]:.4f} < high"
    sc = pd.read_csv(TAB / "scale_D_by_size.csv")
    w = (a.assign(s=a["mean_D"] * a["n"]).groupby(["size", "model"])[["s", "n"]].sum().assign(mean=lambda t: t["s"] / t["n"])["mean"]
         .reset_index().merge(sc[["size", "model", "mean_D"]], on=["size", "model"], validate="one_to_one"))
    v = float((w["mean"] - w["mean_D"]).abs().max())
    checks.append(dict(check="n-weighted mean of tercile D == scale_D_by_size mean D (20 cells)", value=v, pass_=v < 1e-12 and len(w) == 20))

    t = tox.assign(tercile=tox["prefix_id"].map(terc), category=tox["category"].map(cfg["display_labels"]))
    comp = pd.concat([pd.crosstab(t["tercile"], t["category"]), pd.crosstab(t["tercile"], t["source_polarity"])], axis=1)
    comp = comp.reindex(TERCILES).reset_index()

    dist = []
    for size in SIZES:
        scores = pd.read_parquet(res / f"scores_{size}.parquet")
        b, pairs = distance_from_base(scores)
        dist.append(b.assign(size=size))
        per_prefix = pairs.groupby(["model", "prefix_id", "polarity_word"])["diff"].mean().unstack("polarity_word")
        dm = ds[size][ds[size]["model"] != "base"].set_index(["model", "prefix_id"])
        v = float(max((per_prefix[POS] - dm["delta_pos"]).abs().max(), (per_prefix[NEG] - dm["delta_neg"]).abs().max()))
        checks.append(dict(check=f"{size}: per-prefix pool mean of logp_M - logp_Base == delta_pos / delta_neg", value=v, pass_=v < 1e-9))
        checks.append(dict(check=f"{size}: 4,100 pairs per aligned model", value=str(sorted(b["n_pairs"].unique())),
                           pass_=(b["n_pairs"] == 4100).all() and len(b) == 5))
    dist = pd.concat(dist, ignore_index=True)
    dist = dist[["size"] + [c for c in dist.columns if c != "size"]]
    gates = pd.DataFrame(checks).rename(columns={"pass_": "pass"})

    for name, df in [("s12f_D_by_toxicity_tercile", a), ("s12f_tercile_composition", comp), ("s12f_distance_from_base", dist),
                     ("s12f_gate_checks", gates)]:
        df.to_csv(TAB / f"{name}.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"checks: {int(gates['pass'].sum())}/{len(gates)} pass")
    print(f"tercile edges: {a['tercile_edges'].iloc[0]}")
    print("\nmean D by tercile (low | mid | high):")
    a["cell"] = a["mean_D"].map("{:+.2f}".format)
    print(a.pivot_table(index=["label", "tercile"], columns="size", values="cell", aggfunc="first")[SIZES].to_string())
    print("\nmean |logp_M - logp_Base| (neutral / derog) and r_all:")
    dist["cell"] = dist.apply(lambda r: f"{r['mean_abs_neutral']:.2f} / {r['mean_abs_derog']:.2f}  r={r['r_all']:.3f}", axis=1)
    print(dist.pivot(index="label", columns="size", values="cell")[SIZES].to_string())


if __name__ == "__main__":
    main()
