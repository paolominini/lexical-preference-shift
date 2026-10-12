"""Toxicity covariate: Detoxify `unbiased` on the truncated prefix of all 500 prefixes.

Writes, never overwriting:
  results/toxicity.parquet             prefix_id, category, source_polarity, n_chars, 7 Detoxify scores, detoxify_model
  reports/step_06_tables/*.csv         coverage, per category x polarity, overlap, correlations, extremes (derogatory words masked)
Usage: .venv/bin/python scripts/06_toxicity.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probing.config import load_config  # noqa: E402
from probing.toxicity import LABELS, load_detoxify, toxicity_table  # noqa: E402

TAB = ROOT / "reports/step_06_tables"
QS = [0.05, 0.25, 0.5, 0.75, 0.95]


def write_new(df: pd.DataFrame, path: Path) -> None:
    """Never overwrite: an existing file is kept after checking it equals the recomputed one (tolerance for BLAS/threads noise)."""
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), df, check_exact=False, rtol=1e-4, atol=1e-5)
        print(f"{path.name} already exists and equals the recomputed table (rtol 1e-4): kept")
        return
    df.to_parquet(path, index=False)


def describe(g: pd.Series) -> dict:
    return dict(n=len(g), mean=g.mean(), sd=g.std(), **{f"q{int(q * 100):02d}": g.quantile(q) for q in QS}, max=g.max(),
                share_ge_05=(g >= 0.5).mean(), share_ge_01=(g >= 0.1).mean())


def distribution_tables(tox: pd.DataFrame, col: str) -> pd.DataFrame:
    """Distribution of `col` overall, by source polarity, by category, and by category x polarity."""
    rows = [dict(group="ALL", category="ALL", polarity="ALL", **describe(tox[col]))]
    for pol, g in tox.groupby("source_polarity"):
        rows.append(dict(group="polarity", category="ALL", polarity=pol, **describe(g[col])))
    for cat, g in tox.groupby("category"):
        rows.append(dict(group="category", category=cat, polarity="ALL", **describe(g[col])))
    for (cat, pol), g in tox.groupby(["category", "source_polarity"]):
        rows.append(dict(group="category x polarity", category=cat, polarity=pol, **describe(g[col])))
    return pd.DataFrame(rows)


def polarity_association(tox: pd.DataFrame, col: str) -> pd.DataFrame:
    """Toxicity vs source polarity (NOT SAFE = 1): Spearman and point-biserial, Mann-Whitney AUC; overall and within category."""
    rows = []
    for cat, g in [("ALL", tox)] + list(tox.groupby("category")):
        y = (g["source_polarity"] == "NOT SAFE").astype(int)
        rho, p_rho = stats.spearmanr(g[col], y)
        r, p_r = stats.pointbiserialr(y, g[col])
        u = stats.mannwhitneyu(g.loc[y == 1, col], g.loc[y == 0, col], alternative="two-sided")
        rows.append(dict(category=cat, n=len(g), spearman_rho=rho, spearman_p=p_rho, pointbiserial_r=r, pointbiserial_p=p_r,
                         auc_notsafe_gt_safe=u.statistic / ((y == 1).sum() * (y == 0).sum()), mannwhitney_p=u.pvalue,
                         mean_notsafe=g.loc[y == 1, col].mean(), mean_safe=g.loc[y == 0, col].mean()))
    return pd.DataFrame(rows)


def category_overlap(tox: pd.DataFrame, col: str) -> pd.DataFrame:
    """Pairwise overlap between categories: AUC (P[X_a > X_b], ties 1/2), KS statistic, median difference; plus Kruskal-Wallis."""
    cats = sorted(tox["category"].unique())
    rows = []
    for a, b in itertools.combinations(cats, 2):
        x, y = tox.loc[tox["category"] == a, col], tox.loc[tox["category"] == b, col]
        rows.append(dict(cat_a=a, cat_b=b, auc_a_gt_b=stats.mannwhitneyu(x, y).statistic / (len(x) * len(y)),
                         ks=stats.ks_2samp(x, y).statistic, median_diff=x.median() - y.median()))
    kw = stats.kruskal(*[tox.loc[tox["category"] == c, col] for c in cats])
    out = pd.DataFrame(rows)
    out.attrs["kruskal"] = (kw.statistic, kw.pvalue)
    return out


def length_association(tox: pd.DataFrame, col: str) -> dict:
    rho, p = stats.spearmanr(tox[col], tox["n_chars"])
    return dict(spearman_rho=rho, p=p)


def figure(tox: pd.DataFrame, path: Path) -> None:
    """Toxicity of the truncated prefix: histogram by source polarity (left), strip plot by category and polarity (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    bins = np.linspace(0, 1, 21)
    for pol, c in [("SAFE", "#4C78A8"), ("NOT SAFE", "#E4572E")]:
        a.hist(tox.loc[tox["source_polarity"] == pol, "toxicity"], bins=bins, alpha=0.6, color=c, label=pol)
    a.set(xlabel="Detoxify toxicity of the truncated prefix", ylabel="prefixes", title="By source polarity")
    a.legend()
    cats = sorted(tox["category"].unique())
    rng = np.random.default_rng(42)
    for i, cat in enumerate(cats):
        for pol, c, off in [("SAFE", "#4C78A8", -0.18), ("NOT SAFE", "#E4572E", 0.18)]:
            y = tox.loc[(tox["category"] == cat) & (tox["source_polarity"] == pol), "toxicity"]
            b.scatter(y, i + off + rng.uniform(-0.1, 0.1, len(y)), s=9, color=c, alpha=0.6)
            b.plot([y.median()] * 2, [i + off - 0.15, i + off + 0.15], color="black", lw=1.5)
    b.set(yticks=range(len(cats)), yticklabels=[c.split(",")[0].split(" - ")[0] for c in cats], xlabel="toxicity (black bar = median)",
          title="By category and source polarity")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))


def main() -> None:
    cfg = load_config()
    res = ROOT / cfg["paths"]["results"]
    TAB.mkdir(parents=True, exist_ok=True)
    prefixes = pd.read_csv(ROOT / cfg["paths"]["prefixes"])
    assert prefixes["final_id"].is_unique and len(prefixes) == 500

    model = load_detoxify("cpu")
    tox = toxicity_table(prefixes, model)
    # Gate: coverage 100% (500 prefixes, every score finite and in [0, 1]).
    assert len(tox) == 500 and set(tox["prefix_id"]) == set(prefixes["final_id"])
    assert np.isfinite(tox[LABELS].to_numpy()).all() and ((tox[LABELS] >= 0) & (tox[LABELS] <= 1)).all().all()
    write_new(tox, res / "toxicity.parquet")

    cov = pd.DataFrame([dict(check="prefixes scored / 500", value=len(tox)), dict(check="NaN or non-finite scores", value=int(tox[LABELS].isna().sum().sum())),
                        dict(check="scores outside [0, 1]", value=int(((tox[LABELS] < 0) | (tox[LABELS] > 1)).sum().sum())),
                        dict(check="unique prefix_id", value=int(tox["prefix_id"].nunique()))])
    cov.to_csv(TAB / "coverage.csv", index=False)
    dist = pd.concat([distribution_tables(tox, c).assign(score=c) for c in ["toxicity", "identity_attack"]])
    dist.to_csv(TAB / "distribution_by_category_polarity.csv", index=False)
    assoc = pd.concat([polarity_association(tox, c).assign(score=c) for c in ["toxicity", "identity_attack"]])
    assoc.to_csv(TAB / "association_with_polarity.csv", index=False)
    ov = category_overlap(tox, "toxicity")
    ov.to_csv(TAB / "category_overlap_toxicity.csv", index=False)
    ov2 = category_overlap(tox, "identity_attack")
    ov2.to_csv(TAB / "category_overlap_identity_attack.csv", index=False)
    corr = tox[LABELS + ["n_chars"]].corr(method="spearman")
    corr.to_csv(TAB / "spearman_between_scores.csv")

    figure(tox, ROOT / cfg["paths"]["results"] / "figures/toxicity_distribution")

    pd.set_option("display.width", 220)
    print(cov.to_string(index=False))
    show = dist[(dist["score"] == "toxicity") & dist["group"].isin(["ALL", "polarity", "category"])]
    print("\ntoxicity by group:\n", show[["category", "polarity", "n", "mean", "q25", "q50", "q75", "q95", "share_ge_05"]].round(3).to_string(index=False))
    print("\nassociation with polarity (toxicity):\n", assoc[assoc["score"] == "toxicity"][["category", "spearman_rho", "spearman_p", "auc_notsafe_gt_safe", "mean_notsafe", "mean_safe"]].round(3).to_string(index=False))
    print("\nKruskal-Wallis toxicity across categories: H=%.2f p=%.3g" % ov.attrs["kruskal"])
    print("length vs toxicity:", length_association(tox, "toxicity"))


if __name__ == "__main__":
    main()
