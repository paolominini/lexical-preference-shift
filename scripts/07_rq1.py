"""RQ1 / H1: each of the five post-trained checkpoints vs Base, on prefix-level D (n = 500).

Reads (no re-scoring): results/delta_1.4b.parquet, results/delta_1.4b_base_nobos.parquet, reports/step_05_tables/per_word_shift.csv.
Writes:
  reports/step_07_tables/h1_tests.csv              Wilcoxon (two-sided), d_z, bootstrap CI, share D > 0, sign test, McNemar (exact);
                                                   reference = Base with BOS (main) and Base without BOS (robustness check d)
  reports/step_07_tables/components_vs_base.csv    mean delta_pos / delta_neg / D per model x category (descriptive, as-deployed)
  reports/step_07_tables/length_check_by_pool.csv  per model x pool with varying n_tokens: mean shift of shorter vs longer words
  reports/step_07_tables/length_check_summary.csv  within-pool slope of shift on n_tokens and the part of D it would imply
  reports/step_07_tables/gate_checks.csv
  results/figures/rq1_D_distribution.{png,pdf}, results/figures/rq1_components_vs_base.{png,pdf}
Usage: .venv/bin/python scripts/07_rq1.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probing.config import load_config  # noqa: E402
from probing.derived import NEG, POS  # noqa: E402
from probing.stats import (bootstrap_mean_ci, mcnemar_paired, paired_summary, varying_pools,  # noqa: E402
                           within_pool_length_slope)

TAB = ROOT / "reports/step_07_tables"
N_BOOT = 10_000
POOL = ["category", "polarity_word"]
TW = "Transgender Woman"
LABEL = {"sft": "SFT", "sft_dpo": "SFT-DPO", "sft_ppo": "SFT-PPO", "sft_kto": "SFT-KTO", "kto": "KTO (direct)"}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a8a"
CAPTION = ("Components vs Base absorb the RAW->CHAT change of input format (Base RAW, post-trained CHAT); they are not "
           "promotion/suppression effects. Only D = delta_pos - delta_neg cancels it. As-deployed comparison.")


def h1_table(dm: pd.DataFrame, dn: pd.DataFrame, aligned: list[str], seed: int) -> pd.DataFrame:
    """One row per (reference, model): paired summary of D and McNemar on I_M (model vs the reference Base)."""
    base_I = dm[dm["model"] == "base"].set_index("prefix_id")["I_M"]
    rows = []
    for ref, frame in [("base_bos", dm), ("base_nobos", dn)]:
        for model in aligned:
            g = frame[frame["model"] == model].set_index("prefix_id").sort_index()
            ref_I = base_I.loc[g.index] if ref == "base_bos" else g["delta_ref"] > 0
            row = dict(reference=ref, model=model, **paired_summary(g["D"].to_numpy(), N_BOOT, seed),
                       **mcnemar_paired(ref_I.to_numpy(), (g["delta"] > 0).to_numpy()))
            rows.append(row)
    out = pd.DataFrame(rows)
    out["wilcoxon_p_bonf5"] = np.minimum(1.0, out["wilcoxon_p"] * len(aligned))
    return out


def components_table(dm: pd.DataFrame, aligned: list[str], seed: int) -> pd.DataFrame:
    """Descriptive: mean delta_pos, delta_neg, D (with bootstrap CIs over prefixes) per model x category and ALL."""
    rows = []
    for model in aligned:
        g_model = dm[dm["model"] == model]
        for cat, g in [("ALL", g_model)] + list(g_model.groupby("category")):
            row = dict(model=model, category=cat, n_prefixes=len(g))
            for col in ["delta_pos", "delta_neg", "D"]:
                lo, hi = bootstrap_mean_ci(g[col].to_numpy(), N_BOOT, seed)
                row.update({f"mean_{col}": g[col].mean(), f"{col}_ci_lo": lo, f"{col}_ci_hi": hi})
            row.update(share_delta_pos_neg=(g["delta_pos"] < 0).mean(), share_delta_neg_neg=(g["delta_neg"] < 0).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def length_by_pool(pw: pd.DataFrame, aligned: list[str]) -> pd.DataFrame:
    """Per model x pool with varying n_tokens: mean shift of the shortest vs the longest words of the pool."""
    vary = varying_pools(pw[pw["model"] == aligned[0]], POOL)
    rows = []
    for (model, cat, pol), g in pw.groupby(["model", "category", "polarity_word"], observed=True):
        if not vary.loc[(cat, pol)]:
            continue
        lo_n, hi_n = g["n_tokens"].min(), g["n_tokens"].max()
        short, long_ = g[g["n_tokens"] == lo_n], g[g["n_tokens"] == hi_n]
        rows.append(dict(model=model, category=cat, pool=pol, n_tokens_short=lo_n, n_tokens_long=hi_n,
                         n_words_short=len(short), n_words_long=len(long_), mean_shift_short=short["mean_shift"].mean(),
                         mean_shift_long=long_["mean_shift"].mean(),
                         long_minus_short=long_["mean_shift"].mean() - short["mean_shift"].mean()))
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], aligned, ordered=True)
    return out.sort_values(["model", "category", "pool"], ascending=[True, True, False]).reset_index(drop=True)


def length_summary(pw: pd.DataFrame, comp: pd.DataFrame, aligned: list[str]) -> pd.DataFrame:
    """Per model: within-pool slope (nat per extra token) and the D it implies from the pools' length gap, vs observed D.

    implied_D(c) = slope x (mean n_tokens of the neutral pool - mean n_tokens of the derogatory pool); ALL = mean over the five
    categories (100 prefixes each, so it is on the same footing as the overall mean D). Variant: slope without the Transgender
    Woman derogatory pool, whose only 3-token word is the one word that rises vs Base (05 tables), so it dominates the pooled slope.
    """
    rows = []
    for model in aligned:
        w = pw[pw["model"] == model]
        fit = within_pool_length_slope(w, POOL)
        fit_x = within_pool_length_slope(w[~((w["category"] == TW) & (w["polarity_word"] == NEG))], POOL)
        n_tok = w.groupby(POOL)["n_tokens"].mean().unstack("polarity_word")
        gap = n_tok[POS] - n_tok[NEG]
        obs = comp[comp["model"] == model].set_index("category")["mean_D"]
        for cat in list(gap.index) + ["ALL"]:
            g_ = gap.mean() if cat == "ALL" else gap[cat]
            rows.append(dict(model=model, category=cat, slope_nat_per_token=fit["slope"], n_words_used=fit["n_words"],
                             n_pools_used=fit["n_pools"], n_pools_longer_lower=fit["n_pools_longer_lower"],
                             n_tokens_gap_neutral_minus_derog=g_, implied_D=fit["slope"] * g_, observed_mean_D=obs[cat],
                             implied_share_of_D=fit["slope"] * g_ / obs[cat],
                             slope_excl_tw_derog=fit_x["slope"], implied_D_excl_tw_derog=fit_x["slope"] * g_,
                             implied_share_excl_tw_derog=fit_x["slope"] * g_ / obs[cat]))
    return pd.DataFrame(rows)


def gate_checks(dm: pd.DataFrame, pw: pd.DataFrame, comp: pd.DataFrame, aligned: list[str]) -> pd.DataFrame:
    """Inputs equal 05: n per model, mean D and share D > 0 vs the 05 summary; per-word contributions vs the delta parquet."""
    s5 = pd.read_csv(ROOT / "reports/step_05_tables/summary_model_category.csv")
    counts = dm[dm["model"].isin(aligned)].groupby("model")["prefix_id"].nunique()
    a = comp[["model", "category", "mean_D"]].merge(s5[["model", "category", "mean_D", "share_D_pos"]], on=["model", "category"],
                                                    suffixes=("", "_s5"), validate="one_to_one")
    share = dm[dm["model"].isin(aligned)].assign(pos=lambda x: x["D"] > 0).groupby("model")["pos"].mean()
    s5_all = s5[s5["category"] == "ALL"].set_index("model")["share_D_pos"]
    contrib = pw.groupby(["model", "category", "polarity_word"], observed=True)["contribution_to_D"].sum().unstack("polarity_word")
    c = comp[comp["category"] != "ALL"].set_index(["model", "category"])
    c = c.join(contrib, how="inner")
    chk = [("prefixes per model == 500", float((counts != 500).sum()), 0.0),
           ("mean D vs S5 summary, 30 cells", float((a["mean_D"] - a["mean_D_s5"]).abs().max()), 1e-12),
           ("share D > 0 vs S5 summary, 5 models", float((share - s5_all.loc[share.index]).abs().max()), 1e-12),
           ("sum of contribution_to_D over both pools vs mean D, 25 cells", float((c[POS] + c[NEG] - c["mean_D"]).abs().max()), 1e-9),
           ("neutral contributions vs mean delta_pos, 25 cells", float((c[POS] - c["mean_delta_pos"]).abs().max()), 1e-9),
           ("derogatory contributions vs -mean delta_neg, 25 cells", float((c[NEG] + c["mean_delta_neg"]).abs().max()), 1e-9)]
    out = pd.DataFrame(chk, columns=["check", "max_abs_error", "tolerance"])
    out["pass"] = out["max_abs_error"] <= out["tolerance"]
    assert out["pass"].all(), out
    assert len(c) == 25
    return out


def _style(ax) -> None:
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)


def figure_d(dm: pd.DataFrame, h1: pd.DataFrame, aligned: list[str], path: Path) -> None:
    """D per model: violin + box (n = 500 each), zero line, mean with bootstrap 95% CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = [dm.loc[dm["model"] == m, "D"].to_numpy() for m in aligned]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    pos = np.arange(len(aligned))
    vp = ax.violinplot(data, positions=pos, widths=0.8, showextrema=False)
    for b in vp["bodies"]:
        b.set(facecolor=BLUE, edgecolor="none", alpha=0.25)
    ax.boxplot(data, positions=pos, widths=0.18, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor=INK, lw=1), medianprops=dict(color=INK, lw=1.5),
               whiskerprops=dict(color=INK, lw=1), capprops=dict(color=INK, lw=1))
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    r = h1[h1["reference"] == "base_bos"].set_index("model").loc[aligned]
    ax.errorbar(pos + 0.27, r["mean"], yerr=[r["mean"] - r["ci_boot_lo"], r["ci_boot_hi"] - r["mean"]], fmt="o", ms=4,
                color=BLUE, ecolor=INK, elinewidth=1.2, capsize=4, label="mean, 95% bootstrap CI")
    for x, m in zip(pos, r["mean"]):
        ax.annotate(f"{m:+.2f}", (x + 0.33, m), va="center", fontsize=9, color=INK)
    ax.set_xticks(pos, [LABEL[m] for m in aligned])
    ax.set_xlim(-0.6, len(aligned) - 0.2)
    ax.set_ylabel("D = Δ_M − Δ_Base (nat)")
    ax.set_title("D per post-trained model vs Base (500 prefixes each; box = median, IQR, 1.5 IQR)", fontsize=10, color=INK)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def figure_components(comp: pd.DataFrame, aligned: list[str], path: Path) -> None:
    """Grouped bars: mean delta_pos and delta_neg per model (all 500 prefixes), 95% bootstrap CIs; caveat in the footer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = comp[comp["category"] == "ALL"].set_index("model").loc[aligned]
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    pos, w = np.arange(len(aligned)), 0.36
    for off, col, color, lab in [(-w / 2, "delta_pos", BLUE, "δ⁺ (neutral pool) vs Base"),
                                 (w / 2, "delta_neg", ORANGE, "δ⁻ (derogatory pool) vs Base")]:
        mean = c[f"mean_{col}"]
        ax.bar(pos + off, mean, width=w, color=color, edgecolor="white", lw=2, label=lab)
        ax.errorbar(pos + off, mean, yerr=[mean - c[f"{col}_ci_lo"], c[f"{col}_ci_hi"] - mean], fmt="none", ecolor=INK,
                    elinewidth=1, capsize=3)
    for x, d in zip(pos, c["mean_D"]):
        ax.annotate(f"D = {d:+.2f}", (x, 0.04), ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(pos, [LABEL[m] for m in aligned])
    ax.set_ylabel("mean change in log-prob vs Base (nat)")
    ax.set_ylim(min(c["delta_neg_ci_lo"].min(), c["delta_pos_ci_lo"].min()) * 1.15, 0.2)
    ax.set_title("Asymmetric attenuation, as-deployed: mean δ⁺ and δ⁻ vs Base (500 prefixes)", fontsize=10, color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    _style(ax)
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(CAPTION, 120)), fontsize=8, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    res = ROOT / cfg["paths"]["results"]
    seed = int(cfg["project"]["seed"])
    aligned = list(cfg["models"]["1.4b"])[1:]
    TAB.mkdir(parents=True, exist_ok=True)
    (res / "figures").mkdir(exist_ok=True)
    dm = pd.read_parquet(res / "delta_1.4b.parquet")
    dn = pd.read_parquet(res / "delta_1.4b_base_nobos.parquet")
    pw = pd.read_csv(ROOT / "reports/step_05_tables/per_word_shift.csv")

    h1 = h1_table(dm, dn, aligned, seed)
    comp = components_table(dm, aligned, seed)
    gates = gate_checks(dm, pw, comp, aligned)
    by_pool = length_by_pool(pw, aligned)
    lsum = length_summary(pw, comp, aligned)
    for name, df in [("h1_tests", h1), ("components_vs_base", comp), ("length_check_by_pool", by_pool),
                     ("length_check_summary", lsum), ("gate_checks", gates)]:
        df.to_csv(TAB / f"{name}.csv", index=False)
    figure_d(dm[dm["model"].isin(aligned)], h1, aligned, res / "figures/rq1_D_distribution")
    figure_components(comp, aligned, res / "figures/rq1_components_vs_base")

    one = pw[pw["model"] == aligned[0]]
    vary = varying_pools(one, POOL)
    dist_strings = one.drop_duplicates("word")["n_tokens"].value_counts().sort_index().to_dict()
    modal = max(dist_strings, key=dist_strings.get)
    print(f"checks: {int(gates['pass'].sum())}/{len(gates)} pass")
    print(f"pools with varying n_tokens: {int(vary.sum())} of {len(vary)}; n_tokens over {len(one['word'].unique())} distinct strings: "
          f"{dist_strings}; off the modal value ({modal}): {sum(v for k, v in dist_strings.items() if k != modal)} strings, "
          f"{int((one['n_tokens'] != modal).sum())} of {len(one)} (category, word) pairs")
    cols = ["model", "mean", "ci_boot_lo", "ci_boot_hi", "d_z", "share_pos", "wilcoxon_p", "mcnemar_p", "share_I_ref", "share_I_model"]
    pd.set_option("display.width", 200)
    for ref in ["base_bos", "base_nobos"]:
        print(f"\nH1, reference {ref}:")
        print(h1[h1["reference"] == ref][cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\nlength check (ALL):")
    print(lsum[lsum["category"] == "ALL"][["model", "slope_nat_per_token", "n_pools_longer_lower", "implied_D", "slope_excl_tw_derog",
                                           "implied_D_excl_tw_derog", "observed_mean_D"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
