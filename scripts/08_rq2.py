"""RQ2: H2 = SFT family (SFT, SFT-DPO, SFT-PPO, SFT-KTO); H2-KTO = Direct KTO vs SFT-KTO. Aligned vs aligned only.

D_{M1,M2}(s) = Delta_M1(s) - Delta_M2(s): Base cancels and every input is CHAT, so the components
delta_pos_{M1,M2}(s) = logp_pos_M1(s) - logp_pos_M2(s) (and delta_neg likewise) are readable here (unlike vs Base).
Pair direction fixed before running: preference model minus SFT; among DPO, PPO, KTO the later minus the earlier; Direct KTO minus SFT-KTO.

Outcome rules (fixed before running):
  H2 supported     = Friedman p < 0.05 AND at least one Bonferroni-significant pair (x6) with |mean D| clearly above 0.05 nat
                     (outside the 0.04-0.06 precision band, i.e. > 0.06); otherwise not supported.
  H2-KTO supported = Wilcoxon p < 0.05 AND |mean D| > 0.05 nat.

Reads (no re-scoring): results/delta_1.4b.parquet; results/delta_1.4b_base_nobos.parquet (check 4 only);
reports/step_07_tables/h1_tests.csv (check 1 only).
Writes:
  reports/step_08_tables/h2_friedman.csv         Friedman on Delta (k = 4, blocks = prefixes): chi2, p, Kendall's W, mean ranks, H2 outcome
  reports/step_08_tables/h2_pairwise.csv         6 pairs: paired summary of D_{M1,M2}, Wilcoxon raw + Bonferroni x6, McNemar exact (+ x6)
  reports/step_08_tables/h2kto_tests.csv         Direct KTO - SFT-KTO: D (confirmatory) + delta_pos, delta_neg (exploratory, uncorrected)
  reports/step_08_tables/components_aligned.csv  mean delta_pos / delta_neg / D per pair x category (descriptive)
  reports/step_08_tables/gate_checks.csv
  results/figures/rq2_delta_boxplots, rq2_pairwise_differences, rq2_components_aligned (.png, .pdf)
Usage: .venv/bin/python scripts/08_rq2.py
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
from probing.stats import bonferroni, bootstrap_mean_ci, friedman_summary, mcnemar_paired, paired_summary  # noqa: E402

TAB = ROOT / "reports/step_08_tables"
N_BOOT = 10_000
ALPHA = 0.05
PRECISION = 0.05        # nat; |mean| below this = precision scale (pilot float16 vs bfloat16 difference); CSV flag uses this strict cut
CLEARLY_ABOVE = 0.06    # upper edge of the 0.04-0.06 "at the precision scale" band used by the H2 outcome rule
FAMILY = ["sft", "sft_dpo", "sft_ppo", "sft_kto"]
H2_PAIRS = [("sft_dpo", "sft"), ("sft_ppo", "sft"), ("sft_kto", "sft"),
            ("sft_ppo", "sft_dpo"), ("sft_kto", "sft_dpo"), ("sft_kto", "sft_ppo")]
KTO_PAIR = ("kto", "sft_kto")
LABEL = {"sft": "SFT", "sft_dpo": "SFT-DPO", "sft_ppo": "SFT-PPO", "sft_kto": "SFT-KTO", "kto": "KTO (direct)"}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a8a"


def pair_label(m1: str, m2: str) -> str:
    return f"{LABEL[m1]} − {LABEL[m2]}"


def wide(dm: pd.DataFrame, col: str) -> pd.DataFrame:
    """Prefix x model table of `col`, rows sorted by prefix_id; asserts 500 complete rows."""
    w = dm.pivot(index="prefix_id", columns="model", values=col).sort_index()
    assert len(w) == 500 and not w.isna().any().any(), col
    return w


def friedman_table(delta: pd.DataFrame) -> pd.DataFrame:
    """Friedman across the four SFT-family models on Delta (blocks = prefixes); mean ranks in the row, not interpreted."""
    f = friedman_summary(delta[FAMILY].to_numpy())
    row = {k: v for k, v in f.items() if k != "mean_ranks"}
    row.update({f"mean_rank_{m}": r for m, r in zip(FAMILY, f["mean_ranks"])})
    return pd.DataFrame([dict(test="friedman_delta_sft_family", **row)])


def pair_row(m1: str, m2: str, delta: pd.DataFrame, ind: pd.DataFrame, seed: int) -> dict:
    """Paired summary of D_{M1,M2} and exact McNemar on I_M (reference = M2: mc_only_ref = I only in M2, mc_only_model = only in M1)."""
    d = (delta[m1] - delta[m2]).to_numpy()
    s = paired_summary(d, N_BOOT, seed)
    return dict(m1=m1, m2=m2, pair=pair_label(m1, m2), **s, exp_mean=float(np.exp(s["mean"])),
                below_precision_scale=abs(s["mean"]) < PRECISION, **mcnemar_paired(ind[m2].to_numpy(), ind[m1].to_numpy()))


def pairwise_table(delta: pd.DataFrame, ind: pd.DataFrame, friedman_p: float, seed: int) -> pd.DataFrame:
    """The 6 H2 pairs: Wilcoxon raw + Bonferroni x6 (confirmatory only if Friedman p < 0.05); McNemar raw + x6 (extra column)."""
    out = pd.DataFrame([dict(hypothesis="H2", **pair_row(m1, m2, delta, ind, seed)) for m1, m2 in H2_PAIRS])
    out["wilcoxon_p_bonf6"] = bonferroni(out["wilcoxon_p"], len(H2_PAIRS))
    out["mcnemar_p_bonf6"] = bonferroni(out["mcnemar_p"], len(H2_PAIRS))
    out["role"] = "confirmatory" if friedman_p < ALPHA else "descriptive (Friedman n.s.)"
    out["counts_for_h2_support"] = (out["wilcoxon_p_bonf6"] < ALPHA) & (out["mean"].abs() > CLEARLY_ABOVE)
    return out


def components(m1: str, m2: str, lp_pos: pd.DataFrame, lp_neg: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    """Per prefix: delta_pos_{M1,M2}, delta_neg_{M1,M2}, D_{M1,M2} (aligned vs aligned, Base not involved)."""
    return pd.DataFrame({"delta_pos": lp_pos[m1] - lp_pos[m2], "delta_neg": lp_neg[m1] - lp_neg[m2], "D": delta[m1] - delta[m2]})


def h2kto_table(delta: pd.DataFrame, ind: pd.DataFrame, comp: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Direct KTO - SFT-KTO: D (confirmatory, single test, McNemar single test) + delta_pos / delta_neg (exploratory, uncorrected)."""
    m1, m2 = KTO_PAIR
    rows = [dict(hypothesis="H2-KTO", quantity="D", role="confirmatory", **pair_row(m1, m2, delta, ind, seed))]
    for col in ["delta_pos", "delta_neg"]:
        s = paired_summary(comp[col].to_numpy(), N_BOOT, seed)
        rows.append(dict(hypothesis="H2-KTO", quantity=col, role="exploratory (uncorrected)", m1=m1, m2=m2,
                         pair=pair_label(m1, m2), **s, exp_mean=float(np.exp(s["mean"])), below_precision_scale=abs(s["mean"]) < PRECISION))
    out = pd.DataFrame(rows)
    d = out.iloc[0]
    out["h2kto_outcome"] = ["supported" if (d["wilcoxon_p"] < ALPHA and abs(d["mean"]) > PRECISION) else "not supported", None, None]
    return out


def components_table(comps: dict[tuple[str, str], pd.DataFrame], cats: pd.Series, seed: int) -> pd.DataFrame:
    """Descriptive: mean delta_pos, delta_neg, D (bootstrap CIs over prefixes) per pair, ALL and per category; share < 0."""
    rows = []
    for (m1, m2), c in comps.items():
        c = c.join(cats)
        for cat, g in [("ALL", c)] + list(c.groupby("category")):
            row = dict(m1=m1, m2=m2, pair=pair_label(m1, m2), category=cat, n_prefixes=len(g))
            for col in ["delta_pos", "delta_neg", "D"]:
                lo, hi = bootstrap_mean_ci(g[col].to_numpy(), N_BOOT, seed)
                row.update({f"mean_{col}": g[col].mean(), f"{col}_ci_lo": lo, f"{col}_ci_hi": hi})
            row.update(share_delta_pos_neg=(g["delta_pos"] < 0).mean(), share_delta_neg_neg=(g["delta_neg"] < 0).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def gate_checks(dm: pd.DataFrame, dn: pd.DataFrame, comps: dict, pw: pd.DataFrame, kto: pd.DataFrame,
                fried: pd.DataFrame) -> pd.DataFrame:
    """Checks 1-5 (check 6, a seeded re-run, is done from the shell)."""
    h1 = pd.read_csv(ROOT / "reports/step_07_tables/h1_tests.csv")
    s7 = h1[h1["reference"] == "base_bos"].set_index("model")["mean"]
    means = pd.concat([pw[["m1", "m2", "mean"]], kto[kto["quantity"] == "D"][["m1", "m2", "mean"]]])
    g1 = max(abs(r["mean"] - (s7[r["m1"]] - s7[r["m2"]])) for _, r in means.iterrows())
    dpos, dneg = wide(dm[dm["model"] != "base"], "delta_pos"), wide(dm[dm["model"] != "base"], "delta_neg")
    g2a = max(float((c["delta_pos"] - c["delta_neg"] - c["D"]).abs().max()) for c in comps.values())
    g2b = max(max(float((c["delta_pos"] - (dpos[m1] - dpos[m2])).abs().max()), float((c["delta_neg"] - (dneg[m1] - dneg[m2])).abs().max()))
              for (m1, m2), c in comps.items())
    D_wide = wide(dm[dm["model"] != "base"], "D")
    f_D = friedman_summary(D_wide[FAMILY].to_numpy())
    f_delta = fried.iloc[0]
    aligned = list(LABEL)
    main_aligned = wide(dm[dm["model"].isin(aligned)], "delta")[aligned]
    nobos_aligned = wide(dn, "delta")[aligned]
    counts = dm.groupby("model")["prefix_id"].nunique()
    chk = [("1. 7 pairwise mean D vs S7 mean D_{M1,Base} - mean D_{M2,Base} (h1_tests.csv)", g1, 1e-12),
           ("2a. delta_pos - delta_neg - D_{M1,M2}, every prefix, 7 pairs", g2a, 1e-9),
           ("2b. components from logp vs delta_pos/neg_M1 - delta_pos/neg_M2 (Base cancels), 7 pairs", g2b, 1e-9),
           ("3a. Friedman chi2 on Delta vs on D", abs(f_D["chi2"] - f_delta["chi2"]), 1e-9),
           ("3b. Friedman p on Delta vs on D", abs(f_D["p"] - f_delta["p"]), 1e-12),
           ("4. aligned Delta, Base-without-BOS file vs main file (max abs diff)", float((main_aligned - nobos_aligned).abs().max().max()), 0.0),
           ("5a. prefixes per model == 500 (models failing)", float((counts != 500).sum()), 0.0),
           ("5b. same prefix order in Delta and Base-without-BOS tables (mismatches)", float((main_aligned.index != nobos_aligned.index).sum()), 0.0)]
    out = pd.DataFrame(chk, columns=["check", "max_abs_error", "tolerance"])
    out["pass"] = out["max_abs_error"] <= out["tolerance"]
    assert out["pass"].all(), out
    return out


def _style(ax, grid_axis: str = "y") -> None:
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK)
    (ax.yaxis if grid_axis == "y" else ax.xaxis).grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))


def figure_delta(delta: pd.DataFrame, path: Path) -> None:
    """Proposal RQ2 Step 1: boxplots of Delta per model, SFT family side by side, Direct KTO set apart; mean marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = FAMILY + ["kto"]
    pos = np.array([0, 1, 2, 3, 4.7])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot([delta[m].to_numpy() for m in models], positions=pos, widths=0.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#dbe8f8", edgecolor=INK, lw=1), medianprops=dict(color=INK, lw=1.5),
               whiskerprops=dict(color=INK, lw=1), capprops=dict(color=INK, lw=1))
    means = [delta[m].mean() for m in models]
    ax.plot(pos, means, "D", ms=6, color=BLUE, markeredgecolor="white", label="mean")
    for x, m in zip(pos, means):
        ax.annotate(f"{m:.2f}", (x + 0.3, m), va="center", fontsize=9, color=INK)
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.axvline(3.85, color=MUTED, lw=0.8, ls=":")
    ax.text(1.5, 1.0, "H2: SFT family", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(4.7, 1.0, "H2-KTO: vs SFT-KTO", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(pos, [LABEL[m] for m in models])
    ax.set_xlim(-0.6, 5.4)
    ax.set_ylabel("Δ = mean logP(neutral) − mean logP(derogatory) (nat)")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    _style(ax)
    note = ("Δ per prefix (n = 500 per model; box = median, IQR, 1.5 IQR; outliers not drawn). Raw Δ carries the word-length effect; "
            "only differences between models (same words, all CHAT) are read.")
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(note, 125)), fontsize=8, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    _save(fig, path)
    plt.close(fig)


def figure_pairs(pw: pd.DataFrame, kto: pd.DataFrame, path: Path) -> None:
    """Forest plot: mean D_{M1,M2} [95% bootstrap CI] for the 6 H2 pairs and H2-KTO; shaded +-0.05 nat precision band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = pd.concat([pw, kto[kto["quantity"] == "D"]], ignore_index=True)
    y = np.array([6.6, 5.6, 4.6, 3.6, 2.6, 1.6, 0.0])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axvspan(-PRECISION, PRECISION, color="#eeeeee", lw=0, label="±0.05 nat (precision scale)")
    ax.axvline(0, color=MUTED, lw=1)
    ax.errorbar(rows["mean"], y, xerr=[rows["mean"] - rows["ci_boot_lo"], rows["ci_boot_hi"] - rows["mean"]], fmt="o", ms=6,
                color=BLUE, ecolor=INK, elinewidth=1.2, capsize=4, markeredgecolor="white")
    for yi, (_, r) in zip(y, rows.iterrows()):
        p = r["wilcoxon_p_bonf6"] if r["hypothesis"] == "H2" else r["wilcoxon_p"]
        tag = "p(Bonf.×6)" if r["hypothesis"] == "H2" else "p"
        ax.annotate(f"{r['mean']:+.3f}   {tag} = {p:.2g}", (r["ci_boot_hi"] + 0.02, yi), va="center", fontsize=8.5, color=INK)
    ax.axhline(0.8, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(y, rows["pair"])
    ax.set_xlim(rows["ci_boot_lo"].min() - 0.05, rows["ci_boot_hi"].max() + 0.33)
    ax.set_xlabel("mean D_{M1,M2} = Δ_M1 − Δ_M2 (nat), 95% bootstrap CI, n = 500")
    ax.set_title("RQ2: paired differences between aligned models (H2: six SFT-family pairs; below: H2-KTO)", fontsize=10, color=INK)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _style(ax, grid_axis="x")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def figure_components(comp: pd.DataFrame, path: Path) -> None:
    """Grouped bars: mean delta_pos / delta_neg of M1 vs M2 (aligned vs aligned, all prefixes), 95% bootstrap CIs; D above each pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = comp[comp["category"] == "ALL"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    pos, w = np.array([0, 1, 2, 3.4]), 0.36
    for off, col, color, lab in [(-w / 2, "delta_pos", BLUE, "δ⁺ (neutral pool): M1 − M2"),
                                 (w / 2, "delta_neg", ORANGE, "δ⁻ (derogatory pool): M1 − M2")]:
        mean = c[f"mean_{col}"]
        ax.bar(pos + off, mean, width=w, color=color, edgecolor="white", lw=2, label=lab)
        ax.errorbar(pos + off, mean, yerr=[mean - c[f"{col}_ci_lo"], c[f"{col}_ci_hi"] - mean], fmt="none", ecolor=INK,
                    elinewidth=1, capsize=3)
    top = max(c["delta_pos_ci_hi"].max(), c["delta_neg_ci_hi"].max())
    for x, d in zip(pos, c["mean_D"]):
        ax.annotate(f"D = {d:+.2f}", (x, top + 0.04), ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(2.7, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(pos, [p.replace(" − ", "\nvs ") for p in c["pair"]])
    ax.set_ylabel("mean change in pool log-prob, M1 − M2 (nat)")
    ax.set_ylim(min(c["delta_pos_ci_lo"].min(), c["delta_neg_ci_lo"].min(), 0) - 0.1, top + 0.18)
    ax.set_title("Components between aligned models (500 prefixes): SFT family vs SFT; Direct KTO vs SFT-KTO", fontsize=10, color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    _style(ax)
    note = ("Aligned vs aligned: Base cancels and all inputs are CHAT, so the components are readable. Negative = M1 gives the pool lower "
            "log-prob than M2. D = δ⁺ − δ⁻. KTO components are exploratory (uncorrected tests); family components are descriptive.")
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(note, 125)), fontsize=8, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, path)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    res = ROOT / cfg["paths"]["results"]
    seed = int(cfg["project"]["seed"])
    TAB.mkdir(parents=True, exist_ok=True)
    (res / "figures").mkdir(exist_ok=True)
    dm = pd.read_parquet(res / "delta_1.4b.parquet")
    dn = pd.read_parquet(res / "delta_1.4b_base_nobos.parquet")
    al = dm[dm["model"] != "base"]
    delta, ind = wide(al, "delta"), wide(al, "I_M").astype(bool)
    lp_pos, lp_neg = wide(al, "logp_pos"), wide(al, "logp_neg")
    cats = al.drop_duplicates("prefix_id").set_index("prefix_id")["category"]

    fried = friedman_table(delta)
    pw = pairwise_table(delta, ind, float(fried["p"].iloc[0]), seed)
    fried["h2_outcome"] = "supported" if (fried["p"].iloc[0] < ALPHA and pw["counts_for_h2_support"].any()) else "not supported"
    comps = {pair: components(*pair, lp_pos, lp_neg, delta) for pair in H2_PAIRS[:3] + [KTO_PAIR]}
    comps_all = {pair: components(*pair, lp_pos, lp_neg, delta) for pair in H2_PAIRS + [KTO_PAIR]}
    kto = h2kto_table(delta, ind, comps[KTO_PAIR], seed)
    comp = components_table(comps, cats, seed)
    gates = gate_checks(dm, dn, comps_all, pw, kto, fried)
    for name, df in [("h2_friedman", fried), ("h2_pairwise", pw), ("h2kto_tests", kto), ("components_aligned", comp),
                     ("gate_checks", gates)]:
        df.to_csv(TAB / f"{name}.csv", index=False)
    figure_delta(delta, res / "figures/rq2_delta_boxplots")
    figure_pairs(pw, kto, res / "figures/rq2_pairwise_differences")
    figure_components(comp, res / "figures/rq2_components_aligned")

    f = fried.iloc[0]
    print(f"checks: {int(gates['pass'].sum())}/{len(gates)} pass")
    print(f"Friedman (k=4, n=500): chi2 = {f['chi2']:.2f}, p = {f['p']:.3g}, Kendall's W = {f['kendall_w']:.4f} -> H2 {f['h2_outcome']}")
    cols = ["pair", "mean", "ci_boot_lo", "ci_boot_hi", "d_z", "share_pos", "wilcoxon_p", "wilcoxon_p_bonf6", "mcnemar_p", "mcnemar_p_bonf6"]
    print(pd.concat([pw, kto[kto["quantity"] == "D"]])[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"H2-KTO: {kto['h2kto_outcome'].iloc[0]}; exploratory components (KTO - SFT-KTO):")
    print(kto.iloc[1:][["quantity", "mean", "ci_boot_lo", "ci_boot_hi", "wilcoxon_p"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
