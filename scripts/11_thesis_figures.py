"""Thesis versions of the figures of scripts 07-09b2 (display labels, no in-figure titles or notes, vector PDF).

Same input files as scripts 07, 08, 09 and 09b2 (nothing is recomputed except what those scripts also compute while drawing:
per-category least-squares lines in the toxicity panels). Titles and explanatory notes go into the LaTeX captions.
PDFs carry no creation date, so a rerun gives byte-identical files.
Reads: results/delta_1.4b.parquet, results/toxicity.parquet, reports/step_07_tables/{h1_tests,components_vs_base}.csv,
       reports/step_08_tables/{h2_pairwise,h2kto_tests,components_aligned}.csv, reports/step_09_tables/h3_shift_by_category.csv,
       reports/step_09_tables/{s9b2_vs_shift,s9b2_rho,s9b2_preference}.csv
Writes: reports/step_11_thesis/figures/*.pdf, an identical copy in thesis/figures/ (so thesis/ compiles on its own),
        reports/step_11_thesis/gate_checks_figures.csv
Usage: .venv/bin/python scripts/11_thesis_figures.py
"""
from __future__ import annotations

import filecmp
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probing.config import load_config  # noqa: E402
from probing.latex import MODEL_LABELS, display_labels, ordered_categories  # noqa: E402

OUT = ROOT / "reports/step_11_thesis"
FIGS = OUT / "figures"
THESIS_FIGS = ROOT / "thesis/figures"
S07, S08, S09 = (ROOT / f"reports/step_{n}_tables" for n in ["07", "08", "09"])
FAMILY = ["sft", "sft_dpo", "sft_ppo", "sft_kto"]
ALIGNED = FAMILY + ["kto"]
PRECISION = 0.05
WIDTH = 6.0             # inches, about the text width of the thesis page; include with width=\textwidth
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#222222", "#8a8a8a", "#e6e6e6"
CAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots 1-5 in display-label order (validated)
ABBR = {"Black Person": "BP", "Homosexual Man": "HM", "Homosexual Woman": "HW", "Sex Worker": "SW", "Transgender Woman": "TW"}
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "pdf.fonttype": 42, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def style(ax, grid_axis: str = "y") -> None:
    """Recessive axes: no top/right spine, light grid behind the data."""
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    (ax.yaxis if grid_axis == "y" else ax.xaxis).grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str, texts: dict[str, list[str]]) -> None:
    """Write a vector PDF without a creation date, copy it to thesis/figures, record every text drawn in the figure (label check)."""
    fig.savefig(FIGS / f"{name}.pdf", metadata={"CreationDate": None}, bbox_inches="tight")
    shutil.copyfile(FIGS / f"{name}.pdf", THESIS_FIGS / f"{name}.pdf")
    texts[name] = [t.get_text() for t in fig.findobj(Text) if t.get_text()]
    plt.close(fig)


def fig_rq1_d(dm: pd.DataFrame, texts: dict) -> None:
    """D per post-trained model: violin + box, zero line, mean with bootstrap 95% CI (h1_tests.csv)."""
    h1 = pd.read_csv(S07 / "h1_tests.csv").query("reference == 'base_bos'").set_index("model").loc[ALIGNED]
    data = [dm.loc[dm["model"] == m, "D"].to_numpy() for m in ALIGNED]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    pos = np.arange(len(ALIGNED))
    vp = ax.violinplot(data, positions=pos, widths=0.8, showextrema=False)
    for b in vp["bodies"]:
        b.set(facecolor=BLUE, edgecolor="none", alpha=0.25)
    ax.boxplot(data, positions=pos, widths=0.18, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor=INK, lw=1), medianprops=dict(color=INK, lw=1.5),
               whiskerprops=dict(color=INK, lw=1), capprops=dict(color=INK, lw=1))
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.errorbar(pos + 0.27, h1["mean"], yerr=[h1["mean"] - h1["ci_boot_lo"], h1["ci_boot_hi"] - h1["mean"]], fmt="o", ms=4,
                color=BLUE, ecolor=INK, elinewidth=1.2, capsize=3, label="mean, 95% bootstrap CI")
    for x, m in zip(pos, h1["mean"]):
        ax.annotate(f"{m:+.2f}", (x + 0.33, m), va="center", fontsize=8, color=INK)
    ax.set_xticks(pos, [MODEL_LABELS[m] for m in ALIGNED])
    ax.set_xlim(-0.6, len(ALIGNED) - 0.2)
    ax.set_ylabel(r"$D_M(c \mid S)$ (nats)")
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    save(fig, "fig_rq1_D_distribution", texts)


def fig_rq1_components(texts: dict) -> None:
    """Mean delta_pos / delta_neg vs Base per model (all prefixes), 95% CIs, D above each pair (components_vs_base.csv)."""
    c = pd.read_csv(S07 / "components_vs_base.csv").query("category == 'ALL'").set_index("model").loc[ALIGNED]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    pos, w = np.arange(len(ALIGNED)), 0.36
    for off, col, color, lab in [(-w / 2, "delta_pos", BLUE, r"$\delta^{+}_M$ (neutral pool) vs Base"),
                                 (w / 2, "delta_neg", ORANGE, r"$\delta^{-}_M$ (derogatory pool) vs Base")]:
        mean = c[f"mean_{col}"]
        ax.bar(pos + off, mean, width=w, color=color, edgecolor="white", lw=2, label=lab)
        ax.errorbar(pos + off, mean, yerr=[mean - c[f"{col}_ci_lo"], c[f"{col}_ci_hi"] - mean], fmt="none", ecolor=INK,
                    elinewidth=1, capsize=3)
    for x, d in zip(pos, c["mean_D"]):
        ax.annotate(rf"$\bar{{D}}_M = {d:+.2f}$", (x, 0.04), ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(pos, [MODEL_LABELS[m] for m in ALIGNED])
    ax.set_ylabel("mean change in log-prob vs Base (nats)")
    ax.set_ylim(min(c["delta_neg_ci_lo"].min(), c["delta_pos_ci_lo"].min()) * 1.15, 0.2)
    ax.legend(frameon=False, loc="lower left")
    style(ax)
    save(fig, "fig_rq1_components_vs_base", texts)


def fig_rq2_delta(dm: pd.DataFrame, texts: dict) -> None:
    """Boxplots of Delta per post-trained model (SFT family, Direct KTO set apart), mean marker."""
    models = ALIGNED
    pos = np.array([0, 1, 2, 3, 4.7])
    data = [dm.loc[dm["model"] == m, "delta"].to_numpy() for m in models]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    ax.boxplot(data, positions=pos, widths=0.5, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#dbe8f8", edgecolor=INK, lw=1), medianprops=dict(color=INK, lw=1.5),
               whiskerprops=dict(color=INK, lw=1), capprops=dict(color=INK, lw=1))
    means = [d.mean() for d in data]
    ax.plot(pos, means, "D", ms=6, color=BLUE, markeredgecolor="white", label="mean")
    for x, m in zip(pos, means):
        ax.annotate(f"{m:.2f}", (x + 0.3, m), va="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.axvline(3.85, color=MUTED, lw=0.8, ls=":")
    ax.text(1.5, 1.0, "H2: SFT family", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.text(4.7, 1.0, "H2-KTO", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(pos, [MODEL_LABELS[m] for m in models])
    ax.set_xlim(-0.6, 5.4)
    ax.set_ylabel(r"$\Delta_M$ (nats)")
    ax.legend(frameon=False, loc="lower left")
    style(ax)
    save(fig, "fig_rq2_delta_boxplots", texts)


def fig_rq2_pairs(texts: dict) -> None:
    """Forest plot of the six H2 pairs and H2-KTO, mean [95% bootstrap CI], shaded +-0.05 nat precision band."""
    pw = pd.read_csv(S08 / "h2_pairwise.csv")
    kto = pd.read_csv(S08 / "h2kto_tests.csv").query("quantity == 'D'")
    rows = pd.concat([pw, kto], ignore_index=True)
    labels = [f"{MODEL_LABELS[a]} − {MODEL_LABELS[b]}" for a, b in zip(rows["m1"], rows["m2"])]
    y = np.array([6.6, 5.6, 4.6, 3.6, 2.6, 1.6, 0.0])
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    ax.axvspan(-PRECISION, PRECISION, color="#eeeeee", lw=0, label="±0.05 nats (numerical tolerance)")
    ax.axvline(0, color=MUTED, lw=1)
    ax.errorbar(rows["mean"], y, xerr=[rows["mean"] - rows["ci_boot_lo"], rows["ci_boot_hi"] - rows["mean"]], fmt="o", ms=5,
                color=BLUE, ecolor=INK, elinewidth=1.2, capsize=3, markeredgecolor="white")
    for yi, m, hi in zip(y, rows["mean"], rows["ci_boot_hi"]):
        ax.annotate(f"{m:+.2f}", (hi + 0.015, yi), va="center", fontsize=8, color=INK)
    ax.axhline(0.8, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(y, labels)
    ax.set_xlim(rows["ci_boot_lo"].min() - 0.05, rows["ci_boot_hi"].max() + 0.12)
    ax.set_xlabel(r"$\bar{D}_{M_1,M_2}$ (nats), 95% bootstrap CI")
    ax.legend(frameon=False, loc="lower right")
    style(ax, grid_axis="x")
    save(fig, "fig_rq2_pairwise_differences", texts)


def fig_rq2_components(texts: dict) -> None:
    """Mean delta_pos / delta_neg of M1 vs M2 (aligned vs aligned, all prefixes), 95% CIs, D above each pair."""
    c = pd.read_csv(S08 / "components_aligned.csv").query("category == 'ALL'").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    pos, w = np.array([0, 1, 2, 3.4]), 0.36
    for off, col, color, lab in [(-w / 2, "delta_pos", BLUE, r"$\delta^{+}_{M_1} - \delta^{+}_{M_2}$ (neutral pool)"),
                                 (w / 2, "delta_neg", ORANGE, r"$\delta^{-}_{M_1} - \delta^{-}_{M_2}$ (derogatory pool)")]:
        mean = c[f"mean_{col}"]
        ax.bar(pos + off, mean, width=w, color=color, edgecolor="white", lw=2, label=lab)
        ax.errorbar(pos + off, mean, yerr=[mean - c[f"{col}_ci_lo"], c[f"{col}_ci_hi"] - mean], fmt="none", ecolor=INK,
                    elinewidth=1, capsize=3)
    top = max(c["delta_pos_ci_hi"].max(), c["delta_neg_ci_hi"].max())
    for x, d in zip(pos, c["mean_D"]):
        ax.annotate(rf"$\bar{{D}}_{{M_1,M_2}} = {d:+.2f}$", (x, top + 0.04), ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(2.7, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(pos, [f"{MODEL_LABELS[a]}\nvs {MODEL_LABELS[b]}" for a, b in zip(c["m1"], c["m2"])])
    ax.set_ylabel(r"mean change in pool log-prob, $M_1 - M_2$ (nats)")
    ax.set_ylim(min(c["delta_pos_ci_lo"].min(), c["delta_neg_ci_lo"].min(), 0) - 0.1, top + 0.18)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)   # above the axes, clear of the bars
    style(ax)
    save(fig, "fig_rq2_components_aligned", texts)


def fig_rq3_shift(cfg: dict, texts: dict) -> None:
    """Shift_M(c) [95% bootstrap CI] per category: SFT family in blue (one marker per model), Direct KTO in orange."""
    sh = pd.read_csv(S09 / "h3_shift_by_category.csv")
    cats, lab = ordered_categories(cfg), display_labels(cfg)
    markers = {"sft": "o", "sft_dpo": "s", "sft_ppo": "^", "sft_kto": "D", "kto": "v"}
    offs = dict(zip(ALIGNED, [0.3, 0.15, 0.0, -0.15, -0.36]))
    fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
    ax.axvline(0, color=MUTED, lw=1)
    for m in ALIGNED:
        s = sh[sh["model"] == m].set_index("category").loc[cats]
        y = np.arange(len(cats))[::-1] + offs[m]
        color = ORANGE if m == "kto" else BLUE
        ax.errorbar(s["mean"], y, xerr=[s["mean"] - s["ci_boot_lo"], s["ci_boot_hi"] - s["mean"]], fmt=markers[m], ms=5.5,
                    color=color, ecolor=color, elinewidth=1.1, capsize=0, markeredgecolor="white", label=MODEL_LABELS[m])
    for i in range(1, len(cats)):
        ax.axhline(i - 0.5, color="#eeeeee", lw=0.8)
    ax.set_yticks(np.arange(len(cats))[::-1], [lab[c] for c in cats])
    ax.set_xlabel(r"$\mathrm{Shift}_M(c)$ (nats), 95% bootstrap CI")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, handletextpad=0.3, columnspacing=1.2)
    style(ax, grid_axis="x")
    save(fig, "fig_rq3_shift_by_category", texts)


def _category_frame(dm: pd.DataFrame) -> pd.DataFrame:
    """Aligned rows of the delta file with prefix toxicity attached."""
    tox = pd.read_parquet(ROOT / "results/toxicity.parquet")[["prefix_id", "toxicity"]]
    return dm.merge(tox, on="prefix_id", how="left", validate="many_to_one")


def fig_rq3_distribution(df: pd.DataFrame, cfg: dict, texts: dict) -> None:
    """Violins of D_M(c|S) per category, one panel per model; diamond = category mean; x ticks = abbreviated display labels."""
    cats, lab = ordered_categories(cfg), display_labels(cfg)
    fig, axes = plt.subplots(1, 5, figsize=(6.3, 3.2), sharey=True)   # drawn at final text width so the tick labels fit
    for ax, m in zip(axes, ALIGNED):
        g = df[df["model"] == m]
        data = [g.loc[g["category"] == c, "D"].to_numpy() for c in cats]
        color = ORANGE if m == "kto" else BLUE
        vp = ax.violinplot(data, positions=range(len(cats)), widths=0.8, showextrema=False)
        for b in vp["bodies"]:
            b.set(facecolor=color, alpha=0.3, edgecolor=color)
        ax.plot(range(len(cats)), [d.mean() for d in data], "D", ms=4, color=INK, markeredgecolor="white")
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.set_xticks(range(len(cats)), [ABBR[lab[c]] for c in cats], fontsize=7, rotation=90)
        ax.set_title(MODEL_LABELS[m], fontsize=8.5, color=INK)
        style(ax)
    axes[0].set_ylabel(r"$D_M(c \mid S)$ (nats)")
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_rq3_shift_distribution", texts)


def fig_rq3_toxicity(df: pd.DataFrame, cfg: dict, texts: dict) -> None:
    """Prefix toxicity vs D_M(c|S): grey points, one least-squares line per category (descriptive), one panel per model."""
    cats, lab = ordered_categories(cfg), display_labels(cfg)
    fig, axes = plt.subplots(1, 5, figsize=(WIDTH, 2.9), sharey=True)
    xs = np.array([0.0, 1.0])
    for ax, m in zip(axes, ALIGNED):
        g = df[df["model"] == m]
        ax.scatter(g["toxicity"], g["D"], s=4, color=MUTED, alpha=0.35, lw=0)
        for c, color in zip(cats, CAT_COLORS):
            gc = g[g["category"] == c]
            b, a = np.polyfit(gc["toxicity"], gc["D"], 1)
            ax.plot(xs, a + b * xs, color=color, lw=2, label=lab[c])
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.set_title(MODEL_LABELS[m], fontsize=8.5, color=INK)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        style(ax)
    fig.supxlabel("prefix toxicity (Detoxify)", fontsize=9, y=-0.04)
    axes[0].set_ylabel(r"$D_M(c \mid S)$ (nats)")
    axes[0].set_ylim(-2.5, 3.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.06))
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_rq3_toxicity_scatter", texts)


def fig_rq3_frequency(cfg: dict, texts: dict) -> None:
    """Category-frequency check: (A) coverage vs SFT-family Shift, (B) derogatory ratio vs Shift; main filled, all-terms variant hollow."""
    vs = pd.read_csv(S09 / "s9b2_vs_shift.csv")
    rho = pd.read_csv(S09 / "s9b2_rho.csv")
    pref = pd.read_csv(S09 / "s9b2_preference.csv")
    lab = display_labels(cfg)
    cnt = pref[(pref["variant"] == "main") & (pref["dataset"] == "pooled") & (pref["polarity"] == "NOT SAFE")].set_index("category")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.8), sharey=True)
    for ax, col, meas, xlab in [(axes[0], "coverage_per_10k", "A coverage", "(A) conversations mentioning the category\nper 10,000 (log scale)"),
                                (axes[1], "derog_ratio", "B derogatory ratio", "(B) derogatory terms: rate in rejected /\nrate in chosen responses")]:
        for v, kw in [("main", dict(color=BLUE, s=36)), ("all_terms", dict(facecolor="none", edgecolor=ORANGE, s=46, lw=1.3))]:
            t = vs[vs["variant"] == v]
            r = rho[(rho["variant"] == v) & (rho["measure"] == meas) & (rho["dataset"] == "pooled")]["rho"].iloc[0]
            name = "main detector" if v == "main" else "all terms"
            ax.scatter(t[col], t["shift"], zorder=3, label=f"{name}, ρ = {r:.1f}", **kw)
        main = vs[vs["variant"] == "main"]
        for _, row in main.iterrows():
            text = lab[row["category"]]
            right_edge = row[col] == main[col].max()   # label of the rightmost point goes to its left, inside the axes
            if col == "derog_ratio":
                k = cnt.loc[row["category"]]
                text += f" ({int(k['chosen_count'])}/{int(k['rejected_count'])})"
            ax.annotate(text, (row[col], row["shift"]), xytext=(-4 if right_edge else 4, 3), textcoords="offset points",
                        ha="right" if right_edge else "left", fontsize=7, color=INK)
        if col == "coverage_per_10k":
            ax.set_xscale("log")
            ax.set_xlim(8, 300)
            ax.set_xticks([10, 20, 50, 100, 200])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.axvline(1, color=MUTED, lw=1)
        ax.set_xlabel(xlab, fontsize=8)
        ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=1)
        ax.grid(True, color=GRID, lw=0.8)
        style(ax)
    axes[0].set_ylabel(r"$\overline{\mathrm{Shift}}(c)$ (nats)")
    save(fig, "fig_rq3_category_frequency", texts)


def label_checks(texts: dict[str, list[str]], cfg: dict) -> pd.DataFrame:
    """Per figure: no internal category name or old model label among the drawn texts; every category label present where expected."""
    internal = [c for c, shown in display_labels(cfg).items() if c != shown] + ["KTO (direct)", "African", "Female Homosexual"]
    shown = set(display_labels(cfg).values())
    with_categories = {"fig_rq3_shift_by_category", "fig_rq3_toxicity_scatter", "fig_rq3_category_frequency"}   # distribution: BP/HM/...
    rows = []
    for name, tx in texts.items():
        joined = "\n".join(tx)
        bad = [w for w in internal if w in joined]
        missing = [s for s in shown if s not in joined] if name in with_categories else []
        same = filecmp.cmp(FIGS / f"{name}.pdf", THESIS_FIGS / f"{name}.pdf", shallow=False)
        rows.append(dict(check=f"{name}: labels, thesis copy", internal_found=";".join(bad) or "none",
                         display_missing=";".join(missing) or "none", thesis_copy_identical=same, passed=not bad and not missing and same))
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    FIGS.mkdir(parents=True, exist_ok=True)
    THESIS_FIGS.mkdir(parents=True, exist_ok=True)
    dm = pd.read_parquet(ROOT / "results/delta_1.4b.parquet")
    dm = dm[dm["model"].isin(ALIGNED)]
    df = _category_frame(dm)
    texts: dict[str, list[str]] = {}
    fig_rq1_d(dm, texts)
    fig_rq1_components(texts)
    fig_rq2_delta(dm, texts)
    fig_rq2_pairs(texts)
    fig_rq2_components(texts)
    fig_rq3_shift(cfg, texts)
    fig_rq3_distribution(df, cfg, texts)
    fig_rq3_toxicity(df, cfg, texts)
    fig_rq3_frequency(cfg, texts)
    checks = label_checks(texts, cfg)
    checks.to_csv(OUT / "gate_checks_figures.csv", index=False)
    print(f"figures: {len(texts)} written to {FIGS.relative_to(ROOT)}; label checks {int(checks['passed'].sum())}/{len(checks)} pass")
    if not checks["passed"].all():
        raise SystemExit(checks[~checks["passed"]].to_string(index=False))


if __name__ == "__main__":
    main()
