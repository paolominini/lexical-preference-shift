"""Thesis tables and figures for the replication across sizes (Section 5.5, descriptive), from the outputs of 12_scale.py only.

No new statistical test. Every number comes from the tables of 12_scale.py (reports/step_12_tables/) or, for the prefix-level components, from the
D tables (results/delta_<size>.parquet) through `components_table` of 07_rq1.py (mean and bootstrap CI, same seed as 07).
The 1.4B column is always the main run (Mac). H1 verdicts per size use the thesis rule (as in `tab_h1`): mean D > 0, bootstrap
95% CI excludes 0, two-sided Wilcoxon p < 0.05 unadjusted (12_scale.py uses Bonferroni x5; the cells that differ are listed in gate_checks_scale.csv).
Reads: reports/step_12_tables/{h1_by_size,h2_order_by_size,h2kto_by_size,h3_family_by_size,h3_order_rho}.csv,
       results/delta_<size>.parquet, reports/step_07_tables/{h1_tests,components_vs_base}.csv,
       reports/step_09_tables/h3_family_summary.csv, thesis/tables/tab_h1.tex (1.4B cross-check), config.yaml
Writes: reports/step_11_thesis/tables/{tab_scale,tab_scale_categories}.tex, reports/step_11_thesis/figures/{fig_scale_D_by_size,
        fig_scale_components}.pdf, identical copies in thesis/tables/ and thesis/figures/,
        reports/step_11_thesis/{scale_h1_verdicts,scale_components_by_size,scale_text_numbers,gate_checks_scale}.csv
Usage: .venv/bin/python scripts/11d_scale_assets.py
"""
from __future__ import annotations

import filecmp
import hashlib
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probing.config import load_config  # noqa: E402
from probing.latex import MODEL_LABELS, display_labels, fmt_ci, fmt_num, ordered_categories, table_env, tabular  # noqa: E402
from probing.latex import write_fragment  # noqa: E402
from probing.scale import ALPHA, load_script  # noqa: E402

t11 = load_script(ROOT / "scripts/11_thesis_tables.py")
f11 = load_script(ROOT / "scripts/11_thesis_figures.py")   # sets the thesis rcParams; reuses style(), save(), colors
d07 = load_script(ROOT / "scripts/07_rq1.py")
plt = f11.plt

OUT, TABLES, THESIS_TABLES = t11.OUT, t11.TABLES, t11.THESIS_TABLES
S07, S09, S12 = (ROOT / f"reports/step_{n}_tables" for n in ["07", "09", "12"])
SIZES = ["1.4b", "2.8b", "6.9b", "12b"]
SIZE_LABELS = {"1.4b": "1.4B", "2.8b": "2.8B", "6.9b": "6.9B", "12b": "12B"}
FAMILY, ALIGNED = t11.FAMILY, t11.ALIGNED
GENERATOR = "scripts/11d_scale_assets.py"
SUP_A, SUP_B = "\\textsuperscript{a}", "\\textsuperscript{b}"


# ---------------------------------------------------------------- data

def h1_verdicts() -> pd.DataFrame:
    """Per size x post-trained checkpoint (reference: the base model with BOS): thesis-rule verdict, significantly negative flag,
    and the verdict of 12_scale.py (Bonferroni x5) for comparison."""
    h = pd.read_csv(S12 / "h1_by_size.csv").query("reference == 'base_bos'").copy()
    sig = h["wilcoxon_p"] < ALPHA
    h["verdict_thesis"] = np.where((h["mean"] > 0) & (h["ci_boot_lo"] > 0) & sig, "supported", "not supported")
    h["significantly_negative"] = (h["mean"] < 0) & (h["ci_boot_hi"] < 0) & sig
    h["verdict_differs_from_s12"] = h["verdict_thesis"] != h["verdict"]
    cols = ["size", "model", "mean", "ci_boot_lo", "ci_boot_hi", "wilcoxon_p", "wilcoxon_p_bonf5", "verdict", "verdict_thesis",
            "significantly_negative", "verdict_differs_from_s12"]
    return h[cols].rename(columns={"verdict": "verdict_s12_bonf5"}).reset_index(drop=True)


def components_by_size(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prefix-level mean delta_pos, delta_neg, D vs the base model [bootstrap CI] per size x checkpoint (components_table of 07, all
    prefixes), and the base model's mean Delta per size."""
    comp, base = [], []
    for size in SIZES:
        dm = pd.read_parquet(ROOT / f"results/delta_{size}.parquet")
        c = d07.components_table(dm, ALIGNED, seed)
        comp.append(c[c["category"] == "ALL"].assign(size=size))
        base.append(dict(size=size, base_mean_delta=float(dm.loc[dm["model"] == "base", "delta"].mean())))
    return pd.concat(comp, ignore_index=True), pd.DataFrame(base)


# ---------------------------------------------------------------- tables

def _h1_cell(r: pd.Series) -> str:
    """Mean D with the marks of tab_scale: a = H1 not supported (thesis rule), b = significantly negative."""
    marks = (["a"] if r["verdict_thesis"] != "supported" else []) + (["b"] if r["significantly_negative"] else [])
    return fmt_num(r["mean"], signed=True) + (f"\\textsuperscript{{{','.join(marks)}}}" if marks else "")


def table_scale(v: pd.DataFrame) -> str:
    """Ch 5: mean D_M [95% bootstrap CI] per checkpoint x size; two lines per checkpoint (as tab_h3_per_model)."""
    v = v.set_index(["model", "size"])
    rows = []
    for m in ALIGNED:
        rows.append([MODEL_LABELS[m]] + [_h1_cell(v.loc[(m, s)]) for s in SIZES])
        rows.append([""] + ["{\\scriptsize " + fmt_ci(v.loc[(m, s), "ci_boot_lo"], v.loc[(m, s), "ci_boot_hi"]) + "}" for s in SIZES])
    body = tabular("lcccc", ["Checkpoint"] + [SIZE_LABELS[s] for s in SIZES], rows, midrules=tuple(range(2, len(rows), 2)))
    note = ("$N = 500$ prefixes; $\\bar{D}_M$ in nats, 95\\% bootstrap confidence intervals; H1 verdicts with the rule of "
            f"Section~\\ref{{sec:h1}}. {SUP_A}\\,H1 not supported. {SUP_B}\\,Significantly negative.")
    return table_env(body, "Mean preference shift relative to the base model across model sizes", "tab:scale", note)


def _span(lo: float, hi: float) -> str:
    """Range 'lo to hi': with negative values an en dash would sit next to a minus sign ('-0.13--0.09')."""
    return f"{fmt_num(lo, signed=True)} to {fmt_num(hi, signed=True)}"


def table_scale_categories(cfg: dict) -> str:
    """Appendix: SFT-family mean Shift per category (second line: min-max over the four checkpoints) x size; last row Spearman rho
    of the SFT-family category order vs 1.4B (descriptive, n = 5, no p)."""
    fam = pd.read_csv(S12 / "h3_family_by_size.csv").set_index(["category", "size"])
    rho = pd.read_csv(S12 / "h3_order_rho.csv").query("series == 'family_mean'").set_index("size")["rho_vs_1_4b"]
    lab = display_labels(cfg)
    rows = []
    for c in ordered_categories(cfg):
        rows.append([lab[c]] + [fmt_num(fam.loc[(c, s), "family_mean"], signed=True) for s in SIZES])
        rows.append([""] + ["{\\scriptsize " + _span(fam.loc[(c, s), "family_min"], fam.loc[(c, s), "family_max"]) + "}"
                            for s in SIZES])
    rows.append(["Spearman $\\rho$ with 1.4B", "--"] + [fmt_num(rho[s], nd=1, signed=True) for s in SIZES[1:]])
    body = tabular("lcccc", ["Category"] + [SIZE_LABELS[s] for s in SIZES], rows, midrules=tuple(range(2, len(rows), 2)))
    note = ("$\\overline{\\mathrm{Shift}}(c)$ in nats over the 100 prefixes of each category: mean over the four SFT-family "
            "checkpoints; second line: range of $\\mathrm{Shift}_M(c)$ over the four (min to max). Last row: Spearman $\\rho$ between the category order and the order at 1.4B "
            "(descriptive, five categories).")
    return table_env(body, "Shift by category across model sizes, SFT family", "tab:scale-categories", note)


# ---------------------------------------------------------------- figures

MARKERS = {"sft": "o", "sft_dpo": "s", "sft_ppo": "^", "sft_kto": "D", "kto": "v"}


def fig_scale_d(v: pd.DataFrame, texts: dict) -> None:
    """Mean D_M [95% bootstrap CI] per checkpoint across sizes (categorical x): SFT family in blue (one marker per checkpoint),
    Direct KTO in orange (as fig_rq3_shift_by_category); dashed zero line."""
    fig, ax = plt.subplots(figsize=(f11.WIDTH, 3.4))
    x = np.arange(len(SIZES))
    offs = dict(zip(ALIGNED, np.linspace(-0.14, 0.14, len(ALIGNED))))
    ax.axhline(0, color=f11.MUTED, lw=1, ls="--")
    for m in ALIGNED:
        g = v[v["model"] == m].set_index("size").loc[SIZES]
        color = f11.ORANGE if m == "kto" else f11.BLUE
        ax.errorbar(x + offs[m], g["mean"], yerr=[g["mean"] - g["ci_boot_lo"], g["ci_boot_hi"] - g["mean"]], fmt=MARKERS[m] + "-",
                    ms=5.5, lw=1.3, color=color, ecolor=color, elinewidth=1.1, capsize=0, markeredgecolor="white",
                    label=MODEL_LABELS[m])
    ax.set_xticks(x, [SIZE_LABELS[s] for s in SIZES])
    ax.set_xlim(-0.4, len(SIZES) - 0.6)
    ax.set_xlabel("model size (parameters)")
    ax.set_ylabel(r"$\bar{D}_M$ (nats)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, handletextpad=0.3, columnspacing=1.2)
    f11.style(ax)
    f11.save(fig, "fig_scale_D_by_size", texts)


def fig_scale_components(comp: pd.DataFrame, texts: dict) -> None:
    """One panel per size, laid out as fig_rq1_components_vs_base: mean delta_pos / delta_neg vs the base model per checkpoint
    (prefix level, all prefixes), 95% bootstrap CIs, mean D_M above each pair (value only, no "D =": the panels are narrow)."""
    fig, axes = plt.subplots(2, 2, figsize=(f11.WIDTH, 5.4), sharey=True)
    pos, w = np.arange(len(ALIGNED)), 0.36
    lo = min(comp["delta_neg_ci_lo"].min(), comp["delta_pos_ci_lo"].min()) * 1.12
    hi = max(0.2, comp["delta_pos_ci_hi"].max() + 0.15, comp["delta_neg_ci_hi"].max() + 0.15)
    for ax, size in zip(axes.flat, SIZES):
        c = comp[comp["size"] == size].set_index("model").loc[ALIGNED]
        for off, col, color, lab in [(-w / 2, "delta_pos", f11.BLUE, r"$\delta^{+}_M$ (neutral pool) vs Base"),
                                     (w / 2, "delta_neg", f11.ORANGE, r"$\delta^{-}_M$ (derogatory pool) vs Base")]:
            mean = c[f"mean_{col}"]
            ax.bar(pos + off, mean, width=w, color=color, edgecolor="white", lw=1.5, label=lab)
            ax.errorbar(pos + off, mean, yerr=[mean - c[f"{col}_ci_lo"], c[f"{col}_ci_hi"] - mean], fmt="none", ecolor=f11.INK,
                        elinewidth=0.9, capsize=2)
        for xi, d in zip(pos, c["mean_D"]):
            ax.annotate(f"{d:+.2f}", (xi, 0.04), ha="center", va="bottom", fontsize=7, color=f11.INK)   # bar D_M (caption)
        ax.axhline(0, color=f11.INK, lw=0.8)
        ax.set_xticks(pos, [MODEL_LABELS[m].replace("Direct ", "Direct\n").replace("SFT-", "SFT-\n") for m in ALIGNED], fontsize=7)
        ax.set_title(SIZE_LABELS[size], fontsize=8.5, color=f11.INK)
        ax.set_ylim(lo, hi)
        f11.style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean change in log-prob\nvs Base (nats)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(wspace=0.08, hspace=0.45)
    f11.save(fig, "fig_scale_components", texts)


# ---------------------------------------------------------------- numbers for the text

def _rng(s: pd.Series, signed: bool = True) -> str:
    f = "{:+.2f}" if signed else "{:.2f}"
    return f"{f.format(s.min())} to {f.format(s.max())}"


def text_numbers(v: pd.DataFrame, comp: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Per size: the numbers Section 5.5 cites (all read from the tables above or the CSVs of 12_scale.py)."""
    order = pd.read_csv(S12 / "h2_order_by_size.csv").set_index("size")
    kto = pd.read_csv(S12 / "h2kto_by_size.csv").query("quantity == 'D'").set_index("size")
    rho = pd.read_csv(S12 / "h3_order_rho.csv").query("series == 'family_mean'").set_index("size")["rho_vs_1_4b"]
    base = base.set_index("size")
    rows = []
    for s in SIZES:
        vs = v[v["size"] == s].set_index("model")
        cs = comp[comp["size"] == s].set_index("model")
        sup = [MODEL_LABELS[m] for m in ALIGNED if vs.loc[m, "verdict_thesis"] == "supported"]
        neg = [MODEL_LABELS[m] for m in ALIGNED if vs.loc[m, "significantly_negative"]]
        k = vs.loc["kto"]
        items = {
            "H1 verdicts (thesis rule)": f"{len(sup)}/5 supported" + (f" ({', '.join(sup)})" if sup else "")
                                         + (f"; significantly negative: {', '.join(neg)}" if neg else ""),
            "SFT-family mean D range": _rng(vs.loc[FAMILY, "mean"]),
            "Direct KTO mean D [CI]": f"{k['mean']:+.2f} [{k['ci_boot_lo']:.2f}, {k['ci_boot_hi']:.2f}], Wilcoxon p = {k['wilcoxon_p']:.2g}",
            "Base mean Delta": f"{base.loc[s, 'base_mean_delta']:.2f}",
            "SFT-family delta_pos range": _rng(cs.loc[FAMILY, "mean_delta_pos"]),
            "SFT-family delta_neg range": _rng(cs.loc[FAMILY, "mean_delta_neg"]),
            "Direct KTO delta_pos / delta_neg": f"{cs.loc['kto', 'mean_delta_pos']:+.2f} / {cs.loc['kto', 'mean_delta_neg']:+.2f}",
            "H2 gaps above the tolerance (of 6)": str(int(order.loc[s, "n_gaps_above_band"])),
            "H2-KTO mean D [CI]": f"{kto.loc[s, 'mean']:+.2f} [{kto.loc[s, 'ci_boot_lo']:.2f}, {kto.loc[s, 'ci_boot_hi']:.2f}]",
            "rho of the SFT-family category order vs 1.4B": "--" if s == "1.4b" else f"{rho[s]:+.1f}",
        }
        rows += [dict(size=SIZE_LABELS[s], quantity=q, value=val) for q, val in items.items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- checks

def tab_h1_cells() -> dict[str, tuple[str, str]]:
    """Mean and CI cells of each row of the thesis table tab_h1 (thesis/tables/tab_h1.tex)."""
    out = {}
    for line in (ROOT / "thesis/tables/tab_h1.tex").read_text().splitlines():
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if cells[0] in MODEL_LABELS.values() and len(cells) == 7:
            out[cells[0]] = (cells[1], cells[2])
    return out


def max_abs_diff(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], cols: list[str]) -> float:
    m = a[keys + cols].merge(b[keys + cols], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    assert len(m) == len(a) == len(b), (len(m), len(a), len(b))
    return float(max((m[f"{c}_a"] - m[f"{c}_b"]).abs().max() for c in cols))


def md5s(paths: list[Path]) -> dict[str, str]:
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in paths}


def checks_numbers(v: pd.DataFrame, comp: pd.DataFrame) -> list[dict]:
    """1.4B values vs the main-run tables (tab_h1, step 07 and step 09 CSVs) and the verdict comparison with 12_scale.py."""
    out = []
    h1 = tab_h1_cells()
    v14 = v[v["size"] == "1.4b"].set_index("model")
    for m in ALIGNED:
        got = (fmt_num(v14.loc[m, "mean"], signed=True), fmt_ci(v14.loc[m, "ci_boot_lo"], v14.loc[m, "ci_boot_hi"]))
        out.append(dict(check=f"tab_scale 1.4B {MODEL_LABELS[m]} == tab_h1 cells", value=" ".join(got), expected=" ".join(h1[MODEL_LABELS[m]])))
    t7 = pd.read_csv(S07 / "h1_tests.csv").query("reference == 'base_bos'")
    out.append(dict(check="1.4B mean D and CI == step_07 h1_tests.csv (max abs diff < 1e-12)",
                    value=str(max_abs_diff(v[v["size"] == "1.4b"], t7, ["model"], ["mean", "ci_boot_lo", "ci_boot_hi", "wilcoxon_p"]) < 1e-12),
                    expected="True"))
    c7 = pd.read_csv(S07 / "components_vs_base.csv").query("category == 'ALL'")
    cols = [f"{p}{q}" for q in ["delta_pos", "delta_neg", "D"] for p in ["mean_"]] + \
           [f"{q}_ci_{e}" for q in ["delta_pos", "delta_neg", "D"] for e in ["lo", "hi"]]
    out.append(dict(check="1.4B components == step_07 components_vs_base.csv ALL (max abs diff < 1e-12)",
                    value=str(max_abs_diff(comp[comp["size"] == "1.4b"], c7, ["model"], cols) < 1e-12), expected="True"))
    f9 = pd.read_csv(S09 / "h3_family_summary.csv")
    f12 = pd.read_csv(S12 / "h3_family_by_size.csv").query("size == '1.4b'")
    out.append(dict(check="1.4B SFT-family Shift == step_09 h3_family_summary.csv (max abs diff < 1e-12)",
                    value=str(max_abs_diff(f12, f9, ["category"], ["family_mean", "family_min", "family_max"]) < 1e-12), expected="True"))
    d12 = pd.read_csv(S12 / "h1_by_size.csv").query("reference == 'base_bos'")
    comp_d = comp[["size", "model", "mean_D"]].merge(d12[["size", "model", "mean"]], on=["size", "model"], validate="one_to_one")
    out.append(dict(check="components mean D == h1_by_size mean, all sizes (max abs diff < 1e-12)",
                    value=str(float((comp_d["mean_D"] - comp_d["mean"]).abs().max()) < 1e-12), expected="True"))
    diff = v[v["verdict_differs_from_s12"]]
    out.append(dict(check="H1 verdicts: thesis rule (unadjusted) vs h1_by_size.csv (Bonferroni x5), cells that differ",
                    value=";".join(f"{r['size']} {r['model']}" for _, r in diff.iterrows()) or "none", expected="none", blocking=False))
    return out


def checks_assets(frags: dict[str, str], texts: dict[str, list[str]], cfg: dict) -> list[dict]:
    """Labels, words, structure and thesis copies of the new tables and figures."""
    out = []
    for name, text in frags.items():
        same = filecmp.cmp(TABLES / f"{name}.tex", THESIS_TABLES / f"{name}.tex", shallow=False)
        out += [dict(check=f"{name}: internal labels", value=";".join(t11.internal_names_found(text, cfg)) or "none", expected="none"),
                dict(check=f"{name}: derogatory words", value=str(len(t11.derogatory_words_found(text))), expected="0"),
                dict(check=f"{name}: dropped terms or 'alignment'",
                     value=";".join(t11.dropped_terms_found(text) + (["alignment"] if re.search("align", text, re.I) else [])) or "none",
                     expected="none"),
                dict(check=f"{name}: structure (braces, & per row)", value=str(t11.structure_ok(text)), expected="True"),
                dict(check=f"{name}: copy in thesis/tables identical", value=str(same), expected="True")]
    lab = f11.label_checks(texts, cfg)
    for _, r in lab.iterrows():
        out.append(dict(check=r["check"], value=str(bool(r["passed"])), expected="True"))
    for name, tx in texts.items():
        joined = "\n".join(tx)
        bad = [w for w in ["align", "Alignment", "KTO (direct)"] if w in joined] + [str(len(t11.derogatory_words_found(joined)))]
        out.append(dict(check=f"{name}: no 'align', no 'KTO (direct)', 0 derogatory words", value=";".join(bad), expected="0"))
    return out


def main() -> None:
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    inputs = [ROOT / f"results/delta_{s}.parquet" for s in SIZES] + sorted(S12.glob("*.csv"))
    before = md5s(inputs)
    v = h1_verdicts()
    comp, base = components_by_size(seed)
    frags = {"tab_scale": (table_scale(v), ["reports/step_12_tables/h1_by_size.csv (H1 rule of Section 3.3, unadjusted)"]),
             "tab_scale_categories": (table_scale_categories(cfg), ["reports/step_12_tables/h3_family_by_size.csv", "h3_order_rho.csv"])}
    for name, (text, sources) in frags.items():
        write_fragment(TABLES / f"{name}.tex", text, sources, generator=GENERATOR)
        shutil.copyfile(TABLES / f"{name}.tex", THESIS_TABLES / f"{name}.tex")
    texts: dict[str, list[str]] = {}
    fig_scale_d(v, texts)
    fig_scale_components(comp, texts)

    v.to_csv(OUT / "scale_h1_verdicts.csv", index=False)
    comp.merge(base, on="size").to_csv(OUT / "scale_components_by_size.csv", index=False)
    nums = text_numbers(v, comp, base)
    nums.to_csv(OUT / "scale_text_numbers.csv", index=False)

    checks = [dict(blocking=True, **c) if "blocking" not in c else c for c in checks_numbers(v, comp)]
    checks += [dict(blocking=True, **c) for c in checks_assets({k: t for k, (t, _) in frags.items()}, texts, cfg)]
    checks.append(dict(check="inputs unchanged (md5 of delta_<size>.parquet and step_12 CSVs before = after)",
                       value=str(before == md5s(inputs)), expected="True", blocking=True))
    out = pd.DataFrame(checks)
    out["passed"] = out["value"] == out["expected"]
    out = out[["check", "value", "expected", "passed", "blocking"]]
    out.to_csv(OUT / "gate_checks_scale.csv", index=False)
    print(f"tables: {len(frags)}, figures: {len(texts)}; checks {int(out['passed'].sum())}/{len(out)} pass")
    if not out["passed"].all():
        print(out[~out["passed"]].to_string(index=False))
    if not out.loc[out["blocking"], "passed"].all():
        raise SystemExit("blocking check failed")


if __name__ == "__main__":
    main()
