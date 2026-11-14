"""LaTeX table fragments for the thesis, built only from existing tables (nothing is recomputed).

Reads: reports/step_07_tables/h1_tests.csv, reports/step_08_tables/{h2_friedman,h2_pairwise,h2kto_tests}.csv,
       reports/step_09_tables/{h3_family_summary,h3_shift_by_category,h3_tests}.csv,
       reports/step_10_tables/{gate_table,h1_robust,h3_shift_robust}.csv, reports/step_06_tables/distribution_by_category_polarity.csv,
       data/final_prefix_set.csv and data/Vocabulary_Final.xlsx (counts only), config.yaml (checkpoints, display labels).
Writes: reports/step_11_thesis/tables/*.tex (booktabs; needs \\usepackage{booktabs} and amsmath), an identical copy in
        thesis/tables/ (so thesis/ compiles on its own), reports/step_11_thesis/gate_checks_tables.csv
Category names follow `config.yaml` `display_labels`; internal names never reach a .tex file (checked).
Usage: .venv/bin/python scripts/11_thesis_tables.py
"""
from __future__ import annotations

import filecmp
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probing.config import load_config  # noqa: E402
from probing.latex import (MODEL_LABELS, display_labels, fmt_ci, fmt_num, fmt_p, fmt_range,  # noqa: E402
                           ordered_categories, table_env, tabular, write_fragment)
from probing.pools import load_prefixes, load_vocabulary  # noqa: E402

OUT = ROOT / "reports/step_11_thesis"
TABLES = OUT / "tables"
THESIS_TABLES = ROOT / "thesis/tables"
SUBSET_NAMES = {"added": "added corpora", "unedited": "unedited prefixes", "A": "half A"}
S06, S07, S08, S09, S10 = (ROOT / f"reports/step_{n}_tables" for n in ["06", "07", "08", "09", "10"])
FAMILY = ["sft", "sft_dpo", "sft_ppo", "sft_kto"]
ALIGNED = FAMILY + ["kto"]
PRECISION_BAND = 0.06   # nat; upper edge of the 0.04-0.06 band: gaps below it are within numerical tolerance
POST_TRAINING = {"base": "none (pre-trained only)", "sft": "SFT", "sft_dpo": "SFT, then DPO", "sft_ppo": "SFT, then offline PPO",
                 "sft_kto": "SFT, then KTO", "kto": "KTO without SFT"}
DAG = "\\textsuperscript{\\dag}"
INPUT_NAMES = {"raw": "plain", "chat": "chat"}   # thesis names of the two input formats


def tex_minus(text: str) -> str:
    """Pair labels in the CSVs use a Unicode minus; in LaTeX text it becomes a math minus."""
    return text.replace(" − ", " $-$ ")


# ---------------------------------------------------------------- Chapter 4

def table_design(cfg: dict) -> str:
    """Ch 4: per category, prefixes by source polarity, pool sizes, prefix toxicity (mean, median)."""
    pre, voc = load_prefixes(), load_vocabulary()
    tox = pd.read_csv(S06 / "distribution_by_category_polarity.csv")
    tox = tox[(tox["score"] == "toxicity") & (tox["polarity"] == "ALL")].set_index("category")
    lab = display_labels(cfg)
    rows = []
    for c in ordered_categories(cfg) + ["ALL"]:
        p = pre if c == "ALL" else pre[pre["category"] == c]
        v = voc if c == "ALL" else voc[voc["category"] == c]
        rows.append(["All" if c == "ALL" else lab[c], str(int((p["polarity"] == "SAFE").sum())),
                     str(int((p["polarity"] == "NOT SAFE").sum())), str(len(p)),
                     str(int((v["polarity"] == "SAFE").sum())), str(int((v["polarity"] == "NOT SAFE").sum())),
                     fmt_num(tox.loc[c, "mean"]), fmt_num(tox.loc[c, "q50"])])
    head = ("& \\multicolumn{3}{c}{Prefixes, by source word} & \\multicolumn{2}{c}{Candidate pool} & \\multicolumn{2}{c}{Prefix toxicity} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}")
    body = "\\setlength{\\tabcolsep}{4pt}\n" + tabular(
        "lrrrrrrr", ["Category", "neutral", "derog.", "total", "$|W^{+}(c)|$", "$|W^{-}(c)|$", "mean", "median"], rows,
        midrules=(len(rows) - 1,), header_lines=head)
    note = ("Prefixes by source word: number of prefixes whose source word (the word the prefix was mined from) is neutral or "
            "derogatory. Candidate pool: number of neutral ($|W^{+}(c)|$) and derogatory ($|W^{-}(c)|$) candidate words scored "
            "after every prefix of the category. Prefix toxicity: Detoxify \\texttt{unbiased} on the prefix text (0--1). "
            "Categories are defined by their word pools, not by their labels.")
    return table_env(body, "Prefix set and candidate pools by category.", "tab:design", note)


def table_checkpoints(cfg: dict) -> str:
    """Ch 4: the six checkpoints, post-training, input format and first token (from config.yaml)."""
    rows = []
    for key, repo in cfg["models"][cfg["run"]["size"]].items():
        kind = "base" if key == "base" else "aligned"
        fmt = INPUT_NAMES[cfg["input_format"][kind]]
        first = "\\texttt{<|endoftext|>}" if cfg["bos"][kind] else "--"
        rows.append([MODEL_LABELS[key], f"\\texttt{{{repo.replace('_', chr(92) + '_')}}}", POST_TRAINING[key], fmt, first])
    body = tabular("lllll", ["Checkpoint", "Hugging Face id", "Post-training", "Input", "First token"], rows, midrules=(1, 5))
    body = "\\resizebox{\\textwidth}{!}{%\n" + body + "%\n}"   # the Hugging Face ids overflow the text width (~131 pt)
    instr = cfg["chat"]["instruction"]
    note = ("Plain: the prefix text alone. Chat: \\texttt{<|user|>}, newline, the instruction ``" + instr + "'', newline, "
            "\\texttt{<|assistant|>}, newline, the prefix (TuluV2 format). The role markers are plain text for the tokenizer. "
            "All six checkpoints are run in float16, log-softmax in float32, batch size 1.")
    return table_env(body, "Checkpoints (Pythia 1.4B, Archangel suite) and input construction.", "tab:checkpoints", note,
                     size="footnotesize")


# ---------------------------------------------------------------- Chapter 5

def table_h1() -> str:
    """H1: each post-trained checkpoint vs Base (reference Base with BOS)."""
    h1 = pd.read_csv(S07 / "h1_tests.csv")
    h1 = h1[h1["reference"] == "base_bos"].set_index("model").loc[ALIGNED]
    rows = [[MODEL_LABELS[m], fmt_num(r["mean"], signed=True), fmt_ci(r["ci_boot_lo"], r["ci_boot_hi"]), fmt_num(np.exp(r["mean"])),
             fmt_num(r["d_z"]), f"{100 * r['share_pos']:.0f}\\%", fmt_p(r["wilcoxon_p"])]
            for m, r in h1.iterrows()]
    body = tabular("lrcrrrr", ["Checkpoint", "$\\bar{D}_M$", "95\\% CI", "$e^{\\bar{D}_M}$", "$d_z$", "$D_M>0$", "Wilcoxon $p$"],
                   rows, midrules=(4,))
    # definitions of D, e^D, d_z and the bootstrap are given in the text
    note = "$N = 500$ prefixes; $\\bar{D}_M$ in nats, 95\\% bootstrap confidence intervals; Wilcoxon $p$ two-sided, unadjusted."
    return table_env(body, "H1: each post-trained checkpoint compared with the base model", "tab:h1", note)


def table_h2() -> str:
    """H2 (six SFT-family pairs, Bonferroni x6) and H2-KTO (Direct KTO vs SFT-KTO); the Friedman test is given in the text."""
    pw = pd.read_csv(S08 / "h2_pairwise.csv")
    kto = pd.read_csv(S08 / "h2kto_tests.csv")
    kto = kto[kto["quantity"] == "D"]
    rows = []
    for _, r in pd.concat([pw, kto], ignore_index=True).iterrows():
        h2 = r["hypothesis"] == "H2"
        mark = DAG if h2 and abs(r["mean"]) <= PRECISION_BAND else ""
        rows.append([f"{MODEL_LABELS[r['m1']]} $-$ {MODEL_LABELS[r['m2']]}", fmt_num(r["mean"], signed=True) + mark,
                     fmt_ci(r["ci_boot_lo"], r["ci_boot_hi"]), fmt_num(r["exp_mean"]), fmt_num(r["d_z"]),
                     f"{100 * r['share_pos']:.0f}\\%", fmt_p(r["wilcoxon_p_bonf6"] if h2 else r["wilcoxon_p"])])
    head = ("\\multicolumn{7}{l}{\\emph{H2: SFT family (rows 1--6); H2-KTO (last row)}} \\\\", "\\addlinespace[2pt]")
    body = tabular("lrcrrrr", ["Comparison $M_1 - M_2$", "$\\bar{D}_{M_1,M_2}$", "95\\% CI", "$e^{\\bar{D}_{M_1,M_2}}$", "$d_z$",
                               "$D_{M_1,M_2}>0$", "Wilcoxon $p$"],
                   rows, midrules=(6,), header_lines=head)
    # \resizebox keeps the table at the text width (it may enlarge it slightly)
    body = "\\resizebox{\\textwidth}{!}{%\n" + body + "%\n}"
    note = ("$N = 500$ prefixes; $\\bar{D}_{M_1,M_2}$ in nats, 95\\% bootstrap confidence intervals. Rows 1--6: Wilcoxon $p$ "
            "Bonferroni-adjusted ($\\times 6$); last row: unadjusted. "
            f"{DAG}\\,Not clearly above the numerical tolerance ($|\\bar{{D}}_{{M_1,M_2}}| \\le {PRECISION_BAND:.2f}$ nats).")
    return table_env(body, "H2 and H2-KTO: paired comparisons between post-trained checkpoints.", "tab:h2", note)


def table_h3(cfg: dict) -> str:
    """H3: SFT-family mean Shift with min-max, Direct KTO mean [95% CI]; robustness variant (c') in a footnote (tests in the text)."""
    fam = pd.read_csv(S09 / "h3_family_summary.csv").set_index("category")
    sh = pd.read_csv(S09 / "h3_shift_by_category.csv")
    kto = sh[sh["model"] == "kto"].set_index("category")
    rob = pd.read_csv(S10 / "h3_shift_robust.csv")
    tw = "Transgender Woman"
    tw_wo = rob[(rob["check"] == "c_extra") & (rob["model"] == "kto") & (rob["category"] == tw)]["mean"].iloc[0]
    lab = display_labels(cfg)
    rows = []
    for c in ordered_categories(cfg):
        mark = "\\textsuperscript{a}" if c == tw else ""
        rows.append([lab[c], fmt_num(fam.loc[c, "family_mean"], signed=True),
                     fmt_range(fam.loc[c, "family_min"], fam.loc[c, "family_max"], signed=True),
                     fmt_num(kto.loc[c, "mean"], signed=True) + mark, fmt_ci(kto.loc[c, "ci_boot_lo"], kto.loc[c, "ci_boot_hi"])])
    head = ("& \\multicolumn{2}{c}{SFT family} & \\multicolumn{2}{c}{Direct KTO} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    body = tabular("lrcrc", ["Category", "$\\overline{\\mathrm{Shift}}(c)$", "min--max", "$\\mathrm{Shift}_M(c)$", "95\\% CI"], rows,
                   header_lines=head)
    note = ("In nats, over the 100 prefixes of each category. SFT family: the family shift $\\overline{\\mathrm{Shift}}(c)$, mean of "
            "$\\mathrm{Shift}_M(c)$ over the four checkpoints, and the range of $\\mathrm{Shift}_M(c)$ over them; Direct KTO: $\\mathrm{Shift}_M(c)$ with 95\\% "
            "bootstrap confidence interval. Test statistics are given in the text. "
            f"\\textsuperscript{{a}}\\,{fmt_num(tw_wo, signed=True)} without the derogatory {tw} word of variant ($c'$).")
    return table_env(body, "H3: shift by category.", "tab:h3", note)


# ---------------------------------------------------------------- Appendix

def table_h3_per_model(cfg: dict) -> str:
    """Appendix: Shift_M(c) [95% CI] for every model x category (two lines per category)."""
    sh = pd.read_csv(S09 / "h3_shift_by_category.csv").set_index(["category", "model"])
    lab = display_labels(cfg)
    rows = []
    for c in ordered_categories(cfg):
        rows.append([lab[c]] + [fmt_num(sh.loc[(c, m), "mean"], signed=True) for m in ALIGNED])
        rows.append([""] + ["{\\scriptsize " + fmt_ci(sh.loc[(c, m), "ci_boot_lo"], sh.loc[(c, m), "ci_boot_hi"]) + "}" for m in ALIGNED])
    body = tabular("lccccc", ["Category"] + [MODEL_LABELS[m] for m in ALIGNED], rows, midrules=tuple(range(2, len(rows), 2)))
    note = ("$\\mathrm{Shift}_M(c)$ in nats over the 100 prefixes of each cell; second line: 95\\% bootstrap confidence "
            "interval.")
    return table_env(body, "Shift by category and checkpoint.", "tab:h3-per-model", note)


# Row labels shortened for print; every key must occur in the robustness summary table.
GATE_LABELS = {"H3 signs: family Shift > 0 (20 cells)": "H3 signs, SFT family $>0$",
               "H3 sign: Direct KTO Transgender Woman < 0": "H3 sign, Direct KTO TW $<0$",
               "H3 order: Sex Worker lowest in 4/4 family": "H3 order, Sex Worker lowest",
               "H2-KTO (Direct KTO − SFT-KTO)": "H2-KTO"}


def _gate_cell(text: str) -> str:
    """Shorten the robustness labels for print; the meaning is unchanged (legend in the appendix prose).
    The k/m subset counts are dropped (a result holds in (a), (e) only if in both subsets, in (c) only if for all nine words);
    a label that holds in one of the two (a) subsets only is printed as partly, with a dagger."""
    if re.fullmatch(r"changes \(.+\) 1/2 \(original\); holds 1/2", text):
        return "partly$^{\\dagger}$"
    text = re.sub(r" \d+/\d+$", "", text)
    return {"not interpretable": "n.i.", "holds (by construction)": "--", "same": "holds"}.get(text, text)


def table_gate(cfg: dict) -> str:
    """Appendix: robustness summary table (one row per result, one column per check); labels and marks are explained in the text."""
    g = pd.read_csv(S10 / "gate_table.csv")
    star = null_kto_significant(pd.read_csv(S10 / "h1_robust.csv"))
    lab = display_labels(cfg)
    assert set(GATE_LABELS) <= set(g["result"]), set(GATE_LABELS) - set(g["result"])
    rows = []
    for _, r in g.iterrows():
        name = GATE_LABELS.get(r["result"], r["result"]).replace("KTO (direct)", "Direct KTO")
        for internal, shown in lab.items():
            name = name.replace(internal, shown)
        cells = []
        for col in ["a", "b", "c", "c_extra", "d", "e"]:
            cell = _gate_cell(r[col])
            if name == "H1 Direct KTO" and col in set(star["check"]):
                cell += "$^{*}$"
            cells.append(cell)
        rows.append([tex_minus(name)] + cells)
    head = ("& (a) & (b) & (c) & ($c'$) & (d) & (e) \\\\",)
    head_row = ["Result", "corpus", "unedited", "homogr.", "TW word", "no BOS", "halves"]
    body = "\\setlength{\\tabcolsep}{4pt}\n" + tabular("lcccccc", head_row, rows, midrules=(5, 14), header_lines=head)
    caption = ("Robustness checks: whether each result of the main analysis holds in each variant; labels and marks are explained "
               "in the text.")
    return table_env(body, caption, "tab:robustness", size="small", short="Robustness checks")


def null_kto_significant(rob: pd.DataFrame) -> pd.DataFrame:
    """Subsets of checks (a), (b), (e) where the null Direct KTO H1 result 'holds' (CI includes 0) while Wilcoxon p < 0.05 (marked with an asterisk)."""
    k = rob[(rob["model"] == "kto") & (rob["reference"] == "base_bos") & rob["check"].isin(["a", "b", "e"])]
    star = k[(k["ci_boot_lo"] < 0) & (k["ci_boot_hi"] > 0) & (k["wilcoxon_p"] < 0.05)]
    assert set(star["subset"]) == set(SUBSET_NAMES), star["subset"].tolist()   # all three subsets, as in the robustness report
    return star


# ---------------------------------------------------------------- checks

def internal_names_found(text: str, cfg: dict) -> list[str]:
    """Internal category names or the old 'KTO (direct)' label in a fragment (must be empty)."""
    bad = [c for c, shown in display_labels(cfg).items() if c != shown and c in text]
    bad += [w for w in ["KTO (direct)", "African", "Female Homosexual", "Prostitute"] if w in text]
    return bad


def derogatory_words_found(text: str) -> list[str]:
    """Derogatory vocabulary words (whole word, case-insensitive) in a fragment (must be empty)."""
    voc = load_vocabulary()
    words = voc.loc[voc["polarity"] == "NOT SAFE", "word"].unique()
    return [w for w in words if re.search(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", text, flags=re.I)]


def dropped_terms_found(text: str) -> list[str]:
    """Terms that the thesis tables no longer use (statistics not reported, internal pool labels, old layout) (must be empty)."""
    return [t for t in ["McNemar", "I_M", "minipage", "SAFE", "vocabulary word"] if t in text]


def structure_ok(text: str) -> bool:
    """Balanced braces; no '<', '>', '|' in text mode (OT1 fonts print them as other glyphs; allowed in math and \\texttt);
    in every tabular, the same number of unescaped '&' on each data row (rows without \\multicolumn)."""
    if text.count("{") - text.count("\\{") != text.count("}") - text.count("\\}"):
        return False
    plain = re.sub(r"\\texttt\{[^}]*\}", "", re.sub(r"\$[^$]*\$", "", text))
    if re.search(r"[<>|]", plain):
        return False
    for tab in re.findall(r"\\begin\{tabular\}\{([^\n]*)\}\n(.*?)\\end\{tabular\}", text, flags=re.S):
        spec = tab[0]
        ncol = len(re.findall(r"p\{[^}]*\}", spec)) + len(re.findall(r"[lrc]", re.sub(r"p\{[^}]*\}", "", spec)))
        for line in tab[1].splitlines():
            if line.endswith("\\\\") and "\\multicolumn" not in line:
                if len(re.findall(r"(?<!\\)&", line)) != ncol - 1:
                    return False
    return True


def number_checks() -> list[dict]:
    """Numbers in the tables vs the sentences of the thesis text (expected strings copied from the text).

    Non-blocking: a mismatch means the sentence and the CSV disagree on rounding or wording, which is fixed in the text, not here.
    """
    h1 = pd.read_csv(S07 / "h1_tests.csv").query("reference == 'base_bos'").set_index("model")
    fr = pd.read_csv(S08 / "h2_friedman.csv").iloc[0]
    pw = pd.read_csv(S08 / "h2_pairwise.csv")
    kto = pd.read_csv(S08 / "h2kto_tests.csv").query("quantity == 'D'").iloc[0]
    t = pd.read_csv(S09 / "h3_tests.csv").query("reference == 'base_bos'")
    f = t[t["model"].isin(FAMILY)]
    sh = pd.read_csv(S09 / "h3_shift_by_category.csv").set_index(["model", "category"])
    fam = h1.loc[FAMILY]
    got = {
        "H1 SFT-family mean D": f"+{fam['mean'].min():.2f} to +{fam['mean'].max():.2f}",
        "H1 SFT-family d_z": f"{fam['d_z'].min():.2f}-{fam['d_z'].max():.2f}",
        "H1 Direct KTO": f"+{h1.loc['kto', 'mean']:.2f}, 95% CI [{h1.loc['kto', 'ci_boot_lo']:.2f}, {h1.loc['kto', 'ci_boot_hi']:.2f}]",
        "H2 Friedman p": f"{fr['p']:.1e}".replace("e-", "e-"),
        "H2 largest gap": f"{pw['mean'].abs().max():.2f} nat",
        "H2 Kendall W": f"{fr['kendall_w']:.2f}",
        "H2 pairs at precision scale": str(int((pw["mean"].abs() <= PRECISION_BAND).sum())),
        "H2-KTO mean, d_z": f"{-kto['mean']:.2f} nat less, d_z {kto['d_z']:.2f}",
        "H3 KW H": f"{f['kw_H'].min():.0f}-{f['kw_H'].max():.0f}",
        "H3 epsilon^2": f"{f['kw_epsilon2'].min():.2f}-{f['kw_epsilon2'].max():.2f}",
        "H3 ANCOVA F": f"{f['ancova_F_category'].min():.1f}-{f['ancova_F_category'].max():.1f}",
        "H3 range raw / adjusted": f"{f['range_raw'].min():.2f}-{f['range_raw'].max():.2f} / {f['range_adj'].min():.2f}-{f['range_adj'].max():.2f}",
        "H3 Direct KTO Transgender Woman": (f"{sh.loc[('kto', 'Transgender Woman'), 'mean']:.2f} [{sh.loc[('kto', 'Transgender Woman'), 'ci_boot_lo']:.2f}, "
                                            f"{sh.loc[('kto', 'Transgender Woman'), 'ci_boot_hi']:.2f}]"),
    }
    expected = {
        "H1 SFT-family mean D": "+0.43 to +0.56", "H1 SFT-family d_z": "0.62-0.79",
        "H1 Direct KTO": "+0.02, 95% CI [-0.04, 0.08]", "H2 Friedman p": "1.2e-82", "H2 largest gap": "0.13 nat",
        "H2 Kendall W": "0.26", "H2 pairs at precision scale": "3", "H2-KTO mean, d_z": "0.41 nat less, d_z -0.91",
        "H3 KW H": "39-59", "H3 epsilon^2": "0.08-0.12", "H3 ANCOVA F": "11.5-16.8", "H3 range raw / adjusted": "0.49-0.64 / 0.51-0.64",
        "H3 Direct KTO Transgender Woman": "-0.54 [-0.72, -0.37]",
    }
    return [dict(check=f"number: {k}", value=got[k], expected=expected[k], passed=got[k] == expected[k], blocking=False)
            for k in expected]


def main() -> None:
    cfg = load_config()
    TABLES.mkdir(parents=True, exist_ok=True)
    THESIS_TABLES.mkdir(parents=True, exist_ok=True)
    frags = {
        "tab_design": (table_design(cfg), ["data/final_prefix_set.csv", "data/Vocabulary_Final.xlsx (counts)",
                                           "reports/step_06_tables/distribution_by_category_polarity.csv"]),
        "tab_checkpoints": (table_checkpoints(cfg), ["config.yaml"]),
        "tab_h1": (table_h1(), ["reports/step_07_tables/h1_tests.csv"]),
        "tab_h2": (table_h2(), ["reports/step_08_tables/h2_pairwise.csv", "h2kto_tests.csv"]),
        "tab_h3": (table_h3(cfg), ["reports/step_09_tables/h3_family_summary.csv", "h3_shift_by_category.csv",
                                   "reports/step_10_tables/h3_shift_robust.csv"]),
        "tab_h3_per_model": (table_h3_per_model(cfg), ["reports/step_09_tables/h3_shift_by_category.csv"]),
        "tab_robustness": (table_gate(cfg), ["reports/step_10_tables/gate_table.csv", "h1_robust.csv"]),
    }
    checks = []
    for name, (text, sources) in frags.items():
        write_fragment(TABLES / f"{name}.tex", text, sources)
        shutil.copyfile(TABLES / f"{name}.tex", THESIS_TABLES / f"{name}.tex")
        same = filecmp.cmp(TABLES / f"{name}.tex", THESIS_TABLES / f"{name}.tex", shallow=False)
        checks += [dict(check=f"{name}: internal labels", value=";".join(internal_names_found(text, cfg)) or "none", expected="none"),
                   dict(check=f"{name}: derogatory words", value=str(len(derogatory_words_found(text))), expected="0"),
                   dict(check=f"{name}: dropped terms", value=";".join(dropped_terms_found(text)) or "none", expected="none"),
                   dict(check=f"{name}: structure (braces, & per row)", value=str(structure_ok(text)), expected="True"),
                   dict(check=f"{name}: copy in thesis/tables identical", value=str(same), expected="True")]
    for c in checks:
        c["passed"], c["blocking"] = c["value"] == c["expected"], True
    checks += number_checks()
    out = pd.DataFrame(checks)
    out.to_csv(OUT / "gate_checks_tables.csv", index=False)
    print(f"tables: {len(frags)} written to {TABLES.relative_to(ROOT)}; checks {int(out['passed'].sum())}/{len(out)} pass")
    if not out["passed"].all():
        print(out[~out["passed"]].to_string(index=False))
    if not out.loc[out["blocking"], "passed"].all():
        raise SystemExit("blocking check failed")


if __name__ == "__main__":
    main()
