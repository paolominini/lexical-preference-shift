"""Robustness checks: do the main results of 07-09 survive the five checks (a-e)?

No re-scoring. Each check builds a variant of the delta / H3 frames and reruns the main statistics through the SAME functions as
07-09 (loaded from scripts/07_rq1.py, 08_rq2.py, 09_rq3.py and 05_derived.py via importlib; same N_BOOT and seed):
  (a) corpus group: 'added' (Civil Comments + SBIC, 242) vs 'original' (HateXplain, DynaHate, MHS, ETHOS; 258);
      descriptive extra: H1 per single corpus (ETHOS excluded, n = 2).
  (b) without the 77 edited (a/an -> the) prefixes (423 left).
  (c) leave-one-word-out for the nine homograph-prone words, one at a time, removed from Base and post-trained scores alike;
      extra (reported separately): the 3-token derogatory word of Transgender Woman.
  (d) reference = Base without BOS; H2 / H2-KTO hold by construction (Base cancels; aligned Delta bit-identical, checked).
  (e) random split in halves, stratified by category (seed from config.yaml), each half rerun.
Survival rule (fixed before running; robustness.py in the package): holds / weakened / weakened (precision scale) / reversed /
not interpretable (pairs at the precision scale of 08) / changes (null H1-KTO whose CI no longer includes 0; outcome labels that change).

Reads: results/{delta_1.4b, delta_1.4b_base_nobos, scores_1.4b, scores_1.4b_base_nobos, toxicity}.parquet,
data/final_prefix_set.csv (read only), reports/step_07..09_tables/*.csv (check 1 only).
Writes (never overwriting a different file) into --out (default reports/step_10_tables/):
  subsets.csv, pool_sizes_lowo.csv, h1_robust.csv, h1_by_corpus.csv, h2_robust.csv, h2kto_robust.csv, h3_tests_robust.csv,
  h3_shift_robust.csv, shift_both_references.csv, bos_per_word.csv, gate_table.csv, gate_checks.csv
Usage: .venv/bin/python scripts/10_robustness.py [--out DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probing.config import load_config  # noqa: E402
from probing.derived import identity_error  # noqa: E402
from probing.pools import load_prefixes  # noqa: E402
from probing.robustness import (CHANGES, HOLDS, SAME, aggregate_labels, corpus_group, drop_word, exclude_edited,  # noqa: E402
                                outcome_label, pool_sizes, stratified_halves, survival_label)
from probing.stats import bootstrap_mean_ci  # noqa: E402


def load_script(name: str) -> ModuleType:
    """Import a numbered script as a module (its main() is guarded, so nothing runs)."""
    spec = importlib.util.spec_from_file_location(f"s{name[:2]}", ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S05, S07, S08, S09 = (load_script(n) for n in ["05_derived.py", "07_rq1.py", "08_rq2.py", "09_rq3.py"])
ALPHA, PRECISION, CLEARLY_ABOVE = S08.ALPHA, S08.PRECISION, S08.CLEARLY_ABOVE
FAMILY, ALIGNED = S09.FAMILY, ["sft", "sft_dpo", "sft_ppo", "sft_kto", "kto"]
SW, TW = "Sex Worker, Prostitute", "Transgender Woman"
LOWO = [("Sex Worker, Prostitute", "escort"), ("Transgender Woman", "tranny"), ("Female Homosexual", "dyke"),
        ("Homosexual Man", "bender"), ("Homosexual Man", "poof"), ("Homosexual Man", "nancy"),
        ("African - Afro American", "coon"), ("Transgender Woman", "shim"), ("Sex Worker, Prostitute", "hoe")]   # check (c): homograph-prone words
NUM_TOL = 1e-12


# ---------------------------------------------------------------- variant runs
def wide(dm: pd.DataFrame, col: str) -> pd.DataFrame:
    """Prefix x model table of `col` (08 `wide` without its n = 500 assertion, so subsets work)."""
    w = dm.pivot(index="prefix_id", columns="model", values=col).sort_index()
    assert not w.isna().any().any(), col
    return w


def h3_frame(base_frame: pd.DataFrame, dm: pd.DataFrame, dn: pd.DataFrame) -> pd.DataFrame:
    """H3 frame (covariates from 09 `load_frame`, coded on all 500 prefixes) with y / y_nobos taken from a variant's delta files."""
    y = dm[dm["model"] != "base"][["prefix_id", "model", "D"]].rename(columns={"D": "y"})
    yn = dn[["prefix_id", "model", "D"]].rename(columns={"D": "y_nobos"})
    f = base_frame.drop(columns=["y", "y_nobos"]).merge(y, on=["prefix_id", "model"], validate="one_to_one")
    f = f.merge(yn, on=["prefix_id", "model"], validate="one_to_one")
    return f.sort_values(["model", "prefix_id"]).reset_index(drop=True)


def h3_tests(frame: pd.DataFrame, yname: str, reference: str) -> pd.DataFrame:
    """Per-model H3 tests of 09 (with rank / 3-bin variants) and its family outcome rule, for any reference."""
    rows = [S09.model_tests(frame[frame["model"] == m], yname, reference, True)[0] for m in ALIGNED]
    t = pd.DataFrame(rows)
    fam = t[t["model"].isin(FAMILY)]
    if (fam["verdict"] == "differ").all() and fam["variants_all_significant"].all():
        outcome = "supported"
    elif (fam["verdict"] != "differ").all():
        outcome = "not supported"
    else:
        outcome = "mixed"
    t["h3_sft_family_outcome"] = np.where(t["model"].isin(FAMILY), outcome, None)   # as in 09: family rows only
    return t


def shift_rows(frame: pd.DataFrame, yname: str, seed: int) -> pd.DataFrame:
    """Shift_M(c): mean, bootstrap 95% CI (as in 09: 1000 resamples of the category's prefixes, seed), n, share > 0."""
    rows = []
    for (m, cat), g in frame.groupby(["model", "category"]):
        y = g[yname].to_numpy()
        lo, hi = bootstrap_mean_ci(y, S09.N_BOOT, seed)
        rows.append(dict(model=m, category=cat, n_prefixes=len(y), mean=y.mean(), ci_boot_lo=lo, ci_boot_hi=hi,
                         ci_excludes_0=bool(lo > 0 or hi < 0), share_pos=float((y > 0).mean())))
    return pd.DataFrame(rows)


def run_variant(dm: pd.DataFrame, dn: pd.DataFrame, frame: pd.DataFrame, ids: list[str], seed: int) -> dict[str, pd.DataFrame]:
    """All main statistics on one variant (prefix subset `ids`, delta files dm / dn, H3 frame)."""
    dm, dn, frame = (x[x["prefix_id"].isin(ids)] for x in (dm, dn, frame))
    h1 = S07.h1_table(dm, dn, ALIGNED, seed)
    al = dm[dm["model"] != "base"]
    delta, ind = wide(al, "delta"), wide(al, "I_M").astype(bool)
    lp_pos, lp_neg = wide(al, "logp_pos"), wide(al, "logp_neg")
    fried = S08.friedman_table(delta)
    pw = S08.pairwise_table(delta, ind, float(fried["p"].iloc[0]), seed)
    fried["h2_outcome"] = "supported" if (fried["p"].iloc[0] < ALPHA and pw["counts_for_h2_support"].any()) else "not supported"
    kto = S08.h2kto_table(delta, ind, S08.components(*S08.KTO_PAIR, lp_pos, lp_neg, delta), seed)
    return dict(h1=h1, fried=fried, pw=pw, kto=kto, h3=h3_tests(frame, "y", "base_bos"), shift=shift_rows(frame, "y", seed),
                h3_nobos=h3_tests(frame, "y_nobos", "base_nobos"), shift_nobos=shift_rows(frame, "y_nobos", seed))


# ---------------------------------------------------------------- survival
def h3_descriptive(shift: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """The three H3 descriptive rows on the point estimates: (label, detail)."""
    s = shift.set_index(["model", "category"])["mean"]
    fam_neg = [f"{m}/{c[:12]}" for (m, c), v in s.items() if m in FAMILY and v <= 0]
    kto_tw = s[("kto", TW)]
    not_lowest = [m for m in FAMILY if s[m].idxmin() != SW]
    return {"H3 signs: family Shift > 0 (20 cells)": (SAME if not fam_neg else CHANGES, "; ".join(fam_neg)),
            "H3 sign: Direct KTO Transgender Woman < 0": (SAME if kto_tw < 0 else CHANGES, f"{kto_tw:+.3f}"),
            "H3 order: Sex Worker lowest in 4/4 family": (SAME if not not_lowest else CHANGES, ", ".join(not_lowest))}


def labels_for(main: dict, var: dict, h1_ref: str = "base_bos", h3_key: str = "h3", shift_key: str = "shift") -> dict[str, str]:
    """Gate-table labels of one variant against the main (full-sample) run."""
    out = {}
    mh1 = main["h1"].set_index(["reference", "model"])
    vh1 = var["h1"].set_index(["reference", "model"])
    for m in ALIGNED:
        a, b = mh1.loc[("base_bos", m)], vh1.loc[(h1_ref, m)]
        sig = bool((a["ci_boot_lo"] > 0 or a["ci_boot_hi"] < 0) and a["wilcoxon_p"] < ALPHA)
        out[f"H1 {S07.LABEL[m]}"] = survival_label(a["mean"], sig, b["mean"], b["ci_boot_lo"], b["ci_boot_hi"], b["wilcoxon_p"])
    out["H2 Friedman"] = HOLDS if var["fried"]["p"].iloc[0] < ALPHA else "weakened"
    for (_, a), (_, b) in zip(main["pw"].iterrows(), var["pw"].iterrows()):
        sig = bool((a["ci_boot_lo"] > 0 or a["ci_boot_hi"] < 0) and a["wilcoxon_p_bonf6"] < ALPHA)
        out[f"H2 {a['pair']}"] = survival_label(a["mean"], sig, b["mean"], b["ci_boot_lo"], b["ci_boot_hi"], b["wilcoxon_p_bonf6"],
                                                precision_scale=abs(a["mean"]) <= CLEARLY_ABOVE)
    out["H2 outcome"] = outcome_label(main["fried"]["h2_outcome"].iloc[0], var["fried"]["h2_outcome"].iloc[0])
    a, b = main["kto"].iloc[0], var["kto"].iloc[0]
    out["H2-KTO (Direct KTO − SFT-KTO)"] = survival_label(a["mean"], True, b["mean"], b["ci_boot_lo"], b["ci_boot_hi"], b["wilcoxon_p"])
    mh3, vh3 = main["h3"].set_index("model"), var[h3_key].set_index("model")
    for m in ALIGNED:
        out[f"H3 verdict {S07.LABEL[m]}"] = outcome_label(mh3.loc[m, "verdict"], vh3.loc[m, "verdict"])
    out["H3 family outcome"] = outcome_label(mh3["h3_sft_family_outcome"].iloc[0], vh3["h3_sft_family_outcome"].iloc[0])
    out.update({k: v[0] for k, v in h3_descriptive(var[shift_key]).items()})
    return out


# ---------------------------------------------------------------- tables
def subsets_table(prefixes: pd.DataFrame, groups: pd.Series, halves: pd.Series, unedited: list[str], seed: int) -> pd.DataFrame:
    """n per check x subset x category x source polarity (ALL rows too), with n edited; the split seed in every (e) row."""
    p = prefixes.set_index("final_id")
    subs = [("full", "all", p.index), ("a", "added", groups.index[groups == "added"]), ("a", "original", groups.index[groups == "original"]),
            ("b", "unedited", pd.Index(unedited)), ("e", "A", halves.index[halves == "A"]), ("e", "B", halves.index[halves == "B"])]
    rows = []
    for check, sub, ids in subs:
        q = p.loc[ids]
        for cat, g in [("ALL", q)] + list(q.groupby("category")):
            for pol, h in [("ALL", g)] + list(g.groupby("polarity")):
                rows.append(dict(check=check, subset=sub, category=cat, source_polarity=pol, n_prefixes=len(h),
                                 n_edited=int(h["edited"].astype(bool).sum()), split_seed=seed if check == "e" else None))
    return pd.DataFrame(rows)


def lowo_pool_table(scores: pd.DataFrame, words: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Pool size before / after removing each word; flag pools left with one word."""
    before = pool_sizes(scores).set_index(["category", "polarity_word"])["n_words"]
    pol = scores.drop_duplicates(["category", "word"]).set_index(["category", "word"])["polarity_word"]
    rows = []
    for sub, cat, w in words:
        pw = pol[(cat, w)]
        rows.append(dict(subset=sub, category=cat, word=w, pool=pw, n_words_before=int(before[(cat, pw)]),
                         n_words_after=int(before[(cat, pw)]) - 1, one_word_pool=int(before[(cat, pw)]) - 1 == 1))
    return pd.DataFrame(rows)


def bos_per_word(scores: pd.DataFrame, nobos: pd.DataFrame) -> pd.DataFrame:
    """Per (category, word): mean Base logP with BOS, without BOS, and their difference over the category's prefixes."""
    b = scores[scores["model"] == "base"][["prefix_id", "category", "word", "polarity_word", "n_tokens", "logp"]]
    n = nobos[["prefix_id", "word", "logp"]].rename(columns={"logp": "logp_nobos"})
    d = b.merge(n, on=["prefix_id", "word"], validate="one_to_one")
    d["bos_minus_nobos"] = d["logp"] - d["logp_nobos"]
    out = d.groupby(["category", "polarity_word", "word", "n_tokens"]).agg(
        n_prefixes=("prefix_id", "size"), mean_logp_bos=("logp", "mean"), mean_logp_nobos=("logp_nobos", "mean"),
        mean_bos_minus_nobos=("bos_minus_nobos", "mean"), sd_bos_minus_nobos=("bos_minus_nobos", "std")).reset_index()
    return out


def gate_cell_table(cells: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Rows = main results, columns = checks; `cells[check][row]` is already aggregated."""
    rows = list(next(iter(cells.values())).keys())
    return pd.DataFrame([{"result": r, **{c: cells[c][r] for c in cells}} for r in rows])


def write_csv_new(df: pd.DataFrame, path: Path) -> None:
    """Never overwrite silently. An existing file is kept only if its text equals the new table; otherwise stop."""
    text = df.to_csv(index=False)
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} exists and differs from the recomputed table: not overwritten")
        return
    path.write_text(text)


# ---------------------------------------------------------------- checks
def max_rel_diff(new: pd.DataFrame, old: pd.DataFrame) -> float:
    """Max |new - old| / max(1, |old|) over the numeric columns both tables share (same row order)."""
    cols = [c for c in old.columns if c in new.columns and pd.api.types.is_numeric_dtype(old[c]) and old[c].dtype != bool]
    a, b = new[cols].astype(float).to_numpy(), old[cols].astype(float).to_numpy()
    assert a.shape == b.shape
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.where(both_nan, 0.0, np.abs(a - b) / np.maximum(1.0, np.abs(b)))
    return float(np.nanmax(np.where(np.isnan(diff), np.inf, diff)))


def text_cols_equal(new: pd.DataFrame, old: pd.DataFrame, cols: list[str]) -> bool:
    """Same text in `cols`, row by row (missing values as empty strings: None and NaN are the same empty CSV cell)."""
    def norm(x: pd.Series) -> np.ndarray:
        return x.astype(object).where(x.notna(), "").astype(str).to_numpy()
    return all((norm(new[c].reset_index(drop=True)) == norm(old[c].reset_index(drop=True))).all() for c in cols)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "reports/step_10_tables")
    out_dir = ap.parse_args().out
    cfg = load_config()
    res = ROOT / cfg["paths"]["results"]
    seed = int(cfg["project"]["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    prefixes = load_prefixes()
    raw_prefixes = pd.read_csv(ROOT / cfg["paths"]["prefixes"])
    dm = pd.read_parquet(res / "delta_1.4b.parquet")
    dn = pd.read_parquet(res / "delta_1.4b_base_nobos.parquet")
    scores = pd.read_parquet(res / "scores_1.4b.parquet")
    nobos = pd.read_parquet(res / "scores_1.4b_base_nobos.parquet")
    frame0, _ = S09.load_frame(res, raw_prefixes)
    all_ids = sorted(prefixes["final_id"])
    gates = []

    # main run on the full sample (reference for every label; check 1 against the 07-09 tables)
    main_run = run_variant(dm, dn, frame0, all_ids, seed)
    t7, t8, t9 = (ROOT / f"reports/step_0{k}_tables" for k in (7, 8, 9))
    h3_s9, _ = S09.tests_tables(frame0)
    for name, new, old, keys in [
            ("h1_tests (both references)", main_run["h1"], pd.read_csv(t7 / "h1_tests.csv"), ["reference", "model"]),
            ("h2_friedman", main_run["fried"], pd.read_csv(t8 / "h2_friedman.csv"), ["h2_outcome"]),
            ("h2_pairwise", main_run["pw"], pd.read_csv(t8 / "h2_pairwise.csv"), ["pair", "counts_for_h2_support"]),
            ("h2kto_tests", main_run["kto"], pd.read_csv(t8 / "h2kto_tests.csv"), ["quantity", "h2kto_outcome"]),
            ("h3_tests (S9 function, both references)", h3_s9, pd.read_csv(t9 / "h3_tests.csv"), ["reference", "model", "verdict"]),
            ("h3_tests (S10 wrapper, base_bos)", main_run["h3"], h3_s9[h3_s9["reference"] == "base_bos"].reset_index(drop=True),
             ["model", "verdict", "h3_sft_family_outcome"]),
            ("h3_shift_by_category", main_run["shift"], pd.read_csv(t9 / "h3_shift_by_category.csv")
             .sort_values(["model", "category"]).reset_index(drop=True), ["model", "category"])]:
        err = max_rel_diff(new.reset_index(drop=True), old)
        gates.append(dict(check=f"1. full sample reproduces {name}", value=err, pass_=err <= NUM_TOL and text_cols_equal(new, old, keys)))

    # check 2: LOWO machinery with nothing removed reproduces the 05 delta files
    dm0 = S05.build_delta_main(scores)
    dn0 = S05.build_delta_nobos(scores[scores["model"] != "base"], nobos)
    for name, a, b in [("delta_1.4b", dm0, dm), ("delta_1.4b_base_nobos", dn0, dn)]:
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)
            ok = True
        except AssertionError:
            ok = False
        gates.append(dict(check=f"2a. rebuilding {name} from scores (no word removed) = file", value=0.0 if ok else np.nan, pass_=ok))

    # (d) by construction: aligned Delta identical with and without the Base-BOS reference
    al = dm[dm["model"] != "base"].set_index(["model", "prefix_id"])["delta"].sort_index()
    aln = dn.set_index(["model", "prefix_id"])["delta"].sort_index()
    gates.append(dict(check="4. aligned Delta identical in main and no-BOS files (H2/H2-KTO hold by construction in d)",
                      value=float((al - aln).abs().max()), pass_=bool((al == aln).all())))

    # subsets
    groups = corpus_group(prefixes)
    halves = stratified_halves(prefixes, seed)
    unedited = exclude_edited(prefixes)
    sub_ids = {"a": {"added": sorted(groups.index[groups == "added"]), "original": sorted(groups.index[groups == "original"])},
               "b": {"unedited": sorted(unedited)},
               "e": {"A": sorted(halves.index[halves == "A"]), "B": sorted(halves.index[halves == "B"])}}
    cat = prefixes.set_index("final_id")["category"]
    gates += [dict(check="3a. (b) unedited = 423 (77 removed)", value=len(unedited), pass_=len(unedited) == 423),
              dict(check="3b. (a) added 242 / original 258, disjoint, union 500", value=len(sub_ids["a"]["added"]),
                   pass_=len(sub_ids["a"]["added"]) == 242 and len(sub_ids["a"]["original"]) == 258
                   and set(sub_ids["a"]["added"]).isdisjoint(sub_ids["a"]["original"])),
              dict(check=f"3c. (e) halves 250/250, 50 per category each, disjoint (seed {seed})", value=len(sub_ids["e"]["A"]),
                   pass_=bool((pd.crosstab(cat.loc[halves.index], halves) == 50).all().all())
                   and set(sub_ids["e"]["A"]).isdisjoint(sub_ids["e"]["B"]))]

    runs: dict[tuple[str, str], dict] = {}
    for check, d in sub_ids.items():
        for sub, ids in d.items():
            runs[(check, sub)] = run_variant(dm, dn, frame0, ids, seed)

    # (c) leave-one-word-out: rebuild delta files without the word, Base and post-trained alike
    tw3 = scores[(scores["category"] == TW) & (scores["polarity_word"] == "NOT SAFE") & (scores["n_tokens"] == 3)]["word"].unique()
    assert len(tw3) == 1, "expected exactly one 3-token TW derogatory word"
    lowo_words = [(f"c_{w}", c, w) for c, w in LOWO] + [("c_extra_TW_3token", TW, tw3[0])]
    id_err, untouched_ok = 0.0, True
    for sub, c, w in lowo_words:
        s_c, n_c = drop_word(scores, c, w), drop_word(nobos, c, w)
        dm_c = S05.build_delta_main(s_c)
        dn_c = S05.build_delta_nobos(s_c[s_c["model"] != "base"], n_c)
        id_err = max(id_err, identity_error(dm_c.dropna(subset=["D"])), identity_error(dn_c))
        other = dm_c["category"] != c
        untouched_ok &= bool((dm_c.loc[other, ["delta", "D"]].fillna(0).to_numpy() == dm.loc[other, ["delta", "D"]].fillna(0).to_numpy()).all())
        check = "c_extra" if sub == "c_extra_TW_3token" else "c"
        runs[(check, sub)] = run_variant(dm_c, dn_c, h3_frame(frame0, dm_c, dn_c), all_ids, seed)
    gates += [dict(check="2b. D = delta_pos - delta_neg in every LOWO variant (max abs error)", value=id_err, pass_=id_err < 1e-9),
              dict(check="2c. LOWO: categories without the removed word bit-identical to main", value=0.0, pass_=untouched_ok)]

    # labels -> summary table (gate_table.csv)
    cells: dict[str, dict[str, str]] = {}
    per_subset_labels = []
    for check in ["a", "b", "c", "c_extra", "e"]:
        labs = {sub: labels_for(main_run, r) for (ch, sub), r in runs.items() if ch == check}
        rows = next(iter(labs.values())).keys()
        cells[check] = {row: aggregate_labels({s.removeprefix("c_"): labs[s][row] for s in labs}) for row in rows}
        per_subset_labels += [dict(check=check, subset=s, result=row, label=lab) for s in labs for row, lab in labs[s].items()]
    d_labels = labels_for(main_run, main_run, h1_ref="base_nobos", h3_key="h3_nobos", shift_key="shift_nobos")
    d_labels.update({k: "holds (by construction)" for k in d_labels if k.startswith("H2")})
    cells["d"] = d_labels
    per_subset_labels += [dict(check="d", subset="base_nobos", result=row, label=lab) for row, lab in d_labels.items()]
    gate_table = gate_cell_table({k: cells[k] for k in ["a", "b", "c", "c_extra", "d", "e"]})

    # long result tables
    def collect(key: str, ref_filter: str | None = None) -> pd.DataFrame:
        parts = [main_run[key].assign(check="full", subset="all")]
        parts += [r[key].assign(check=ch, subset=sub) for (ch, sub), r in runs.items()]
        t = pd.concat(parts, ignore_index=True)
        if ref_filter is not None:
            t = t[t["reference"] == ref_filter]
        return t

    h1_r = pd.concat([collect("h1", "base_bos"),
                      main_run["h1"][main_run["h1"]["reference"] == "base_nobos"].assign(check="d", subset="base_nobos")], ignore_index=True)
    h2_r = pd.concat([collect("fried").assign(pair="Friedman (SFT family)"), collect("pw")], ignore_index=True)
    kto_r = collect("kto")
    h3_r = pd.concat([collect("h3"), main_run["h3_nobos"].assign(check="d", subset="base_nobos")], ignore_index=True)
    shift_r = pd.concat([collect("shift"), main_run["shift_nobos"].assign(check="d", subset="base_nobos")], ignore_index=True)
    main_means = main_run["shift"].set_index(["model", "category"])["mean"]
    shift_r["main_mean"] = [main_means[(m, c)] for m, c in zip(shift_r["model"], shift_r["category"])]
    shift_r["sign_kept"] = np.sign(shift_r["mean"]) == np.sign(shift_r["main_mean"])
    rho = shift_r.groupby(["check", "subset", "model"]).apply(
        lambda g: stats.spearmanr(g["mean"], g["main_mean"]).statistic, include_groups=False).rename("spearman_vs_main_category_means")
    shift_r = shift_r.merge(rho.reset_index(), on=["check", "subset", "model"])
    both = main_run["shift"].merge(main_run["shift_nobos"], on=["model", "category", "n_prefixes"], suffixes=("_bos", "_nobos"))
    both["nobos_minus_bos"] = both["mean_nobos"] - both["mean_bos"]

    by_corpus = []
    for corpus in ["civil", "hatexplain", "dynahate", "mhs", "sbic"]:
        ids = prefixes.loc[prefixes["corpus"] == corpus, "final_id"]
        h = S07.h1_table(dm[dm["prefix_id"].isin(ids)], dn[dn["prefix_id"].isin(ids)], ALIGNED, seed)
        by_corpus.append(h[h["reference"] == "base_bos"].assign(corpus=corpus))
    by_corpus = pd.concat(by_corpus, ignore_index=True)

    gate_df = pd.DataFrame(gates).rename(columns={"pass_": "pass"})
    tables = [("subsets", subsets_table(prefixes, groups, halves, unedited, seed)), ("pool_sizes_lowo", lowo_pool_table(scores, lowo_words)),
              ("h1_robust", h1_r), ("h1_by_corpus", by_corpus), ("h2_robust", h2_r), ("h2kto_robust", kto_r),
              ("h3_tests_robust", h3_r), ("h3_shift_robust", shift_r), ("shift_both_references", both),
              ("bos_per_word", bos_per_word(scores, nobos)), ("survival_labels", pd.DataFrame(per_subset_labels)),
              ("gate_table", gate_table), ("gate_checks", gate_df)]
    for name, t in tables:
        write_csv_new(t, out_dir / f"{name}.csv")

    print(f"checks: {int(gate_df['pass'].sum())}/{len(gate_df)} pass")
    if not gate_df["pass"].all():
        print(gate_df[~gate_df["pass"]].to_string(index=False))
    with pd.option_context("display.max_colwidth", 60, "display.width", 250):
        print(gate_table.to_string(index=False))


if __name__ == "__main__":
    main()
