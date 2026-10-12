"""Category-level frequency correlate, descriptive only (n = 5 categories, no test).

Design and expected directions were fixed before counting:
  detector per category = its vocabulary terms of both polarities minus the homograph-prone words of robustness check (c), with plurals
  (s/es, men, women, people/persons) and space / hyphen / closed-compound variants; a mention is counted once (longest term first).
  (A) coverage = units mentioning the category (any term, either polarity, once per unit) per 10,000 units; unit = HH pair (chosen
      transcript + rejected final turn), SHP post with all its comments, OASST message tree (English messages). Expected: more -> larger Shift.
  (B) preference signal, responses only: occurrences per million response words in rejected vs chosen responses, ratio rejected / chosen
      (HH final assistant turn; SHP preferred vs non-preferred comment, per comparison; OASST best- vs worst-ranked English assistant
      sibling reply, null ranks excluded). Derogatory = main, neutral = control. Expected: higher derogatory ratio -> larger Shift.
  Comparison with the SFT-family Shift_M(c) (h3_family_summary.csv): ranks + descriptive Spearman rho, no p; wording rho >= 0.7
  "in the expected direction", <= -0.7 "opposite to the expected direction", otherwise "no clear ordering".
  Robustness: detector with all terms (no homograph exclusion); African coverage without the single term "african".
Reads the dataset files offline from the HF cache (pinned revisions; no network). Only counts are written, never text.

Writes:
  results/category_frequency_counts.csv           raw counts with provenance (kept if it exists; --recount checks equality)
  reports/step_09_tables/s9b2_coverage.csv        (A) per variant x category x dataset (+ pooled)
  reports/step_09_tables/s9b2_preference.csv      (B) per variant x category x polarity x dataset (+ pooled)
  reports/step_09_tables/s9b2_term_hits.csv       per term: units mentioning it, response occurrences (which terms drive a category)
  reports/step_09_tables/s9b2_vs_shift.csv        per variant x category: Shift, coverage, ratios, ranks
  reports/step_09_tables/s9b2_rho.csv             descriptive rho + wording per variant x measure x dataset
  reports/step_09_tables/s9b2_gate_checks.csv
  results/figures/rq3_category_frequency (.png, .pdf)
Usage: .venv/bin/python scripts/09b2_category_frequency.py [--recount]
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from probing.config import load_config  # noqa: E402
from probing.frequency import (category_terms, count_words, detector, final_assistant_turn, hh_differs_before_final,  # noqa: E402
                               hh_response_pair, oasst_comparisons, shp_response_pair, variant_pattern)
from probing.pools import load_vocabulary  # noqa: E402

TAB = ROOT / "reports/step_09_tables"
S10C = {"escort", "tranny", "dyke", "bender", "poof", "nancy", "coon", "shim", "hoe"}   # homograph-prone words, excluded in robustness check (c)
DATASETS = ["hh", "shp", "oasst"]
VARIANTS = ["main", "all_terms", "af_no_african"]
AF = "African - Afro American"
POLS = ["SAFE", "NOT SAFE"]
REPOS = {  # pinned dataset revisions
    "hh": ("Anthropic/hh-rlhf", "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"),
    "shp": ("stanfordnlp/SHP", "e94b5f32602712d78ed494fe79105b1959396686"),
    "oasst": ("OpenAssistant/oasst1", "fdf72ae0827c1cda404aff25b6603abec9e3399b"),
}
EXPECTED_POOLS = {"African - Afro American": (3, 4), "Female Homosexual": (3, 1), "Homosexual Man": (5, 2),
                  "Sex Worker, Prostitute": (4, 4), "Transgender Woman": (3, 3)}
SHORT = {"African - Afro American": "African", "Female Homosexual": "Female Homosexual", "Homosexual Man": "Homosexual Man",
         "Sex Worker, Prostitute": "Sex Worker", "Transgender Woman": "Transgender Woman"}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a8a"


PATTERNS = {"hh": "*/train.jsonl.gz", "shp": "*/train.json", "oasst": "data/train-*.parquet"}
EXPECTED_FILES = {"hh": 4, "shp": 18, "oasst": 1}


def local_files(key: str) -> list[Path]:
    """Train files of the pinned revision, read offline from the HF cache snapshot.
    HH red-team is excluded (not preference pairs)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    repo, rev = REPOS[key]
    snap = Path(HF_HUB_CACHE) / f"datasets--{repo.replace('/', '--')}" / "snapshots" / rev
    files = sorted(f for f in snap.glob(PATTERNS[key]) if not f.parent.name.startswith("red-team"))
    if not files:
        raise FileNotFoundError(f"no cached files for {repo}@{rev}: download them once with huggingface_hub.snapshot_download("
                                f"repo_id='{repo}', repo_type='dataset', revision='{rev}', allow_patterns='{PATTERNS[key]}')")
    return files


def build_detectors(vocab: pd.DataFrame) -> tuple[dict, dict, re.Pattern, dict]:
    """(variant, category) -> {"any", "SAFE", "NOT SAFE"} patterns; term -> pattern; prefilter over all terms; term lists."""
    terms = {"main": category_terms(vocab, S10C), "all_terms": category_terms(vocab, set())}
    det: dict[tuple[str, str], dict[str, re.Pattern]] = {}
    for v in ["main", "all_terms"]:
        for cat, pools in terms[v].items():
            det[(v, cat)] = {"any": detector(pools["SAFE"] + pools["NOT SAFE"]), **{p: detector(pools[p]) for p in POLS}}
    af = [t for t in terms["main"][AF]["SAFE"] + terms["main"][AF]["NOT SAFE"] if t != "african"]
    det[("af_no_african", AF)] = {"any": detector(af)}
    all_terms = sorted({t for pools in terms["all_terms"].values() for ts in pools.values() for t in ts})
    term_pat = {t: re.compile(variant_pattern(t)) for t in all_terms}
    return det, term_pat, detector(all_terms), terms


class Counts:
    """Accumulates (A) unit hits, (B) response occurrences and words, per-term hits, and meta counts."""

    def __init__(self, det: dict, term_pat: dict, prefilter: re.Pattern) -> None:
        self.det, self.term_pat, self.pre = det, term_pat, prefilter
        self.units, self.cov, self.term_units = Counter(), Counter(), Counter()
        self.words, self.occ, self.term_occ, self.meta = Counter(), Counter(), Counter(), Counter()

    def unit(self, ds: str, texts: list[str]) -> dict:
        """Count one unit (conversation). Returns {(variant, cat): hit} for the independent OASST check."""
        self.units[ds] += 1
        lows = [t.lower() for t in texts if t]
        lows = [s for s in lows if self.pre.search(s)]
        hits = {k: any(d["any"].search(s) for s in lows) for k, d in self.det.items()} if lows else {k: False for k in self.det}
        for k, h in hits.items():
            self.cov[(ds,) + k] += h
        for t, p in self.term_pat.items():
            self.term_units[(ds, t)] += any(p.search(s) for s in lows)
        return hits

    def response_counts(self, text: str) -> tuple[int, dict, dict]:
        low = text.lower()
        if not self.pre.search(low):
            return count_words(text), {}, {}
        occ = {(k[0], k[1], p): len(d[p].findall(low)) for k, d in self.det.items() if k[0] != "af_no_african" for p in POLS}
        tocc = {t: len(p.findall(low)) for t, p in self.term_pat.items()}
        return count_words(text), occ, tocc

    def response(self, ds: str, role: str, counted: tuple[int, dict, dict]) -> None:
        w, occ, tocc = counted
        self.words[(ds, role)] += w
        for k, n in occ.items():
            self.occ[(ds,) + k + (role,)] += n
        for t, n in tocc.items():
            self.term_occ[(ds, t, role)] += n


def count_all(vocab: pd.DataFrame) -> pd.DataFrame:
    """Stream the three datasets once and return the long counts table."""
    det, term_pat, pre, _ = build_detectors(vocab)
    c = Counts(det, term_pat, pre)
    for name in local_files("hh"):
        with gzip.open(name, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ch, rj = hh_response_pair(r)
                c.meta[("hh", "pairs")] += 1
                c.meta[("hh", "differs_before_final")] += hh_differs_before_final(r)
                c.meta[("hh", "identical_final_turns")] += ch == rj
                c.unit("hh", [r["chosen"], final_assistant_turn(r["rejected"])])
                c.meta[("hh", "comparisons_usable")] += 1
                c.response("hh", "chosen", c.response_counts(ch))
                c.response("hh", "rejected", c.response_counts(rj))
        print(f"  hh {name.parent.name}: {c.meta[('hh', 'pairs')]} pairs")
    cache: dict[str, tuple] = {}
    for name in local_files("shp"):
        posts: dict[str, tuple[str, dict]] = {}
        with open(name, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                c.meta[("shp", "pairs")] += 1
                h, com = posts.setdefault(r["post_id"], (r["history"], {}))
                com[r["c_root_id_A"]], com[r["c_root_id_B"]] = r["human_ref_A"], r["human_ref_B"]
                if r["labels"] not in (0, 1) or not r["human_ref_A"].strip() or not r["human_ref_B"].strip():
                    continue
                c.meta[("shp", "comparisons_usable")] += 1
                ids = (r["c_root_id_A"], r["c_root_id_B"]) if int(r["labels"]) == 1 else (r["c_root_id_B"], r["c_root_id_A"])
                for role, cid, txt in zip(["chosen", "rejected"], ids, shp_response_pair(r)):
                    if cid not in cache:
                        cache[cid] = c.response_counts(txt)
                    c.response("shp", role, cache[cid])
        for h, com in posts.values():
            c.unit("shp", [h] + list(com.values()))
            c.meta[("shp", "comments_in_units")] += len(com)
        print(f"  shp {name.parent.name}: {c.meta[('shp', 'pairs')]} pairs, {c.units['shp']} posts")
    d = pd.read_parquet(local_files("oasst")[0])
    en = d[d["lang"] == "en"]
    mismatch = 0
    for _, g in en.groupby("message_tree_id", sort=True):
        hits = c.unit("oasst", g["text"].tolist())
        joined = " | ".join(g["text"]).lower()           # independent check: hit on the concatenated tree text
        mismatch += sum(hits[k] != bool(dd["any"].search(joined)) for k, dd in det.items())
    c.meta[("oasst", "concat_check_mismatches")] = mismatch
    ea = en[en["role"] == "assistant"]
    c.meta[("oasst", "rows")] = len(d)
    c.meta[("oasst", "en_assistant_replies")] = len(ea)
    c.meta[("oasst", "en_assistant_null_rank_dropped")] = int(ea["rank"].isna().sum())
    comp = oasst_comparisons(d)
    c.meta[("oasst", "comparisons_usable")] = len(comp)
    c.meta[("oasst", "used_replies_deleted")] = int(comp["chosen_deleted"].sum() + comp["rejected_deleted"].sum())
    for ch, rj in zip(comp["chosen"], comp["rejected"]):
        c.response("oasst", "chosen", c.response_counts(ch))
        c.response("oasst", "rejected", c.response_counts(rj))
    rows = []
    rev = {k: v[1] for k, v in REPOS.items()}
    for ds, n in c.units.items():
        rows.append(dict(kind="units", dataset=ds, value=n))
    for (ds, v, cat), n in c.cov.items():
        rows.append(dict(kind="coverage_hits", dataset=ds, variant=v, category=cat, value=n))
    for (ds, role), n in c.words.items():
        rows.append(dict(kind="response_words", dataset=ds, role=role, value=n))
    for ds in DATASETS:
        for k in det:
            if k[0] == "af_no_african":
                continue
            for p in POLS:
                for role in ["chosen", "rejected"]:
                    rows.append(dict(kind="response_occurrences", dataset=ds, variant=k[0], category=k[1], polarity=p, role=role,
                                     value=c.occ[(ds, k[0], k[1], p, role)]))
        for t in term_pat:
            rows.append(dict(kind="term_units", dataset=ds, term=t, value=c.term_units[(ds, t)]))
            for role in ["chosen", "rejected"]:
                rows.append(dict(kind="term_occurrences", dataset=ds, term=t, role=role, value=c.term_occ[(ds, t, role)]))
    for (ds, name), n in c.meta.items():
        rows.append(dict(kind="meta", dataset=ds, term=name, value=n))
    out = pd.DataFrame(rows, columns=["kind", "dataset", "revision", "variant", "category", "polarity", "role", "term", "value"])
    out["revision"] = out["dataset"].map(rev)
    return out.sort_values(["kind", "dataset", "variant", "category", "polarity", "role", "term"], na_position="first").reset_index(drop=True)


def load_or_count(path: Path, vocab: pd.DataFrame, recount: bool) -> pd.DataFrame:
    """Computed once; an existing file is never overwritten (with --recount it must be identical)."""
    if path.exists() and not recount:
        print(f"{path.name} exists: loaded (use --recount to verify it)")
        return pd.read_csv(path)
    counts = count_all(vocab)
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(io.StringIO(counts.to_csv(index=False))))
        print(f"{path.name} already exists and equals the recount: kept")
    else:
        counts.to_csv(path, index=False)
    return pd.read_csv(path)


def _get(counts: pd.DataFrame, kind: str, **kw) -> pd.DataFrame:
    sel = counts["kind"] == kind
    for k, v in kw.items():
        sel &= counts[k] == v
    return counts[sel]


def coverage_table(counts: pd.DataFrame) -> pd.DataFrame:
    units = _get(counts, "units").set_index("dataset")["value"]
    hits = _get(counts, "coverage_hits")
    rows = []
    for (v, cat), g in hits.groupby(["variant", "category"], sort=False):
        h = g.set_index("dataset")["value"]
        for ds in DATASETS + ["pooled"]:
            n = units[DATASETS].sum() if ds == "pooled" else units[ds]
            k = h[DATASETS].sum() if ds == "pooled" else h[ds]
            rows.append(dict(variant=v, category=cat, dataset=ds, n_units=int(n), units_mentioning=int(k), per_10k=k / n * 1e4))
    out = pd.DataFrame(rows)
    out["variant"] = pd.Categorical(out["variant"], VARIANTS)
    return out.sort_values(["variant", "category"], kind="stable").reset_index(drop=True)


def preference_table(counts: pd.DataFrame) -> pd.DataFrame:
    words = _get(counts, "response_words").set_index(["dataset", "role"])["value"]
    occ = _get(counts, "response_occurrences")
    rows = []
    for (v, cat, pol), g in occ.groupby(["variant", "category", "polarity"], sort=False):
        o = g.set_index(["dataset", "role"])["value"]
        for ds in DATASETS + ["pooled"]:
            dss = DATASETS if ds == "pooled" else [ds]
            r = {role: (sum(o[(d, role)] for d in dss), sum(words[(d, role)] for d in dss)) for role in ["chosen", "rejected"]}
            pm = {role: n / w * 1e6 for role, (n, w) in r.items()}
            rows.append(dict(variant=v, category=cat, polarity=pol, dataset=ds,
                             chosen_count=int(r["chosen"][0]), chosen_words=int(r["chosen"][1]), chosen_per_million=pm["chosen"],
                             rejected_count=int(r["rejected"][0]), rejected_words=int(r["rejected"][1]),
                             rejected_per_million=pm["rejected"],
                             ratio_rejected_chosen=pm["rejected"] / pm["chosen"] if r["chosen"][0] > 0 else np.nan))
    return pd.DataFrame(rows)


def term_table(counts: pd.DataFrame, vocab: pd.DataFrame) -> pd.DataFrame:
    units = _get(counts, "units").set_index("dataset")["value"]
    tu = _get(counts, "term_units").pivot(index="term", columns="dataset", values="value")
    to = _get(counts, "term_occurrences").groupby(["term", "role"])["value"].sum().unstack()
    v = vocab.assign(term=vocab["word"].str.strip().str.lower())
    used = v.assign(cp=v["category"] + " / " + v["polarity"]).groupby("term")["cp"].agg("; ".join)
    out = pd.DataFrame(index=tu.index)
    out["used_in"] = used.reindex(out.index)
    out["excluded_in_main"] = out.index.isin(S10C)
    for ds in DATASETS:
        out[f"units_{ds}"] = tu[ds]
    out["units_pooled"] = tu[DATASETS].sum(axis=1)
    out["per_10k_pooled"] = out["units_pooled"] / units[DATASETS].sum() * 1e4
    out["response_occ_chosen_pooled"] = to["chosen"]
    out["response_occ_rejected_pooled"] = to["rejected"]
    return out.reset_index()


def wording(rho: float) -> str:
    """Fixed before counting."""
    if np.isnan(rho):
        return "undefined"
    return "in the expected direction" if rho >= 0.7 else "opposite to the expected direction" if rho <= -0.7 else "no clear ordering"


def rank_desc(x: pd.Series) -> pd.Series:
    return x.rank(ascending=False, method="average")


def vs_shift(cov: pd.DataFrame, pref: pd.DataFrame, fam: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per variant x category: Shift, coverage, ratios and ranks; descriptive rho with the fixed wording per variant x measure x dataset."""
    shift = fam.set_index("category")["family_mean"]
    rows, rho_rows = [], []
    for v in VARIANTS:
        pv = "main" if v == "af_no_african" else v           # the African-coverage variant changes (A) only
        for ds in ["pooled"] + DATASETS:
            c = cov[(cov["variant"] == "main") & (cov["dataset"] == ds)].set_index("category")["per_10k"]
            if v != "main":
                cv = cov[(cov["variant"] == v) & (cov["dataset"] == ds)].set_index("category")["per_10k"]
                c = cv.combine_first(c) if v == "af_no_african" else cv
            r = {pol: pref[(pref["variant"] == pv) & (pref["dataset"] == ds) & (pref["polarity"] == pol)]
                 .set_index("category")["ratio_rejected_chosen"] for pol in POLS}
            t = pd.DataFrame({"shift": shift, "coverage_per_10k": c, "derog_ratio": r["NOT SAFE"], "neutral_ratio": r["SAFE"]})
            t.index.name = "category"
            if ds == "pooled":
                tt = t.assign(shift_rank=rank_desc(t["shift"]), coverage_rank=rank_desc(t["coverage_per_10k"]),
                              derog_ratio_rank=rank_desc(t["derog_ratio"]), neutral_ratio_rank=rank_desc(t["neutral_ratio"]))
                rows.append(tt.reset_index().assign(variant=v))
            for meas, col in [("A coverage", "coverage_per_10k"), ("B derogatory ratio", "derog_ratio"),
                              ("B neutral ratio (control)", "neutral_ratio")]:
                ok = t[col].notna()
                rho = stats.spearmanr(t.loc[ok, "shift"], t.loc[ok, col]).statistic if ok.sum() >= 3 else np.nan
                rho_rows.append(dict(variant=v, measure=meas, dataset=ds, n_categories=int(ok.sum()), rho=rho, wording=wording(rho)))
    vs = pd.concat(rows, ignore_index=True)
    return vs[["variant"] + [c for c in vs.columns if c != "variant"]], pd.DataFrame(rho_rows)


def gate_checks(counts: pd.DataFrame, terms: dict, fam: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    meta = _get(counts, "meta").set_index(["dataset", "term"])["value"]
    units = _get(counts, "units").set_index("dataset")["value"]
    rows = []

    def add(check: str, value, expected, ok: bool, blocking: bool = True) -> None:
        rows.append(dict(check=check, value=value, expected=expected, pass_=bool(ok), blocking=blocking))

    pairs = int(meta[("hh", "pairs")])
    dif = int(meta[("hh", "differs_before_final")])
    add("1a. HH pairs that differ before the final turn (info)", f"{dif} ({dif / pairs:.2%})", "reported", True, False)
    add("1b. HH pairs with identical final turns (info)", int(meta[("hh", "identical_final_turns")]), "reported", True, False)
    add("2a. SHP usable comparisons (label 0/1, both comments non-empty) / pairs", f"{int(meta[('shp', 'comparisons_usable')])} / "
        f"{int(meta[('shp', 'pairs')])}", "<= 348,718", int(meta[("shp", "pairs")]) == 348_718)
    add("2b. OASST usable comparisons (parents with >= 2 ranked English assistant replies)", int(meta[("oasst", "comparisons_usable")]),
        "> 0", meta[("oasst", "comparisons_usable")] > 0)
    add("2c. OASST English assistant replies / dropped for null rank / deleted among used (info)",
        f"{int(meta[('oasst', 'en_assistant_replies')])} / {int(meta[('oasst', 'en_assistant_null_rank_dropped')])} / "
        f"{int(meta[('oasst', 'used_replies_deleted')])}", "reported", True, False)
    nf = {k: len(local_files(k)) for k in DATASETS}
    add("0. cached train files at the pinned revisions (offline)", nf, EXPECTED_FILES, nf == EXPECTED_FILES)
    add("3a. units: HH pairs", int(units["hh"]), 160_800, units["hh"] == 160_800)
    add("3b. units: SHP posts (= unique posts of the first version)", int(units["shp"]), 38_993, units["shp"] == 38_993)
    add("3c. units: OASST English message trees (info)", int(units["oasst"]), "reported", True, False)
    fm = fam[[f"shift_{m}" for m in ["sft", "sft_dpo", "sft_ppo", "sft_kto"]]].mean(axis=1)
    add("4a. Shift = h3_family_summary family_mean (recomputed from the four columns)", float((fm - fam["family_mean"]).abs().max()),
        "<= 1e-12", float((fm - fam["family_mean"]).abs().max()) <= 1e-12 and len(fam) == 5)
    pools = {cat: (len(p["SAFE"]), len(p["NOT SAFE"])) for cat, p in terms["main"].items()}
    add("4b. pools after the S10 (c) exclusion (neutral, derogatory)", pools, EXPECTED_POOLS, pools == EXPECTED_POOLS)
    add("5. OASST coverage: any-message hit == hit on concatenated tree text (mismatches)", int(meta[("oasst", "concat_check_mismatches")]),
        0, meta[("oasst", "concat_check_mismatches")] == 0)
    p = cov[cov["dataset"] == "pooled"].pivot(index="category", columns="variant", values="units_mentioning")
    add("6. nesting: main <= all_terms; African without 'african' <= main", "checked", "true",
        bool((p["main"] <= p["all_terms"]).all() and p.loc[AF, "af_no_african"] <= p.loc[AF, "main"]))
    out = pd.DataFrame(rows).rename(columns={"pass_": "pass"})
    assert out.loc[out["blocking"], "pass"].all(), out
    return out


def figure(vs: pd.DataFrame, rho: pd.DataFrame, pref: pd.DataFrame, path: Path) -> None:
    """Two panels: (A) coverage vs Shift, (B) derogatory ratio vs Shift; main filled, all-terms variant hollow; 5 labelled points
    (panel B labels carry the main chosen / rejected derogatory counts)."""
    from matplotlib.ticker import ScalarFormatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    for ax, col, meas, xlab in [(axes[0], "coverage_per_10k", "A coverage", "(A) conversations mentioning the category per 10,000 (pooled; log)"),
                                (axes[1], "derog_ratio", "B derogatory ratio", "(B) derogatory terms: rate in rejected / rate in chosen responses (pooled)")]:
        for v, style in [("main", dict(color=BLUE, s=55)), ("all_terms", dict(facecolor="none", edgecolor=ORANGE, s=70, lw=1.4))]:
            t = vs[vs["variant"] == v]
            r = rho[(rho["variant"] == v) & (rho["measure"] == meas) & (rho["dataset"] == "pooled")]["rho"].iloc[0]
            lab = "main (homograph-prone words excluded)" if v == "main" else "variant: all terms"
            ax.scatter(t[col], t["shift"], zorder=3, label=f"{lab}, rho = {r:.1f}", **style)
        cnt = pref[(pref["variant"] == "main") & (pref["dataset"] == "pooled") & (pref["polarity"] == "NOT SAFE")].set_index("category")
        for _, row in vs[vs["variant"] == "main"].iterrows():
            lab = SHORT[row["category"]]
            if col == "derog_ratio":
                k = cnt.loc[row["category"]]
                lab += f" ({int(k['chosen_count'])} / {int(k['rejected_count'])})"
            ax.annotate(lab, (row[col], row["shift"]), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
        if col == "coverage_per_10k":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(ScalarFormatter())
            ax.tick_params(axis="x", which="minor", labelsize=7)
        else:
            ax.axvline(1, color=MUTED, lw=1)
        ax.set_xlabel(xlab, fontsize=8.5)
        ax.set_ylabel("SFT-family Shift_M(c) (nat)")
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.grid(True, color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Topic frequency (descriptive, n = 5): feedback-data coverage and preference signal vs SFT-family Shift", fontsize=10, color=INK)
    note = ("Expected (fixed before counting): more coverage -> larger Shift; higher derogatory ratio -> larger Shift. rho is descriptive "
            "(n = 5, no p): >= 0.7 expected direction, <= -0.7 opposite, otherwise no clear ordering. Panel B labels: derogatory "
            "occurrences in chosen / rejected responses (main detector).")
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(note, 160)), fontsize=8, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recount", action="store_true", help="recount from the datasets and check the counts file is identical")
    args = ap.parse_args()
    cfg = load_config()
    res = ROOT / cfg["paths"]["results"]
    (res / "figures").mkdir(exist_ok=True)
    vocab = load_vocabulary(ROOT / cfg["paths"]["vocabulary"])
    fam = pd.read_csv(TAB / "h3_family_summary.csv")
    _, _, _, terms = build_detectors(vocab)

    counts = load_or_count(res / "category_frequency_counts.csv", vocab, args.recount)
    cov = coverage_table(counts)
    pref = preference_table(counts)
    vs, rho = vs_shift(cov, pref, fam)
    gates = gate_checks(counts, terms, fam, cov)
    for name, t in [("s9b2_coverage", cov), ("s9b2_preference", pref), ("s9b2_term_hits", term_table(counts, vocab)),
                    ("s9b2_vs_shift", vs), ("s9b2_rho", rho), ("s9b2_gate_checks", gates)]:
        t.to_csv(TAB / f"{name}.csv", index=False)
    figure(vs, rho, pref, res / "figures/rq3_category_frequency")

    print(f"checks: {int(gates['pass'].sum())}/{len(gates)} pass")
    print(vs.round(3).to_string(index=False))
    print(rho[rho["dataset"] == "pooled"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
