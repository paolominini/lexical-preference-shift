#!/usr/bin/env python3
"""
build_prefix_dataset.py
=======================
Natural-prefix extraction (tiers T1/T2/T3): the text before a singular occurrence
of a vocabulary word, cut right before it.

The source corpora are not redistributed with this repository: download them from
their original sources (under their own licenses) into --data-dir.

Pipeline (each step is a small function, easy to swap or extend):
  load corpora -> clean text -> find vocabulary hits (singular only)
  -> classify tier from the left context -> build prefix ending in "the"
  -> filters (min context, priming) -> dedup -> export

Usage
  python code/prefix_extraction/build_prefix_dataset.py --data-dir DIR --vocab data/Vocabulary_Final.xlsx --out prefix_dataset.csv
  (--exclusions defaults to data/manual_exclusions.csv of this repository)

Expected files in --data-dir:
  hatexplain.json  ethos.csv (';'-separated)  dynahate.csv  measuring_hate_speech.csv (raw, annotator-level)

To add a corpus: write a loader returning a DataFrame[doc_id, text, (extra cols)]
and register it in CORPORA.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

try:
    import ftfy
except ImportError:  # pragma: no cover
    ftfy = None

# ----------------------------------------------------------------------------
# Extraction rules, all in one place so that they can be read and checked
# ----------------------------------------------------------------------------
DET_SWAP = {"a", "an", "this", "that", "another", "one", "your", "my", "his", "her",
            "their", "our", "its", "ur"}
NOT_MOD = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them", "is", "are",
           "was", "were", "be", "been", "am", "do", "does", "did", "have", "has", "had", "to",
           "of", "in", "on", "at", "for", "with", "from", "by", "and", "or", "but", "if", "so",
           "as", "than", "not", "no", "the", "a", "an", "this", "that", "these", "those", "what",
           "who", "which", "when", "where", "how", "why", "just", "like", "call", "called"}
TOK = re.compile(r"[A-Za-z0-9_'’-]+|[^\sA-Za-z0-9_'’-]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
PLACEHOLDER = re.compile(r"<[a-z_]+>")
PLURAL_SLOT = r"\b(?:all|both|half|most|many|some|one|any|each|none|few|several) of the$|\b(?:all|both|half) the$"
PREDICATIVE = {"is", "am", "are", "was", "were", "be", "been", "being", "it's", "he's", "she's", "you're",
               "i'm", "that's", "its", "as", "like", "become", "became", "becomes", "call", "called",
               "calling", "calls", "such", "ur", "im", "youre", "hes", "shes", "u", "into"}
ADVERBS = {"not", "also", "just", "really", "only", "still", "even", "truly", "never", "always", "definitely"}
OBJ_PRONOUNS = {"me", "you", "him", "her", "us", "them", "it", "somebody", "someone", "yourself", "myself"}
CALL_VERBS = {"call", "called", "calling", "calls", "make", "makes", "made", "consider", "considered", "label", "labelled", "labeled"}          # HateXplain: <user> <number> <censored> ...
HANDLE = re.compile(r"@\w+")
URL = re.compile(r"https?://\S+|www\.\S+")


@dataclass
class Config:
    min_ctx: int = 5
    ctx_mode: str = "words"        # "words": whitespace tokens containing a letter | "alpha": legacy [A-Za-z]+ fragments
    priming_scope: str = "prefix"  # "prefix": vocab word inside the prefix | "text": anywhere to the left (legacy)
    clean: bool = True
    context: str = "sentence"      # "sentence": last sentence fragment | "full": everything before the determiner
    strip_placeholders: bool = True  # remove HateXplain tokens (<user>, <number>, ...) from the text
    exclude_sources: tuple = ("queer", "homosexual")  # still probed in W pools, never used to MINE prefixes
    max_ctx: int = 40              # longer contexts are dropped
    extend_below: int = 10         # if the last sentence has fewer words, prepend the previous one (if total <= max_ctx)
    edit_mode: str = "minimal"     # "minimal": edit only a/an directly before the word | "the": always end in "the" (old)


# ----------------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------------
_CTRL = {0x18: "'", 0x19: "'", 0x1C: '"', 0x1D: '"', 0x0D: " "}
_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\u2060\u2063\u2066\u2067\u2068\u2069"
                             "\u202a\u202b\u202c\u202d\u202e\ufeff"), None)


def clean_text(t: str) -> str:
    t = str(t).translate(_CTRL)
    if ftfy is not None:
        t = ftfy.fix_text(t)            # repairs mojibake (ETHOS), entities, odd controls
    t = html.unescape(t).translate(_ZW)
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------------------
# Corpus loaders  -> DataFrame[doc_id, text, extras...]
# ----------------------------------------------------------------------------
def load_hatexplain(p: Path) -> pd.DataFrame:
    d = json.load(open(p, encoding="utf-8"))
    return pd.DataFrame({"doc_id": list(d.keys()),
                         "text": [" ".join(v["post_tokens"]) for v in d.values()]})


def load_ethos(p: Path) -> pd.DataFrame:
    e = pd.read_csv(p, sep=";")
    return pd.DataFrame({"doc_id": [f"ethos_{i}" for i in range(len(e))], "text": e["comment"].astype(str)})


def load_dynahate(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p)
    return pd.DataFrame({"doc_id": [f"dyna_{i}" for i in range(len(d))], "text": d["text"].astype(str)})


def load_mhs(p: Path) -> pd.DataFrame:
    cols = ["comment_id", "text", "target_race_black", "target_sexuality_gay", "target_sexuality_lesbian",
            "target_gender_transgender_women", "target_gender_transgender_men",
            "target_gender_transgender_unspecified"]
    m = pd.read_csv(p, usecols=cols)
    g = m.groupby("comment_id", sort=False).agg(
        text=("text", "first"),
        mhs_t_black=("target_race_black", "mean"),
        mhs_t_gay=("target_sexuality_gay", "mean"),
        mhs_t_lesbian=("target_sexuality_lesbian", "mean"),
        mhs_t_transwoman=("target_gender_transgender_women", "mean"),
        mhs_t_transmen=("target_gender_transgender_men", "mean"),
        mhs_t_transunsp=("target_gender_transgender_unspecified", "mean"),
    ).reset_index().rename(columns={"comment_id": "doc_id"})
    g["doc_id"] = g["doc_id"].astype(str)
    g["mhs_t_trans_any"] = g[["mhs_t_transwoman", "mhs_t_transmen", "mhs_t_transunsp"]].max(axis=1)
    return g


def load_civil(p: Path) -> pd.DataFrame:
    c = pd.read_csv(p)
    return pd.DataFrame({"doc_id": [f"civil_{i}" for i in range(len(c))], "text": c["text"].astype(str)})


def load_sbic(p: Path) -> pd.DataFrame:
    """p = folder holding SBIC_v2_agg_{trn,dev,tst}.csv (one row per post)."""
    parts = [pd.read_csv(Path(p).parent / f"SBIC_v2_agg_{s}.csv").assign(split=s) for s in ("trn", "dev", "tst")]
    d = pd.concat(parts, ignore_index=True).drop_duplicates("post")
    return pd.DataFrame({"doc_id": [f"sbic_{i}" for i in range(len(d))], "text": d["post"].astype(str),
                         "sbic_target": d["targetMinority"].astype(str), "sbic_source": d["dataSource"].astype(str)})


CORPORA = {  # name -> (loader, default filename)
    "hatexplain": (load_hatexplain, "hatexplain.json"),
    "ethos": (load_ethos, "ethos.csv"),
    "dynahate": (load_dynahate, "dynahate.csv"),
    "mhs": (load_mhs, "measuring_hate_speech.csv"),
    "civil": (load_civil, "civil_comments_filtered.csv"),
    "sbic": (load_sbic, "SBIC_v2_agg_trn.csv"),
}


def load_corpora(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {n: fn(Path(data_dir) / f) for n, (fn, f) in CORPORA.items()}


# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------
def norm(w: str) -> str:
    return re.sub(r"[\s-]+", " ", str(w).lower().strip())


def build_vocab(xlsx: str):
    v = pd.read_excel(xlsx, sheet_name="Final_Probing_Set")
    v["WORD"] = v["WORD"].map(norm)
    info = {}
    for w, g in v.groupby("WORD"):
        cats = tuple(sorted(set(g["CATEGORY"])))
        pols = set(g["POLARITY"])
        assert len(pols) == 1, f"{w}: conflicting polarity {pols}"
        info[w] = dict(cats=cats, cat=cats[0] if len(cats) == 1 else "SHARED", pol=pols.pop())

    def pat(w):
        return re.escape(w).replace(r"\ ", r"[\s-]+").replace(r"\-", r"[\s-]?")

    alt = "|".join(pat(w) for w in sorted(info, key=len, reverse=True))   # longest match wins
    L, R = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"
    hit_re = re.compile(L + "(" + alt + ")" + R, re.I)                     # singular only
    prime_re = re.compile(L + "(?:" + alt + ")(?:s|es)?" + R, re.I)        # singular or plural
    plural_re = re.compile(L + "(?:" + alt + ")(?:s|es)" + R, re.I)
    return info, hit_re, prime_re, plural_re


# ----------------------------------------------------------------------------
# Tier classification
# ----------------------------------------------------------------------------
def classify(left: str):
    """Return (tier, det_start_char, dropped_modifiers) for the text left of a hit."""
    toks = [(m.group(), m.start()) for m in TOK.finditer(left)]
    if not toks:
        return "X", None, ""
    last = toks[-1][0].lower()
    if last == "the":
        return "T1", toks[-1][1], ""
    if last in DET_SWAP:
        return "T2", toks[-1][1], ""
    for k in (1, 2, 3):
        if len(toks) <= k:
            break
        anchor = toks[-1 - k][0].lower()
        mods = [t[0].lower() for t in toks[len(toks) - k:]]
        if anchor in DET_SWAP or anchor == "the":
            if all(m.isalpha() and m not in NOT_MOD for m in mods):
                return "T3", toks[-1 - k][1], " ".join(mods)
            return "X", None, ""
        first_mod = toks[-k][0]
        if not first_mod.isalpha() or first_mod.lower() in NOT_MOD:
            break
    return "X", None, ""


# ----------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------
def _n_ctx(sent: str, mode: str) -> int:
    if mode == "alpha":
        return len(re.findall(r"[A-Za-z]+", sent))
    return sum(bool(re.search(r"[A-Za-z]", w)) for w in sent.split())


def extract(df: pd.DataFrame, corpus: str, info, hit_re, prime_re, plural_re, cfg: Config) -> pd.DataFrame:
    extra_cols = [c for c in df.columns if c.startswith(("mhs_", "sbic_"))]
    rows = []
    for rec in df.to_dict("records"):
        text = clean_text(rec["text"]) if cfg.clean else str(rec["text"])
        if cfg.strip_placeholders:
            text = re.sub(r"\s+", " ", PLACEHOLDER.sub(" ", text)).strip()
        for m in hit_re.finditer(text):
            w = norm(m.group(1))
            if w in cfg.exclude_sources:
                continue
            left = text[:m.start()]
            tier, det_i, mods = classify(left)
            r = dict(corpus=corpus, doc_id=rec["doc_id"], source_word=w, category=info[w]["cat"],
                     shared_cats="|".join(info[w]["cats"]) if info[w]["cat"] == "SHARED" else "",
                     polarity=info[w]["pol"], tier=tier, orig_tail="", dropped_mods=mods,
                     prefix="", n_ctx=0, ctx_ok=False, primed=False, next_word="", ctx_truncated=False,
                     edited=False, right_ctx="", ctx_sent="")
            right = text[m.end():]
            nw = re.match(r"\s*([A-Za-z][A-Za-z']*)", right)
            r["next_word"] = nw.group(1).lower() if nw else ""
            for c in extra_cols:
                r[c] = rec.get(c)
            if tier != "X":
                ctx = left[:det_i]
                if not ctx.strip():
                    sent = ""
                elif cfg.context == "sentence":
                    pieces = [p for p in SENT_SPLIT.split(ctx) if p.strip()]
                    sent = pieces[-1] if pieces else ""
                    if len(sent.split()) < cfg.extend_below and len(pieces) > 1:
                        two = (pieces[-2].strip() + " " + sent.strip())
                        if len(two.split()) <= cfg.max_ctx:
                            sent = two
                else:
                    sent = ctx
                sent = sent.strip()
                r["ctx_truncated"] = bool(ctx.strip()) and sent != ctx.strip()   # earlier sentences were dropped
                r["orig_tail"] = left[det_i:].strip()
                det = r["orig_tail"].split()[0].lower()
                if cfg.edit_mode == "the" or tier == "T1":
                    r["prefix"] = (sent + " the").strip()
                    r["edited"] = tier != "T1"
                elif tier == "T2" and det in {"a", "an"}:
                    # the only case that must be edited: a/an right before the slot would penalise vowel/consonant-initial words
                    r["prefix"] = (sent + " the").strip()
                    r["edited"] = True
                else:
                    # T2 with this/that/your/his... or T3 (det + modifiers): original text, no edit
                    r["prefix"] = (sent + " " + r["orig_tail"]).strip()
                    r["edited"] = False
                r["right_ctx"] = right[:120]
                r["ctx_sent"] = sent
                r["n_ctx"] = _n_ctx(sent, cfg.ctx_mode)
                r["ctx_ok"] = cfg.min_ctx <= r["n_ctx"] <= cfg.max_ctx
                scope = left if cfg.priming_scope == "text" else r["prefix"]
                r["primed"] = bool(prime_re.search(scope))
            rows.append(r)
    return pd.DataFrame(rows)


def extract_all(corpora, info, hit_re, prime_re, plural_re, cfg: Config) -> pd.DataFrame:
    parts = [extract(df, n, info, hit_re, prime_re, plural_re, cfg) for n, df in corpora.items()]
    out = pd.concat(parts, ignore_index=True)
    p = out.prefix.str.lower()
    out["plural_slot"] = p.str.contains(PLURAL_SLOT)
    w = out.ctx_sent.str.lower().str.replace(r"[^\w'’ ]", " ", regex=True).str.split()
    prev = w.str[-1].fillna("")
    prev2 = w.str[-2].fillna("")
    pred = (prev.isin(PREDICATIVE) | prev.str.endswith(("'s", "'re", "'m", "’s"))
            | (prev.isin(ADVERBS) & prev2.isin(PREDICATIVE))          # "is not a", "is also a"
            | (prev.isin(OBJ_PRONOUNS) & prev2.isin(CALL_VERBS)))     # "called me a", "make him a"
    out["predicative_edit"] = out.edited & pred
    out["usable"] = (out.tier != "X") & out.ctx_ok & ~out.primed & ~out.plural_slot & ~out.predicative_edit
    out["flag_placeholder"] = out.prefix.str.contains(PLACEHOLDER)
    out["flag_handle"] = out.prefix.str.contains(HANDLE)
    out["flag_url"] = out.prefix.str.contains(URL)
    out["flag_lower_nopunct"] = out.prefix.apply(lambda s: s == s.lower() and not re.search(r"[.!?,;:]", s))
    return out


# ----------------------------------------------------------------------------
# Dedup
# ----------------------------------------------------------------------------
def near_key(s: str) -> str:
    s = s.lower()
    s = re.sub(r"https?://\S+|www\.\S+|@\w+|<[^>]+>|&\w+;|\brt\b", " ", s)
    s = re.sub(r"[^a-z ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def dedup_key(df: pd.DataFrame, scope: str) -> pd.Series:
    p = df.prefix.str.lower()
    if scope == "group":      # legacy: unique within corpus x category x polarity
        return df.corpus + "|" + df.category + "|" + df.polarity + "|" + p
    if scope == "category":   # exact text, pooled across corpora and polarity, per category
        return df.category + "|" + p
    if scope == "global":     # exact text, ignoring category
        return p
    if scope == "near":       # category + normalised text (handles, urls, punctuation, digits stripped)
        return df.category + "|" + df.prefix.map(near_key)
    raise ValueError(scope)


def apply_dedup(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "none":
        return df
    return df.loc[~dedup_key(df, scope).duplicated(keep="first")]


def n_unique(df: pd.DataFrame, tiers, scope: str) -> pd.Series:
    """Unique prefixes per category among usable rows of the given tiers."""
    u = df[df.usable & df.tier.isin(tiers)]
    return u.assign(_k=dedup_key(u, scope)).groupby("category")["_k"].nunique()


# default list of hand-excluded hits (non-person senses), kept with the other data files
EXCLUSIONS = Path(__file__).resolve().parents[2] / "data" / "manual_exclusions.csv"


def apply_manual_exclusions(df: pd.DataFrame, path) -> pd.DataFrame:
    """Drop hits listed in a CSV (corpus, doc_id, source_word, reason) - e.g. non-person senses."""
    if path is None or not Path(path).exists():
        return df
    ex = pd.read_csv(path, dtype=str)
    key = df.corpus + "|" + df.doc_id.astype(str) + "|" + df.source_word
    bad = set(ex.corpus + "|" + ex.doc_id + "|" + ex.source_word)
    return df[~key.isin(bad)]


def build_final(data_dir, vocab, cfg: Config | None = None, exclusions="manual_exclusions.csv",
                dedup="category"):
    """Whole pipeline in one call. Returns (all candidate hits, final dataset)."""
    cfg = cfg or Config()
    info, hit_re, prime_re, plural_re = build_vocab(vocab)
    cand = extract_all(load_corpora(Path(data_dir)), info, hit_re, prime_re, plural_re, cfg)
    final = apply_dedup(apply_manual_exclusions(cand[cand.usable], exclusions), dedup).copy()
    final.insert(0, "prefix_id", [f"p{i:05d}" for i in range(len(final))])
    return cand, final


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", default="prefix_dataset.csv")
    ap.add_argument("--exclusions", default=str(EXCLUSIONS))
    ap.add_argument("--dedup", default="category", choices=["none", "group", "category", "global", "near"])
    ap.add_argument("--min-ctx", type=int, default=5)
    a = ap.parse_args()
    cand, final = build_final(a.data_dir, a.vocab, Config(min_ctx=a.min_ctx), a.exclusions, a.dedup)
    final.to_csv(a.out, index=False)
    print(f"hits={len(cand)}  usable={int(cand.usable.sum())}  final={len(final)}  -> {a.out}")
    print(final.groupby(["category", "tier"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
