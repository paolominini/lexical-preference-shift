"""Robustness checks: prefix subsets, leave-one-word-out scores and the survival rule. Pure functions, no I/O.

Subsets return prefix ids; `drop_word` returns a scores table without one (category, word) pair; `survival_label` and
`aggregate_labels` implement the survival rule, fixed before the checks were run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ADDED_CORPORA = ("civil", "sbic")   # added at selection because the four hate-speech corpora were not enough (Prefix_Selection_Log)
HOLDS, WEAKENED, WEAKENED_PREC, REVERSED, NOT_INTERP = "holds", "weakened", "weakened (precision scale)", "reversed", "not interpretable"
SAME, CHANGES = "same", "changes"
SEVERITY = {HOLDS: 0, SAME: 0, "holds (by construction)": 0, NOT_INTERP: 0, WEAKENED: 1, WEAKENED_PREC: 1, CHANGES: 2, REVERSED: 3}


def exclude_edited(prefixes: pd.DataFrame) -> list[str]:
    """Check (b): ids of the prefixes whose article was NOT edited (a/an -> the)."""
    return prefixes.loc[~prefixes["edited"].astype(bool), "final_id"].tolist()


def corpus_group(prefixes: pd.DataFrame) -> pd.Series:
    """Check (a): 'added' (Civil Comments, SBIC) or 'original' (HateXplain, DynaHate, MHS, ETHOS), indexed by prefix id."""
    g = np.where(prefixes["corpus"].isin(ADDED_CORPORA), "added", "original")
    return pd.Series(g, index=prefixes["final_id"].to_numpy(), name="corpus_group")


def stratified_halves(prefixes: pd.DataFrame, seed: int) -> pd.Series:
    """Check (e): random split into halves A / B, stratified by category (each category split in two equal parts).

    Within each category (sorted by id, then shuffled with one generator seeded once) the first half of the permutation is A.
    Returns 'A' / 'B' indexed by prefix id. Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for _, g in prefixes.sort_values("final_id").groupby("category", sort=True):
        ids = g["final_id"].to_numpy()
        perm = rng.permutation(len(ids))
        half = len(ids) // 2
        out.update({ids[i]: ("A" if k < half else "B") for k, i in enumerate(perm)})
    return pd.Series(out, name="half").sort_index()


def drop_word(scores: pd.DataFrame, category: str, word: str) -> pd.DataFrame:
    """Check (c): scores without the rows of one (category, word) pair, for every model. Raises if the pair is absent."""
    hit = (scores["category"] == category) & (scores["word"] == word)
    if not hit.any():
        raise ValueError(f"word not in the {category} pool")
    return scores.loc[~hit]


def pool_sizes(scores: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct words per (category, polarity_word) pool."""
    return scores.groupby(["category", "polarity_word"])["word"].nunique().rename("n_words").reset_index()


def survival_label(main_mean: float, main_significant: bool, mean: float, ci_lo: float, ci_hi: float, p: float,
                   precision_scale: bool = False, alpha: float = 0.05, precision: float = 0.05) -> str:
    """Survival of one effect under one variant.

    precision_scale (main result inside the float16 precision band of 08_rq2.py) -> 'not interpretable'.
    Null main result (CI includes 0): 'holds' if the variant CI still includes 0, else 'changes'.
    Significant main result: 'reversed' if the sign flips; 'holds' if CI excludes 0, p < alpha and |mean| >= precision;
    'weakened (precision scale)' if significant but |mean| < precision; 'weakened' otherwise (CI includes 0 or p >= alpha).
    """
    if precision_scale:
        return NOT_INTERP
    ci_excludes_0 = ci_lo > 0 or ci_hi < 0
    if not main_significant:
        return CHANGES if ci_excludes_0 else HOLDS
    if np.sign(mean) != np.sign(main_mean):
        return REVERSED
    if ci_excludes_0 and p < alpha:
        return WEAKENED_PREC if abs(mean) < precision else HOLDS
    return WEAKENED


def outcome_label(main: str, variant: str) -> str:
    """Outcome rows (H2 / H2-KTO outcome, H3 verdicts): 'holds' if the label is unchanged, else 'changes (<new label>)'."""
    return HOLDS if main == variant else f"{CHANGES} ({variant})"


def aggregate_labels(labels: dict[str, str]) -> str:
    """One summary-table cell from several subsets {subset: label}: counts per label, worst first; subsets named for non-holds."""
    def sev(lab: str) -> int:
        return SEVERITY.get(lab, SEVERITY[CHANGES] if lab.startswith(CHANGES) else 2)
    n = len(labels)
    groups: dict[str, list[str]] = {}
    for sub, lab in labels.items():
        groups.setdefault(lab, []).append(sub)
    parts = []
    for lab in sorted(groups, key=lambda x: (-sev(x), x)):
        subs = groups[lab]
        names = "" if sev(lab) == 0 or n == 1 else f" ({', '.join(subs)})"
        parts.append(f"{lab} {len(subs)}/{n}{names}" if n > 1 else lab)
    return "; ".join(parts)
