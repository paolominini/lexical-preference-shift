"""Fast tests for the robustness subsets and the survival rule (robustness.py)."""
from __future__ import annotations

import pandas as pd

from probing.pools import load_prefixes
from probing.robustness import (aggregate_labels, corpus_group, drop_word, exclude_edited, outcome_label, pool_sizes,
                                stratified_halves, survival_label)


def _prefixes() -> pd.DataFrame:
    return load_prefixes()


def test_exclude_edited_removes_exactly_77() -> None:
    p = _prefixes()
    keep = exclude_edited(p)
    assert len(keep) == 423 and len(set(keep)) == 423
    assert set(p["final_id"]) - set(keep) == set(p.loc[p["edited"].astype(bool), "final_id"])


def test_corpus_groups_partition() -> None:
    g = corpus_group(_prefixes())
    assert (g == "added").sum() == 242 and (g == "original").sum() == 258 and g.index.is_unique and len(g) == 500


def test_drop_word_removes_one_word_only() -> None:
    s = pd.DataFrame({"category": ["A", "A", "A", "B", "A", "A"], "word": ["x", "y", "x", "x", "y", "z"],
                      "polarity_word": ["SAFE", "NOT SAFE", "SAFE", "SAFE", "NOT SAFE", "NOT SAFE"], "model": list("bbmmmm"),
                      "logp": range(6)})
    out = drop_word(s, "A", "x")
    assert len(out) == len(s) - 2
    assert not ((out["category"] == "A") & (out["word"] == "x")).any()
    assert ((out["category"] == "B") & (out["word"] == "x")).sum() == 1      # same string in another category is kept
    ps = pool_sizes(out).set_index(["category", "polarity_word"])["n_words"]
    assert ps[("A", "NOT SAFE")] == 2 and ("A", "SAFE") not in ps.index
    try:
        drop_word(s, "B", "z")
        raise AssertionError("absent word must raise")
    except ValueError:
        pass


def test_halves_stratified_disjoint_reproducible() -> None:
    p = _prefixes()
    h = stratified_halves(p, 42)
    assert len(h) == 500 and h.index.is_unique and set(h) == {"A", "B"}
    cat = p.set_index("final_id")["category"]
    counts = pd.crosstab(cat.loc[h.index], h)
    assert (counts == 50).all().all()
    assert h.equals(stratified_halves(p, 42))
    assert not h.equals(stratified_halves(p, 43))


def test_survival_label_hand_cases() -> None:
    assert survival_label(0.5, True, 0.4, 0.2, 0.6, 1e-5) == "holds"
    assert survival_label(0.5, True, 0.3, -0.1, 0.7, 0.2) == "weakened"
    assert survival_label(0.5, True, 0.03, 0.01, 0.05, 1e-3) == "weakened (precision scale)"
    assert survival_label(0.5, True, -0.1, -0.3, 0.1, 0.4) == "reversed"
    assert survival_label(0.02, False, 0.01, -0.05, 0.07, 0.6) == "holds"
    assert survival_label(0.02, False, 0.14, 0.07, 0.21, 0.002) == "changes"
    assert survival_label(-0.04, True, -0.04, -0.05, -0.03, 1e-9, precision_scale=True) == "not interpretable"
    assert outcome_label("supported", "supported") == "holds"
    assert outcome_label("differ", "mixed") == "changes (mixed)"


def test_aggregate_labels_worst_first() -> None:
    assert aggregate_labels({"x": "holds", "y": "holds"}) == "holds 2/2"
    assert aggregate_labels({"x": "holds", "y": "weakened"}) == "weakened 1/2 (y); holds 1/2"
    assert aggregate_labels({"only": "reversed"}) == "reversed"
