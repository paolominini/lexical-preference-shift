#!/usr/bin/env python3
"""
compose_final_set.py
====================
Rebuilds the final prefix set from data/selection_decisions.csv, i.e. from the
reviewed shortlist and its decisions, and checks it against data/final_prefix_set.csv.

Step 1 (AI selection): per category, 50 SAFE- and 50 NOT SAFE-sourced prefixes;
  'keep' before 'reserve', unedited before edited, then review order (random,
  fixed seed). If one polarity has fewer than 50, the other fills the gap.
Step 2 (author validation): rows the author rejected are swapped for the
  reserve named in 'replaced_by'.

  python code/prefix_extraction/compose_final_set.py --decisions data/selection_decisions.csv --final data/final_prefix_set.csv

It needs no corpus: the decision file holds every reviewed candidate with its text and decision.
"""
import argparse
import pandas as pd

LEVEL = {"unedited (the / det+adj)": 1, "unedited (other determiner)": 2, "edited a/an->the": 3}


def ai_selection(d: pd.DataFrame) -> set:
    d = d.assign(lvl=d.edit_level.map(LEVEL))
    chosen = set()
    for _, g in d.groupby("category"):
        pool = {p: pd.concat([g[(g.polarity == p) & (g.ai_decision == "keep")].sort_values(["lvl", "review_order"]),
                              g[(g.polarity == p) & (g.ai_decision == "reserve")].sort_values(["lvl", "review_order"])])
                for p in ("SAFE", "NOT SAFE")}
        n_s, n_n = min(50, len(pool["SAFE"])), min(50, len(pool["NOT SAFE"]))
        if n_s < 50: n_n = min(len(pool["NOT SAFE"]), 100 - n_s)
        if n_n < 50: n_s = min(len(pool["SAFE"]), 100 - n_n)
        chosen |= set(pool["SAFE"].head(n_s).candidate_id) | set(pool["NOT SAFE"].head(n_n).candidate_id)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--final", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.decisions)
    step1 = ai_selection(d)
    assert step1 == set(d[d.in_ai_selection].candidate_id), "AI selection does not reproduce"
    swaps = d.dropna(subset=["replaced_by"]).set_index("candidate_id").replaced_by.to_dict()
    final = (step1 - set(swaps)) | set(swaps.values())
    expected = set(pd.read_csv(a.final).candidate_id)
    print(f"step 1: {len(step1)} prefixes | author swaps: {len(swaps)} | final: {len(final)}")
    print("reproduces final_prefix_set.csv:", final == expected)


if __name__ == "__main__":
    main()
