#!/usr/bin/env python3
"""
select_candidates.py
====================
Builds the list of candidate prefixes for MANUAL validation.

1. Extraction with build_prefix_dataset.py (minimal-edit mode).
2. Automatic semantic pre-filters on the ORIGINAL sentence (spaCy):
     - modifier use: the source word modifies another noun ("the African coast")
     - metalinguistic use: the word is mentioned, not used ("the word ...", "the term ...")
     - off-category compound: "sand ..." / "dune ..." + slur (anti-Arab slurs)
3. Per category, sample N candidates (default 140), half SAFE- and half
   NOT SAFE-sourced; if one polarity has too few, the other fills the gap.
   Categories with fewer than N candidates are taken whole.
4. Writes review_candidates.xlsx (to be filled by hand) and candidates.csv.

After review, keep only rows marked "y": they are the final prefix set.

This script does not by itself reproduce the 972-candidate shortlist that was reviewed
(its default is --n 140 and it has no extension step), and it needs the source corpora,
which are not redistributed. The final set is rebuilt from data/selection_decisions.csv
(every reviewed candidate with its decision) by compose_final_set.py.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd
import build_prefix_dataset as B

META = {"word", "words", "term", "terms", "slur", "slurs", "label", "n", "phrase", "name"}
OFFCAT = {"sand", "dune"}
DIRECTIONS = {"south", "north", "east", "west", "central", "southern", "northern", "eastern", "western", "sub", "saharan"}
# heads that still denote people (collective or idiomatic); WordNet handles the rest
PERSON_EXTRA = {"people", "folks", "folk", "community", "population", "men", "women", "kids", "children",
                "ass", "family", "families", "crowd", "gang"}


def _is_common_noun(w: str) -> bool:
    """True if WordNet knows w and its most frequent sense is a noun (filters 'lol', 'ps', names, verbs)."""
    from nltk.corpus import wordnet as wn
    s = wn.synsets(w.lower())
    return bool(s) and s[0].pos() == "n"


def _is_person_noun(w: str) -> bool:
    """True if one of the first two noun senses of w is a kind of person (WordNet)."""
    w = w.lower()
    if w in PERSON_EXTRA:
        return True
    from nltk.corpus import wordnet as wn
    person = wn.synset("person.n.01")
    for s in wn.synsets(w, pos="n")[:2]:
        if s == person or person in s.closure(lambda x: x.hypernyms()):
            return True
    return False
SHORT = {"African - Afro American": "African", "Female Homosexual": "Female Homosexual",
         "Homosexual Man": "Homosexual Man", "Sex Worker, Prostitute": "Sex Worker",
         "Transgender Woman": "Transgender Woman"}


def semantic_flags(df: pd.DataFrame) -> pd.DataFrame:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    reasons = []
    for r in df.itertuples():
        sent, tail = r.ctx_sent, r.orig_tail
        text = f"{sent} {tail} ".lstrip()
        start = len(text)
        word_surface = r.source_word
        text = text + word_surface + r.right_ctx[:60]
        doc = nlp(text)
        span = doc.char_span(start, start + len(word_surface), alignment_mode="expand")
        why = ""
        pre = (sent.lower().split()[-2:] if sent else []) + tail.lower().split()
        if set(t.strip('"\'') for t in pre) & META:
            why = "metalinguistic"
        elif set(tail.lower().split()) & OFFCAT or r.right_ctx.lower().lstrip().startswith(("dune", "sand")):
            why = "off-category compound"
        elif tail and tail.split()[-1].lower().strip("-") in DIRECTIONS:
            why = "geographic compound (South/North/Central ...)"
        elif re.match(r"\s*-?\s*americans?\b", r.right_ctx, re.I) and r.source_word == "african":
            why = "part of 'African American(s)'"
        elif span is not None and len(span):
            last = span[-1]
            nxt = doc[last.i + 1] if last.i + 1 < len(doc) else None
            head = last.head if last.head.i > last.i else None
            if head is None and nxt is not None and last.dep_ == "nmod":
                head = nxt
            if (head is not None and head.i == last.i + 1 and head.pos_ == "PROPN" and head.text[:1].isupper()
                    and not _is_person_noun(head.text)):
                why = "part of a proper name"
            # reject only when the word modifies a NON-person noun ("the African coast", "the word X")
            if (head is not None and last.dep_ in {"compound", "amod", "nmod"} and head.i == last.i + 1
                    and head.pos_ == "NOUN" and head.text.isalpha() and head.text.islower()
                    and _is_common_noun(head.text) and not _is_person_noun(head.text)):
                why = why or "modifier of a non-person noun"
        reasons.append(why)
    out = df.copy()
    out["auto_reject"] = reasons
    return out


def sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    parts = []
    for c, g in df.groupby("category"):
        if len(g) <= n:
            parts.append(g); continue
        safe, bad = g[g.polarity == "SAFE"], g[g.polarity == "NOT SAFE"]
        k_safe = min(len(safe), n // 2)
        k_bad = min(len(bad), n - k_safe)
        k_safe = min(len(safe), n - k_bad)
        parts.append(pd.concat([safe.sample(k_safe, random_state=seed), bad.sample(k_bad, random_state=seed)]))
    return pd.concat(parts).sample(frac=1, random_state=seed).sort_values("category", kind="stable")


def write_workbook(cand: pd.DataFrame, path: Path, n_target: int, floor: int):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    F = Font(name="Arial", size=10); FB = Font(name="Arial", size=10, bold=True)
    wb = Workbook()

    ws = wb.active; ws.title = "Instructions"
    lines = [
        ("Validation of the candidate prefixes", FB),
        ("", F),
        ("For each row of the 'Candidates' sheet fill in the keep column (yellow cells): y = keep, n = reject.", F),
        ("If you reject, choose a reason in 'reason' (drop-down menu). 'notes' is free text.", F),
        ("", F),
        ("Keep the row only if all three are true:", FB),
        ("1. In the original sentence the word refers to a person (not an adjective of another noun, not 'the word X').", F),
        ("2. The prefix (what the model sees) makes sense and every word of the two pools would fit grammatically after it.", F),
        ("3. The person spoken about can belong to the category (no off-category reference).", F),
        ("", F),
        ("Columns: 'original' = original sentence with the word between [[ ]]; 'prefix' = text given to the model, the slot is ___.", F),
        ("'edited' = y only if an a/an before the word was replaced with 'the' (the only edited case).", F),
        ("", F),
        ("Filled-in example:", FB),
        ("original: I had to use a lot of strength to say he's a [[X]] boy   |   prefix: ... say he's the ___   |   keep: n   |   reason: non-person / modifier use", F),
        ("", F),
        (f"Target: at least {floor} 'y' rows per category (see the 'Summary' sheet).", F),
    ]
    for i, (t, f) in enumerate(lines, 1):
        c = ws.cell(row=i, column=1, value=t); c.font = f
    ws.column_dimensions["A"].width = 150

    ws = wb.create_sheet("Candidates")
    cols = ["cand_id", "category", "tier", "edited", "source_word", "polarity", "corpus",
            "original", "prefix", "keep", "reason", "notes"]
    widths = [9, 18, 5, 7, 16, 10, 11, 70, 70, 7, 26, 30]
    for j, (h, w) in enumerate(zip(cols, widths), 1):
        c = ws.cell(row=1, column=j, value=h); c.font = FB
        ws.column_dimensions[c.column_letter].width = w
    yellow = PatternFill("solid", start_color="FFFF00")
    wrap = Alignment(wrap_text=True, vertical="top")
    for i, r in enumerate(cand.itertuples(), 2):
        vals = [r.cand_id, SHORT[r.category], r.tier, "y" if r.edited else "n", r.source_word, r.polarity,
                r.corpus, r.original, r.prefix + " ___", None, None, None]
        for j, v in enumerate(vals, 1):
            c = ws.cell(row=i, column=j, value=v); c.font = F; c.alignment = wrap
        for j in (10, 11, 12):
            ws.cell(row=i, column=j).fill = yellow
    last = len(cand) + 1
    dv1 = DataValidation(type="list", formula1='"y,n"', allow_blank=True)
    dv2 = DataValidation(type="list", allow_blank=True,
                         formula1='"non-person / modifier use,metalinguistic (the word X),meaning changed by edit,incomprehensible,off-category,other"')
    ws.add_data_validation(dv1); ws.add_data_validation(dv2)
    dv1.add(f"J2:J{last}"); dv2.add(f"K2:K{last}")
    ws.freeze_panes = "A2"; ws.auto_filter.ref = f"A1:L{last}"

    ws = wb.create_sheet("Summary")
    hdr = ["category", "candidates", "kept (y)", "rejected (n)", "to do", f"floor {floor} met"]
    for j, h in enumerate(hdr, 1):
        c = ws.cell(row=1, column=j, value=h); c.font = FB
        ws.column_dimensions[c.column_letter].width = 20
    for i, cat in enumerate(SHORT.values(), 2):
        ws.cell(row=i, column=1, value=cat)
        ws.cell(row=i, column=2, value=f'=COUNTIF(Candidates!$B$2:$B${last},A{i})')
        ws.cell(row=i, column=3, value=f'=COUNTIFS(Candidates!$B$2:$B${last},A{i},Candidates!$J$2:$J${last},"y")')
        ws.cell(row=i, column=4, value=f'=COUNTIFS(Candidates!$B$2:$B${last},A{i},Candidates!$J$2:$J${last},"n")')
        ws.cell(row=i, column=5, value=f'=B{i}-C{i}-D{i}')
        ws.cell(row=i, column=6, value=f'=IF(C{i}>={floor},"yes","no")')
        for j in range(1, 7):
            ws.cell(row=i, column=j).font = F
    t = len(SHORT) + 2
    ws.cell(row=t, column=1, value="Total").font = FB
    for j, col in zip(range(2, 6), "BCDE"):
        c = ws.cell(row=t, column=j, value=f"=SUM({col}2:{col}{t - 1})"); c.font = FB
    ws.cell(row=t + 2, column=1, value=f"Candidates per category: {n_target} sampled (all if fewer available); "
                                       "at most half SAFE-sourced. Floor set with the thesis supervisors.").font = F
    wb.move_sheet("Summary", offset=-1)
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--n", type=int, default=140)
    ap.add_argument("--floor", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    excl = B.EXCLUSIONS
    cand_all, final = B.build_final(a.data_dir, a.vocab, B.Config(), excl)
    flagged = semantic_flags(final)
    ok = flagged[flagged.auto_reject == ""]
    cand = sample(ok, a.n, a.seed).reset_index(drop=True)
    cand.insert(0, "cand_id", [f"c{i:04d}" for i in range(len(cand))])
    cand["original"] = [f"{s} {ot} [[{w}]]{rc[:80]}".replace("  ", " ").strip()
                        for s, ot, w, rc in zip(cand.ctx_sent, cand.orig_tail, cand.source_word, cand.right_ctx)]
    cand.to_csv(out / "candidates.csv", index=False)
    write_workbook(cand, out / "review_candidates.xlsx", a.n, a.floor)

    summ = pd.DataFrame({
        "extracted": final.groupby("category").size(),
        "auto-rejected": flagged[flagged.auto_reject != ""].groupby("category").size(),
        "available": ok.groupby("category").size(),
        "SAFE available": ok[ok.polarity == "SAFE"].groupby("category").size(),
        "candidates": cand.groupby("category").size(),
        "SAFE in candidates": cand[cand.polarity == "SAFE"].groupby("category").size(),
        "edited (a/an->the)": cand[cand.edited].groupby("category").size(),
    }).fillna(0).astype(int).rename(index=SHORT)
    print(summ.to_string())
    print("\nauto-reject reasons:", flagged.auto_reject.value_counts().drop("", errors="ignore").to_dict())
    return flagged, cand


if __name__ == "__main__":
    main()
