"""Topic-frequency correlate: string match of the probe words in the feedback datasets (HH-RLHF, SHP, OASST).

Word level: case-insensitive, whole word, counts per million words.
A term matches when it is not preceded or followed by an ASCII letter; inner spaces match any whitespace. Each term is counted
independently (overlaps count in both), inflected or hyphenated variants are not counted. Words = `\\w+` tokens.
Category level (used by scripts/09b2_category_frequency.py): `variant_pattern` / `detector` add plurals and space/hyphen/closed-compound
variants and count a mention once; response-pair helpers for HH, SHP and OASST. Only counts leave this module, never text (SHP: Reddit terms).
"""
from __future__ import annotations

import re
from collections.abc import Iterable

WORD = re.compile(r"\w+")
HH_ASSISTANT = "\n\nAssistant:"


def term_pattern(term: str) -> re.Pattern:
    """Whole-word pattern for a lowercase term; inner spaces become `\\s+`."""
    body = r"\s+".join(re.escape(p) for p in term.lower().split())
    return re.compile(rf"(?<![a-z]){body}(?![a-z])")


def compile_terms(terms: Iterable[str]) -> dict[str, tuple[str, re.Pattern]]:
    """term -> (first word, used as a cheap substring pre-check; compiled pattern)."""
    return {t: (t.lower().split()[0], term_pattern(t)) for t in terms}


def count_terms(text: str, compiled: dict[str, tuple[str, re.Pattern]]) -> dict[str, int]:
    """Occurrences of each term in `text` (lowercased here). The pre-check only skips texts that cannot match."""
    low = text.lower()
    return {t: (len(p.findall(low)) if first in low else 0) for t, (first, p) in compiled.items()}


def count_words(text: str) -> int:
    """Denominator: number of `\\w+` tokens."""
    return len(WORD.findall(text))


def final_assistant_turn(transcript: str) -> str:
    """Text after the last `\\n\\nAssistant:` of an HH-RLHF transcript (the turn that differs between chosen and rejected)."""
    i = transcript.rfind(HH_ASSISTANT)
    return transcript[i + len(HH_ASSISTANT):] if i >= 0 else transcript


def hh_texts(record: dict) -> list[str]:
    """HH-RLHF pair -> texts counted once: the chosen transcript and the final assistant turn of the rejected one."""
    return [record["chosen"], final_assistant_turn(record["rejected"])]


def shp_texts(record: dict, seen_posts: set[str], seen_comments: set[str]) -> list[str]:
    """SHP pair -> texts not seen before: the post `history` once per post_id, each comment once per c_root_id."""
    out = []
    if record["post_id"] not in seen_posts:
        seen_posts.add(record["post_id"])
        out.append(record["history"])
    for side in ("A", "B"):
        cid = record[f"c_root_id_{side}"]
        if cid not in seen_comments:
            seen_comments.add(cid)
            out.append(record[f"human_ref_{side}"])
    return out


# ---- category-level detector with variants, response pairs ----
IRREGULAR_PLURAL = {"man": "m[ae]n", "woman": "wom[ae]n", "person": "(?:persons|person|people)"}


def variant_pattern(term: str) -> str:
    """Regex body for a lowercase term with variants: plural on the last word (s/es; man/men, woman/women, person/people/persons) and
    multi-word terms joined by whitespace, a hyphen or nothing. Letter boundaries on both sides."""
    words = term.lower().split()
    last = IRREGULAR_PLURAL.get(words[-1], re.escape(words[-1]) + "(?:e?s)?")
    body = r"(?:\s+|-)?".join([re.escape(w) for w in words[:-1]] + [last])
    return rf"(?<![a-z]){body}(?![a-z])"


def detector(terms: Iterable[str]) -> re.Pattern:
    """One alternation over the terms, longest first, so a mention is counted once (e.g. "african american" not also "african").
    Apply to lowercased text."""
    ordered = sorted({t.lower() for t in terms}, key=lambda t: (-len(t), t))
    return re.compile("|".join(f"(?:{variant_pattern(t)})" for t in ordered))


def category_terms(vocab, exclude: set[str]) -> dict[str, dict[str, list[str]]]:
    """category -> {polarity: lowercase terms in vocabulary order}, without the excluded words."""
    out: dict[str, dict[str, list[str]]] = {}
    for cat, pol, word in zip(vocab["category"], vocab["polarity"], vocab["word"]):
        w = str(word).strip().lower()
        if w not in exclude:
            out.setdefault(cat, {}).setdefault(pol, []).append(w)
    return out


def _before_final(transcript: str) -> str:
    i = transcript.rfind(HH_ASSISTANT)
    return transcript[:i] if i >= 0 else ""


def hh_response_pair(record: dict) -> tuple[str, str]:
    """HH-RLHF pair -> (chosen final assistant turn, rejected final assistant turn)."""
    return final_assistant_turn(record["chosen"]), final_assistant_turn(record["rejected"])


def hh_differs_before_final(record: dict) -> bool:
    """True if chosen and rejected differ before their final assistant turn."""
    return _before_final(record["chosen"]) != _before_final(record["rejected"])


def shp_response_pair(record: dict) -> tuple[str, str]:
    """SHP pair -> (preferred comment, non-preferred comment); labels = 1 means A is preferred."""
    a, b = record["human_ref_A"], record["human_ref_B"]
    return (a, b) if int(record["labels"]) == 1 else (b, a)


def oasst_comparisons(df):
    """OASST messages -> one row per parent: best (lowest rank) vs worst (highest rank) English assistant reply among siblings.
    Null ranks excluded; parents with fewer than 2 ranked English assistant replies dropped. Columns: parent_id, chosen, rejected,
    n_ranked, chosen_deleted, rejected_deleted."""
    import pandas as pd
    a = df[(df["role"] == "assistant") & (df["lang"] == "en") & df["rank"].notna()]
    a = a.sort_values(["parent_id", "rank", "message_id"])
    rows = []
    for pid, g in a.groupby("parent_id", sort=True):
        if len(g) < 2 or g["rank"].iloc[0] == g["rank"].iloc[-1]:
            continue
        best, worst = g.iloc[0], g.iloc[-1]
        rows.append(dict(parent_id=pid, chosen=best["text"], rejected=worst["text"], n_ranked=len(g),
                         chosen_deleted=bool(best["deleted"]), rejected_deleted=bool(worst["deleted"])))
    return pd.DataFrame(rows, columns=["parent_id", "chosen", "rejected", "n_ranked", "chosen_deleted", "rejected_deleted"])
