"""Tests for the topic-frequency string-match counters on toy strings with neutral terms only."""
from __future__ import annotations

from probing.frequency import compile_terms, count_terms, count_words, final_assistant_turn, hh_texts, shp_texts


def _counts(text: str, terms: list[str]) -> dict[str, int]:
    return count_terms(text, compile_terms(terms))


def test_case_insensitive_whole_word() -> None:
    c = _counts("Gay rights. GAY. gays, gayness, (gay) gay2", ["gay"])
    assert c["gay"] == 4                        # Gay, GAY, (gay), gay2; not "gays" (inflected) nor "gayness"


def test_inner_space_hyphen_and_overlap() -> None:
    text = "a sex   worker; a sex\nworker; a sex-worker; sex workers; homosexual man and homosexual"
    c = _counts(text, ["sex worker", "homosexual", "homosexual man"])
    assert c["sex worker"] == 2                 # any whitespace inside; hyphen and plural not counted at word level
    assert c["homosexual man"] == 1
    assert c["homosexual"] == 2                 # counted independently: also inside "homosexual man"


def test_no_match_inside_longer_words() -> None:
    c = _counts("Africanist, panafrican, african.", ["african"])
    assert c["african"] == 1


def test_zero_for_absent_terms() -> None:
    assert _counts("nothing here", ["black person", "call girl"]) == {"black person": 0, "call girl": 0}


def test_count_words() -> None:
    assert count_words("Hello, world! It's 2 o'clock.") == 7    # \w+ tokens: Hello world It s 2 o clock
    assert count_words("") == 0


def test_hh_final_turn_and_texts() -> None:
    chosen = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: and?\n\nAssistant: a gay man"
    rejected = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: and?\n\nAssistant: no"
    assert final_assistant_turn(chosen) == " a gay man"
    texts = hh_texts({"chosen": chosen, "rejected": rejected})
    assert texts == [chosen, " no"]             # chosen transcript once + rejected final turn (shared history not double counted)


def test_shp_dedup() -> None:
    seen_posts: set[str] = set()
    seen_comments: set[str] = set()
    r1 = {"post_id": "p1", "history": "post", "c_root_id_A": "a", "c_root_id_B": "b", "human_ref_A": "ca", "human_ref_B": "cb"}
    r2 = {"post_id": "p1", "history": "post", "c_root_id_A": "a", "c_root_id_B": "c", "human_ref_A": "ca", "human_ref_B": "cc"}
    assert shp_texts(r1, seen_posts, seen_comments) == ["post", "ca", "cb"]
    assert shp_texts(r2, seen_posts, seen_comments) == ["cc"]    # same post and comment A already counted


# ---- category level ----
import pandas as pd  # noqa: E402

from probing.frequency import (category_terms, detector, hh_differs_before_final, hh_response_pair, oasst_comparisons,  # noqa: E402
                               shp_response_pair)


def test_variants_plural_irregular_hyphen_compound() -> None:
    d = detector(["trans woman", "black person", "lesbian", "gay man"])
    text = ("Trans women, a trans-woman, one transwoman; black people, black persons; Lesbians; gay men; "
            "gay human; lesbianism; transwomanhood")
    assert len(d.findall(text.lower())) == 7    # not "gay human", "lesbianism", "transwomanhood"


def test_detector_longest_first_counts_a_mention_once() -> None:
    d = detector(["african", "african american"])
    assert d.findall("an african american and an african") == ["african american", "african"]


def test_category_terms_exclusion() -> None:
    vocab = pd.DataFrame({"category": ["C", "C", "C"], "polarity": ["SAFE", "SAFE", "NOT SAFE"], "word": ["Escort", "sex worker", "x"]})
    assert category_terms(vocab, exclude={"escort"}) == {"C": {"SAFE": ["sex worker"], "NOT SAFE": ["x"]}}
    assert category_terms(vocab, exclude=set())["C"]["SAFE"] == ["escort", "sex worker"]


def test_hh_pair_helpers() -> None:
    same = {"chosen": "\n\nHuman: q\n\nAssistant: good", "rejected": "\n\nHuman: q\n\nAssistant: bad"}
    diff = {"chosen": "\n\nHuman: q\n\nAssistant: good", "rejected": "\n\nHuman: other\n\nAssistant: bad"}
    assert hh_response_pair(same) == (" good", " bad")
    assert not hh_differs_before_final(same)
    assert hh_differs_before_final(diff)


def test_shp_label_direction() -> None:
    r = {"human_ref_A": "a", "human_ref_B": "b", "labels": 1}
    assert shp_response_pair(r) == ("a", "b")              # labels = 1 -> A preferred
    assert shp_response_pair({**r, "labels": 0}) == ("b", "a")


def test_oasst_best_vs_worst_null_rank_excluded() -> None:
    df = pd.DataFrame({
        "message_id": ["m1", "m2", "m3", "m4", "m5", "m6"],
        "parent_id": ["p", "p", "p", "p", "q", "q"],
        "role": ["assistant"] * 6,
        "lang": ["en", "en", "en", "de", "en", "en"],
        "rank": [1.0, 0.0, None, 3.0, 0.0, None],
        "text": ["mid", "best", "unranked", "german", "only", "unranked2"],
        "deleted": [False] * 6,
    })
    out = oasst_comparisons(df)
    assert len(out) == 1                                   # parent q has a single ranked English reply
    assert (out.iloc[0]["chosen"], out.iloc[0]["rejected"]) == ("best", "mid")
