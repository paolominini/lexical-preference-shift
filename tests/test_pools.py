"""Test 8: pools and pairs (8/5/10/10/8 words per category, 4,100 pairs)."""
from collections import Counter

from probing.pools import pools

EXPECTED = {"African - Afro American": 8, "Female Homosexual": 5, "Homosexual Man": 10,
            "Sex Worker, Prostitute": 10, "Transgender Woman": 8}


def test_pool_sizes(vocab):
    p = pools(vocab)
    assert {c: sum(len(w) for w in v.values()) for c, v in p.items()} == EXPECTED
    assert all(set(v) == {"SAFE", "NOT SAFE"} for v in p.values())
    assert len(vocab) == 41 and vocab["word"].nunique() == 39


def test_pairs_count_and_category(pairs, prefixes, vocab):
    assert len(pairs) == 4100
    assert Counter(p.prefix_id for p in pairs) == {
        pid: EXPECTED[c] for pid, c in zip(prefixes["final_id"], prefixes["category"])}
    allowed = {(c, w, pol) for c, pol, w in vocab[["category", "polarity", "word"]].itertuples(index=False)}
    assert all((p.category, p.word, p.polarity_word) in allowed for p in pairs)
    assert len({(p.prefix_id, p.word) for p in pairs}) == 4100
