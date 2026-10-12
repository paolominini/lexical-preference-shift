"""Tests 6-7: BOS exactly once and exact word-token positions, on the real CHAT string."""
import pytest

from probing.config import is_aligned, uses_bos
from probing.inputs import BOS_ID, context_text, encode_pair
from probing.scoring import target_positions

KEYS = ["base", "sft", "sft_dpo", "sft_ppo", "sft_kto", "kto"]


def one_per_category(pairs):
    seen, out = set(), []
    for p in pairs:
        if p.category not in seen:
            seen.add(p.category)
            out += [q for q in pairs if q.prefix_id == p.prefix_id]
    return out


def enc_for(tok, cfg, p, key, bos=None):
    fmt = "chat" if is_aligned(key) else "raw"
    return encode_pair(tok, context_text(p.prefix, fmt, cfg["chat"]), p.word,
                       bos=uses_bos(key, cfg) if bos is None else bos)


@pytest.mark.parametrize("key", KEYS)
def test_bos_exactly_once(tok, cfg, pairs, key):
    for p in one_per_category(pairs):
        ids = enc_for(tok, cfg, p, key).ids
        assert uses_bos(key, cfg)  # BOS for all six
        assert ids[0] == BOS_ID and ids.count(BOS_ID) == 1


def test_bos_sensitivity_flag_removes(tok, cfg, pairs):
    """Robustness check (d): Base without BOS; the flag also works for the aligned keys."""
    p = pairs[0]
    assert BOS_ID not in enc_for(tok, cfg, p, "base", bos=False).ids
    assert BOS_ID not in enc_for(tok, cfg, p, "sft", bos=False).ids


def test_chat_string_is_exact(tok, cfg, pairs):
    """The decoded context is exactly the TuluV2 frame + prefix (AF000)."""
    p = pairs[0]
    assert p.prefix_id == "AF000"
    e = enc_for(tok, cfg, p, "sft")
    expected = "<|user|>\nContinue the following sentence naturally.\n<|assistant|>\n" + p.prefix
    assert tok.decode(e.ids[1:e.word_start]) == expected
    assert tok.decode(e.ids[e.word_start:]) == " " + p.word


@pytest.mark.parametrize("key", ["base", "sft"])
def test_word_position_real_strings(tok, cfg, pairs, key):
    for p in one_per_category(pairs):
        e = enc_for(tok, cfg, p, key)
        word_ids = tok.encode(" " + p.word, add_special_tokens=False)
        assert list(e.ids[e.word_start:]) == word_ids
        assert e.n_word_tokens == len(word_ids)
        pos, tgt = target_positions(e)
        assert tgt == word_ids
        assert pos == list(range(e.word_start - 1, len(e.ids) - 1))


def test_multitoken_words_covered(tok, vocab):
    """1, 2, 3 and 4-token words all exist in the vocabulary, so the gather is tested on each length."""
    lengths = {len(tok.encode(" " + w, add_special_tokens=False)) for w in vocab["word"]}
    assert lengths == {1, 2, 3, 4}


@pytest.mark.parametrize("fmt,bos", [("raw", False), ("chat", True)])
def test_all_pairs_encode(tok, cfg, pairs, fmt, bos):
    """Boundary check passes for all 4,100 pairs in both formats (repeats the tokenization round-trip check in the builder)."""
    for p in pairs:
        encode_pair(tok, context_text(p.prefix, fmt, cfg["chat"]), p.word, bos=bos)


def test_trailing_space_rejected(tok):
    with pytest.raises(ValueError):
        encode_pair(tok, "the ", "word", bos=False)
