"""Tests 1-5 on a tiny random GPT-NeoX in float32 (logic, no float16 noise). Tolerance 1e-4."""
import random

import pytest
import torch

from probing.scoring import score, score_batch

ATOL = 1e-4


def logps(scores):
    return [s.logp for s in scores]


def manual_logp(model, ids, word_start):
    """Plain forward on the unpadded sequence, log-softmax by hand (float64), one token at a time."""
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0].double()
    total = 0.0
    for k in range(word_start, len(ids)):
        row = logits[k - 1]
        total += float(row[ids[k]] - torch.logsumexp(row, dim=0))
    return total


def chain_rule_logp(model, ids, word_start):
    """Independent check: one forward per word token on the growing context, last logit only."""
    total = 0.0
    with torch.no_grad():
        for k in range(word_start, len(ids)):
            last = model(input_ids=torch.tensor([ids[:k]])).logits[0, -1].double()
            total += float(torch.log_softmax(last, dim=0)[ids[k]])
    return total


def test_batched_equals_single(tiny_model, encs):
    single = [score_batch(tiny_model, [e])[0].logp for e in encs]
    batched = logps(score(tiny_model, encs, batch_size=8))
    assert len({len(e.ids) for e in encs}) > 1  # the batches really mix lengths
    assert single == pytest.approx(batched, abs=ATOL)


def test_sum_equals_manual(tiny_model, encs):
    got = logps(score(tiny_model, encs, batch_size=8))
    for e, g in zip(encs, got):
        assert g == pytest.approx(manual_logp(tiny_model, list(e.ids), e.word_start), abs=ATOL)


def test_sum_equals_chain_rule(tiny_model, encs):
    multi = [e for e in encs if e.n_word_tokens >= 2][:6]
    got = logps(score(tiny_model, multi, batch_size=8))
    for e, g in zip(multi, got):
        assert g == pytest.approx(chain_rule_logp(tiny_model, list(e.ids), e.word_start), abs=ATOL)


def test_token_logps_are_per_token(tiny_model, encs):
    for e, s in zip(encs, score(tiny_model, encs)):
        assert s.n_tokens == e.n_word_tokens
        assert all(v < 0 for v in s.token_logps)


@pytest.mark.parametrize("extra_pad,pad_id", [(0, 1), (5, 0), (20, 0), (20, 50276)])
def test_padding_independent(tiny_model, encs, extra_pad, pad_id):
    ref = logps(score_batch(tiny_model, encs))
    assert logps(score_batch(tiny_model, encs, pad_id=pad_id, extra_pad=extra_pad)) == pytest.approx(ref, abs=ATOL)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_batch_order_independent(tiny_model, encs, seed):
    ref = logps(score(tiny_model, encs, batch_size=4))
    perm = list(range(len(encs)))
    random.Random(seed).shuffle(perm)
    # score_batch directly so the shuffled order really changes batch composition
    shuffled = [encs[i] for i in perm]
    got = [s.logp for b in range(0, len(shuffled), 4) for s in score_batch(tiny_model, shuffled[b:b + 4])]
    assert [got[perm.index(i)] for i in range(len(encs))] == pytest.approx(ref, abs=ATOL)


def test_deterministic(tiny_model, encs):
    assert logps(score(tiny_model, encs)) == logps(score(tiny_model, encs))


def test_rejects_empty_context(tiny_model):
    from probing.inputs import Encoded
    with pytest.raises(ValueError):
        score_batch(tiny_model, [Encoded((5, 6), 0)])
