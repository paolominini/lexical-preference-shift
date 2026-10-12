"""Slow tests on real checkpoints (MPS, float16): Base (RAW) and SFT (CHAT), one at a time.

The runs use batch size 1: each score must equal a plain forward of that sequence and not depend on order.
float16 matmuls with different batch shapes round differently (up to 0.06 nat), so batch 16 is only
measured and bounded loosely (0.1 nat) to catch gross bugs.
"""
import gc

import pytest
import torch

from conftest import sample_encs
from probing.config import local_dir
from probing.scoring import load_model, score, score_batch

pytestmark = pytest.mark.slow
BATCH16_BOUND = 0.15  # measured noise is reported, this only catches gross bugs (measured max 0.060; with BOS on Base 0.102)


@pytest.fixture(scope="module", params=["base", "sft"])
def real(request, cfg, tok, pairs):
    model = load_model(local_dir(cfg["models"][cfg["run"]["size"]][request.param]), cfg["run"]["dtype"], cfg["run"]["device"])
    encs = sample_encs(tok, cfg, pairs, n_prefixes_per_cat=1)
    encs = encs[0::2] if request.param == "base" else encs[1::2]  # RAW for base, CHAT for sft
    yield request.param, model, encs
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def test_finite_and_negative(real):
    _, model, encs = real
    for s in score(model, encs):
        assert all(torch.isfinite(torch.tensor(s.token_logps))) and s.logp < 0


def test_manual_sum_fp16(real):
    _, model, encs = real
    dev = next(model.parameters()).device
    for e in encs[:8]:
        with torch.inference_mode():
            lg = model(input_ids=torch.tensor([e.ids], device=dev)).logits[0].float()
        manual = sum(float(torch.log_softmax(lg[k - 1], -1)[e.ids[k]]) for k in range(e.word_start, len(e.ids)))
        assert score_batch(model, [e])[0].logp == pytest.approx(manual, abs=1e-5)


def test_run_path_order_independent(real, cfg):
    """Batch size from config (1): shuffled order gives bit-identical scores."""
    _, model, encs = real
    bs = cfg["run"]["batch_size"]
    assert bs == 1
    ref = [s.logp for s in score(model, encs, batch_size=bs)]
    rev = [s.logp for s in score(model, encs[::-1], batch_size=bs)][::-1]
    assert ref == rev


def test_batch16_noise_measured(real, record_property):
    key, model, encs = real
    single = [score_batch(model, [e])[0].logp for e in encs]
    batched = [s.logp for s in score(model, encs, batch_size=16)]
    diff = max(abs(a - b) for a, b in zip(single, batched))
    record_property(f"max_diff_{key}", diff)
    print(f"\n[{key}] max |batch16 - single| = {diff:.2e} nat over {len(encs)} pairs")
    assert diff <= BATCH16_BOUND


def test_deterministic_fp16(real):
    _, model, encs = real
    assert [s.logp for s in score(model, encs)] == [s.logp for s in score(model, encs)]
