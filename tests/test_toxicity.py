"""Toxicity covariate: table structure and order (fast, fake model) and the real Detoxify checkpoint (slow)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probing.toxicity import LABELS, MODEL_TYPE, load_detoxify, score_texts, toxicity_table


class FakeDetoxify:
    """Scores depend only on the text (its length), like a model with working padding masks; counts the batches."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def predict(self, texts: list[str]) -> dict[str, list[float]]:
        self.batches.append(len(texts))
        return {k: [(len(t) % 10 + i) / 20 for t in texts] for i, k in enumerate(LABELS)}


def test_score_texts_keeps_order_and_batches():
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    fake = FakeDetoxify()
    out = score_texts(fake, texts, batch_size=2)
    assert fake.batches == [2, 2, 1]
    assert list(out.columns) == LABELS and len(out) == 5
    assert out["toxicity"].tolist() == [len(t) / 20 for t in texts]
    pd.testing.assert_frame_equal(out, score_texts(FakeDetoxify(), texts, batch_size=1))


def test_toxicity_table_has_provenance_and_full_coverage():
    pre = pd.DataFrame({"final_id": ["X1", "X2", "X3"], "category": ["a", "a", "b"], "polarity": ["SAFE", "NOT SAFE", "SAFE"],
                        "prefix": ["one the", "two words the", "three"]})
    tab = toxicity_table(pre, FakeDetoxify(), batch_size=2)
    assert tab["prefix_id"].tolist() == ["X1", "X2", "X3"]
    assert tab["source_polarity"].tolist() == ["SAFE", "NOT SAFE", "SAFE"]
    assert tab["n_chars"].tolist() == [7, 13, 5]
    assert (tab["detoxify_model"] == MODEL_TYPE).all() and tab[LABELS].notna().all().all()


@pytest.mark.slow
def test_real_detoxify_batched_equals_single_and_deterministic():
    """Padding independence and determinism on the real `unbiased` checkpoint (CPU)."""
    model = load_detoxify("cpu")
    texts = ["Thank you for the kind words. I really appreciate the", "You are a stupid idiot and everybody hates the",
             "The committee will meet on Tuesday to discuss the", "Shut up, you worthless piece of garbage. Nobody wants the"]
    a, b, c = score_texts(model, texts, 4), score_texts(model, texts, 1), score_texts(model, texts, 4)
    assert np.abs(a.to_numpy() - b.to_numpy()).max() < 1e-4
    assert np.array_equal(a.to_numpy(), c.to_numpy())
    assert a["toxicity"].iloc[1] > 0.5 > a["toxicity"].iloc[0] and a["toxicity"].iloc[3] > a["toxicity"].iloc[2]
