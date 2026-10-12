"""Toxicity covariate: Detoxify `unbiased` run locally on the truncated prefix.

The text scored is exactly the `prefix` column of the prefix set (text up to and including the last word before the
probe slot), never the original sentence. Scores are probabilities in [0, 1] (sigmoid of the model logits).
"""
from __future__ import annotations

import pandas as pd
import torch

MODEL_TYPE = "unbiased"
# Detoxify `unbiased` outputs, in the order returned by the library.
LABELS = ["toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"]


def load_detoxify(device: str = "cpu"):
    """Detoxify `unbiased`. The checkpoint is downloaded once by the library into the torch hub cache."""
    from detoxify import Detoxify
    return Detoxify(MODEL_TYPE, device=device)


@torch.inference_mode()
def score_texts(model, texts: list[str], batch_size: int = 16) -> pd.DataFrame:
    """One row per text, one column per Detoxify label, same order as `texts`.

    Texts are scored in fixed-size batches in the given order, so a score depends only on the texts of its batch
    through padding, which the attention mask removes (checked in tests/test_toxicity.py).
    """
    rows: list[dict[str, float]] = []
    for i in range(0, len(texts), batch_size):
        out = model.predict(texts[i:i + batch_size])
        n = len(texts[i:i + batch_size])
        rows += [{k: float(out[k][j]) for k in out} for j in range(n)]
    return pd.DataFrame(rows)


def toxicity_table(prefixes: pd.DataFrame, model, batch_size: int = 16) -> pd.DataFrame:
    """Scores for every prefix, with provenance columns (prefix_id, category, source polarity, text length)."""
    scores = score_texts(model, prefixes["prefix"].tolist(), batch_size)
    meta = prefixes[["final_id", "category", "polarity"]].rename(columns={"final_id": "prefix_id", "polarity": "source_polarity"})
    out = pd.concat([meta.reset_index(drop=True), scores], axis=1)
    out.insert(3, "n_chars", prefixes["prefix"].str.len().to_numpy())
    out["detoxify_model"] = MODEL_TYPE
    return out
