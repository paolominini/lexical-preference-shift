"""Batched joint log-probability of the probe word after its context.

logP(w|S) = sum_i log P(t_i | S, t_1..t_{i-1}) over the word's tokens. The logit at position j predicts the
token at position j+1, so the word tokens ids[k] for k >= word_start are read from positions k-1.
Padding is on the right and built by hand with an attention mask: real tokens keep positions 0..L-1, so rotary
positions are unchanged and the tokenizers never need a pad_token (Base has none).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from probing.inputs import Encoded

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass(frozen=True)
class WordScore:
    """Per-token log-probs of the word and their sum (float32 log-softmax)."""
    token_logps: tuple[float, ...]

    @property
    def logp(self) -> float:
        return float(sum(self.token_logps))

    @property
    def n_tokens(self) -> int:
        return len(self.token_logps)


def pick_device(name: str = "auto") -> str:
    """'auto' = cuda if available, else mps, else cpu (on the Mac: mps)."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Seed torch (inference is deterministic anyway; this guards any sampling added later)."""
    torch.manual_seed(seed)


def load_model(model_dir: Path | str, dtype: str, device: str = "auto") -> PreTrainedModel:
    """Load a local checkpoint in `dtype` (config `run.dtype`: float16 for all six), eval mode, on `device`.

    Attention is pinned to SDPA: in float16 the eager attention of Pythia-1.4B returns NaN on MPS.
    """
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=DTYPES[dtype], attn_implementation="sdpa",
                                                 local_files_only=True)
    return model.to(pick_device(device)).eval()


def pad_right(seqs: list[tuple[int, ...]], pad_id: int = 0, extra_pad: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to the longest sequence (+ `extra_pad`); mask is 1 on real tokens, 0 on padding."""
    width = max(len(s) for s in seqs) + extra_pad
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask


def target_positions(enc: Encoded) -> tuple[list[int], list[int]]:
    """(logit positions, target token ids) for the word: position k-1 predicts ids[k]."""
    ks = range(enc.word_start, len(enc.ids))
    return [k - 1 for k in ks], [enc.ids[k] for k in ks]


@torch.inference_mode()
def score_batch(model: PreTrainedModel, encs: list[Encoded], pad_id: int = 0, extra_pad: int = 0) -> list[WordScore]:
    """Score one batch in a single forward pass. Log-softmax only at the word positions, in float32."""
    if any(e.word_start < 1 for e in encs):
        raise ValueError("the word needs at least one context token before it")
    device = next(model.parameters()).device
    ids, mask = pad_right([e.ids for e in encs], pad_id, extra_pad)
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
    rows, pos, tgt, owner = [], [], [], []
    for b, e in enumerate(encs):
        p, t = target_positions(e)
        rows += [b] * len(p)
        pos += p
        tgt += t
        owner += [b] * len(p)
    sel = logits[torch.tensor(rows, device=device), torch.tensor(pos, device=device)]  # (N, V)
    logp = torch.log_softmax(sel.float(), dim=-1)
    tok_lp = logp.gather(1, torch.tensor(tgt, device=device).unsqueeze(1)).squeeze(1).cpu().tolist()
    out: list[list[float]] = [[] for _ in encs]
    for b, v in zip(owner, tok_lp):
        out[b].append(v)
    return [WordScore(tuple(v)) for v in out]


def score(model: PreTrainedModel, encs: list[Encoded], batch_size: int = 16) -> list[WordScore]:
    """Score many pairs; batches are formed by length (less padding) and results returned in input order."""
    order = sorted(range(len(encs)), key=lambda i: len(encs[i].ids))
    res: dict[int, WordScore] = {}
    for s in range(0, len(order), batch_size):
        idx = order[s: s + batch_size]
        for i, ws in zip(idx, score_batch(model, [encs[i] for i in idx])):
            res[i] = ws
    return [res[i] for i in range(len(encs))]
