"""Resumable scoring loop for one loaded checkpoint (used by the pilot and the full run).

Pairs are processed prefix by prefix; pairs already in the writer are skipped (resume); rows are written as a new
part every `every` prefixes, so an interruption loses at most `every` prefixes of work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby
from time import perf_counter
from typing import Callable

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from probing.config import is_aligned
from probing.inputs import context_text, encode_pair
from probing.io import PartWriter
from probing.pools import Pair
from probing.scoring import score


@dataclass
class RunStats:
    """What one session did: pairs scored/skipped and wall time per prefix (forward passes only)."""
    n_scored: int = 0
    n_skipped: int = 0
    prefix_seconds: dict[str, float] = field(default_factory=dict)
    prefix_pairs: dict[str, int] = field(default_factory=dict)


def input_format(key: str, cfg: dict) -> str:
    """'raw' for Base, 'chat' for the five aligned checkpoints (fixed by checkpoint type, never crossed)."""
    return cfg["input_format"]["aligned" if is_aligned(key) else "base"]


def run_checkpoint(model: PreTrainedModel, tok: PreTrainedTokenizerBase, pairs: list[Pair], key: str, repo: str,
                   cfg: dict, writer: PartWriter, *, bos: bool, dtype: str, every: int = 10,
                   max_prefixes: int | None = None, on_prefix: Callable[[int], None] | None = None) -> RunStats:
    """Score every pair not yet in `writer` with batch size `cfg['run']['batch_size']` (1: the batch shape changes float16 rounding).

    `bos` is passed explicitly (main run: `config.uses_bos`, True for all six; robustness check: Base without).
    `max_prefixes` stops after that many prefixes with new work (used to simulate an interruption).
    `on_prefix(n_pairs)` is called once per prefix, scored or skipped as already done (progress bar).
    """
    fmt = input_format(key, cfg)
    stats, buf, n_new = RunStats(), [], 0
    seen: set[str] = set()
    for pid, grp in groupby(pairs, key=lambda p: p.prefix_id):
        if pid in seen:
            raise ValueError("pairs must be grouped by prefix")
        seen.add(pid)
        grp = list(grp)
        todo = [p for p in grp if not writer.is_done(pid, key, p.word)]
        stats.n_skipped += len(grp) - len(todo)
        if not todo:
            if on_prefix:
                on_prefix(len(grp))
            continue
        if max_prefixes is not None and n_new >= max_prefixes:
            break
        n_new += 1
        ctx = context_text(grp[0].prefix, fmt, cfg["chat"])
        encs = [encode_pair(tok, ctx, p.word, bos=bos) for p in todo]
        t0 = perf_counter()
        scores = score(model, encs, batch_size=cfg["run"]["batch_size"])  # .cpu() inside syncs the device
        stats.prefix_seconds[pid] = perf_counter() - t0
        stats.prefix_pairs[pid] = len(todo)
        for p, ws in zip(todo, scores):
            buf.append(dict(prefix_id=pid, category=p.category, model=key, model_repo=repo, word=p.word,
                            polarity_word=p.polarity_word, n_tokens=ws.n_tokens, logp=ws.logp, input_format=fmt,
                            bos=bos, dtype=dtype))
        stats.n_scored += len(todo)
        if on_prefix:
            on_prefix(len(grp))
        if n_new % every == 0:
            writer.write(buf)
            buf = []
    if buf:
        writer.write(buf)
    return stats
