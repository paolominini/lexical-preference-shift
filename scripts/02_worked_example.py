"""Worked example: log P(" word" | S) by hand vs by `scoring.score_batch`, Base (RAW) and SFT (CHAT).

Uses one real prefix and one neutral two-token word (derogatory words are never printed). Writes reports/step_02_tables/worked_example.csv
and prints at most ~20 lines.

Usage: .venv/bin/python scripts/02_worked_example.py --prefix AF000 --word african
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from probing.config import is_aligned, load_config, local_dir  # noqa: E402
from probing.inputs import context_text, encode_pair  # noqa: E402
from probing.pools import load_prefixes  # noqa: E402
from probing.scoring import load_model, score_batch, set_seed  # noqa: E402


def by_hand(model, ids: tuple[int, ...], word_start: int) -> list[dict]:
    """Plain forward of the unpadded sequence; for each word token: logit - logsumexp at the previous position."""
    dev = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([ids], device=dev)).logits[0].float()
    rows = []
    for k in range(word_start, len(ids)):
        row = logits[k - 1]
        lp = float(row[ids[k]] - torch.logsumexp(row, dim=0))
        rank = int((row > row[ids[k]]).sum()) + 1
        rows.append(dict(position=k - 1, token_id=ids[k], logit=float(row[ids[k]]),
                         logsumexp=float(torch.logsumexp(row, dim=0)), logp=lp, prob=math.exp(lp), rank=rank))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="AF000")
    ap.add_argument("--word", default="african")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    pfx = load_prefixes().set_index("final_id").loc[args.prefix, "prefix"]
    out = []
    for key in ("base", "sft"):
        d = local_dir(cfg["models"]["1.4b"][key], cfg["paths"]["models_dir"])
        tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
        fmt = "chat" if is_aligned(key) else "raw"
        enc = encode_pair(tok, context_text(pfx, fmt, cfg["chat"]), args.word, bos=is_aligned(key))
        model = load_model(d, cfg["run"]["dtype"], cfg["run"]["device"])
        hand = by_hand(model, enc.ids, enc.word_start)
        func = score_batch(model, [enc])[0]
        for r, f in zip(hand, func.token_logps):
            out.append(dict(model=key, format=fmt, seq_len=len(enc.ids), word_start=enc.word_start,
                            token=tok.convert_ids_to_tokens(r["token_id"]), **r, logp_function=f))
        print(f"{key} ({fmt.upper()}, {len(enc.ids)} tokens, word starts at {enc.word_start}):")
        for r in hand:
            print(f"  pos {r['position']:>2} -> {tok.convert_ids_to_tokens(r['token_id'])!r:12s} logit {r['logit']:7.3f}"
                  f" - logsumexp {r['logsumexp']:7.3f} = {r['logp']:8.4f}  (p={r['prob']:.2e}, rank {r['rank']})")
        tot = sum(r["logp"] for r in hand)
        print(f"  by hand sum = {tot:.4f} | function = {func.logp:.4f} | diff = {abs(tot - func.logp):.1e}"
              f" | P(word) = {math.exp(tot):.2e}")
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    tdir = ROOT / "reports/step_02_tables"
    tdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(tdir / "worked_example.csv", index=False)


if __name__ == "__main__":
    main()
