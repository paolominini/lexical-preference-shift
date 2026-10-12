"""Full scoring run: 500 prefixes x 6 checkpoints (4,100 pairs each), then robustness (d).

Main run: every checkpoint with BOS from `config.uses_bos` (`<|endoftext|>` first for all six), batch size 1,
SDPA, weights in config `run.dtype` (float16), log-softmax float32. Parts go to results/scores_<size>/<model>/.
Robustness (d), scored last: Base without BOS, parts in results/scores_<size>_base_nobos/base/.
The two sets are never mixed. Resumable: pairs already in the parts are skipped, so re-running the same command after
an interruption continues where it stopped. One model in memory at a time. This script does not consolidate the
parts (see 04b_gate_and_consolidate.py).

Usage:
  .venv/bin/python scripts/04_run_scoring.py --smoke   # 2 prefixes x Base in a temp folder, prints one input
  caffeinate -i .venv/bin/python scripts/04_run_scoring.py
"""
from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from probing.config import load_config, local_dir, uses_bos  # noqa: E402
from probing.inputs import BOS_ID, context_text, encode_pair  # noqa: E402
from probing.io import PartWriter, read_parts  # noqa: E402
from probing.pools import Pair, build_pairs, load_prefixes, load_vocabulary  # noqa: E402
from probing.run import input_format, run_checkpoint  # noqa: E402
from probing.scoring import load_model, set_seed  # noqa: E402

N_TOKENIZER = 50277
EVERY = 10   # a new part every 10 prefixes (an interruption loses at most ~10 prefixes of work)


def release() -> None:
    """Return freed memory to the system. The caller must `del` its model first: only one model may be in memory."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def show_input(tok, pair: Pair, key: str, cfg: dict, bos: bool) -> None:
    """Print the exact ids of one pair: BOS check on the code path the run uses."""
    ctx = context_text(pair.prefix, input_format(key, cfg), cfg["chat"])
    enc = encode_pair(tok, ctx, pair.word, bos=bos)
    toks = tok.convert_ids_to_tokens(list(enc.ids))
    print(f"  [{key}] bos={bos} first ids {list(enc.ids[:6])} = {toks[:6]}; n_ids={len(enc.ids)}; "
          f"word_start={enc.word_start}, word tokens={enc.n_word_tokens} (word not printed)")
    assert (enc.ids[0] == BOS_ID) == bos, f"{key}: first id does not match bos={bos}"


def smoke(cfg: dict, pairs: list[Pair]) -> None:
    """Tokenizer-only input check for all six, then 2 prefixes x Base through `run_checkpoint` (nothing saved)."""
    first = pairs[0]
    print(f"first pair: prefix_id={first.prefix_id}")
    toks = {}
    for key, repo in cfg["models"][cfg["run"]["size"]].items():
        toks[key] = AutoTokenizer.from_pretrained(local_dir(repo, cfg["paths"]["models_dir"]), local_files_only=True)
        assert len(toks[key]) == N_TOKENIZER
        show_input(toks[key], first, key, cfg, uses_bos(key, cfg))
    ids = list(dict.fromkeys(p.prefix_id for p in pairs))[:2]
    sub = [p for p in pairs if p.prefix_id in ids]
    d = local_dir(cfg["models"][cfg["run"]["size"]]["base"], cfg["paths"]["models_dir"])
    model = load_model(d, cfg["run"]["dtype"], cfg["run"]["device"])
    with tempfile.TemporaryDirectory() as tmp:
        w = PartWriter(Path(tmp) / "base")
        t0 = perf_counter()
        run_checkpoint(model, toks["base"], sub, "base", cfg["models"][cfg["run"]["size"]]["base"], cfg, w,
                       bos=uses_bos("base", cfg), dtype=cfg["run"]["dtype"], every=EVERY)
        df = read_parts(w.out_dir)
    print(f"smoke: {len(df)} rows for {len(ids)} prefixes in {perf_counter() - t0:.1f}s; bos={set(df['bos'])}, "
          f"format={set(df['input_format'])}, model={set(df['model'])}, non-finite logp={int((~np.isfinite(df['logp'])).sum())}")
    assert set(df["bos"]) == {True} and len(df) == len(sub)
    del model
    release()


def run_job(key: str, repo: str, out_dir: Path, bos: bool, pairs: list[Pair], cfg: dict) -> None:
    """Score one checkpoint into `out_dir` (resumable). Refuses to continue parts written with another BOS setting."""
    writer = PartWriter(out_dir)
    prior = read_parts(out_dir)
    if len(prior) and set(prior["bos"]) != {bos}:
        sys.exit(f"{out_dir} holds rows with bos={set(prior['bos'])}, expected {bos}: never mix the two sets")
    if len(writer.done) == len(pairs):
        print(f"{key} (bos={bos}): already complete, skipped")
        return
    d = local_dir(repo, cfg["paths"]["models_dir"])
    tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
    assert len(tok) == N_TOKENIZER
    t0 = perf_counter()
    model = load_model(d, cfg["run"]["dtype"], cfg["run"]["device"])
    print(f"\n{key} (bos={bos}): loaded in {perf_counter() - t0:.0f}s, {len(writer.done)}/{len(pairs)} pairs already done")
    show_input(tok, pairs[0], key, cfg, bos)
    t0 = perf_counter()
    with tqdm(total=len(pairs), desc=f"{key}{'' if bos else ' no-BOS'}", unit="pair") as bar:
        stats = run_checkpoint(model, tok, pairs, key, repo, cfg, writer, bos=bos, dtype=cfg["run"]["dtype"], every=EVERY,
                               on_prefix=bar.update)
    print(f"{key}: scored {stats.n_scored}, skipped {stats.n_skipped} (resume), {perf_counter() - t0:.0f}s")
    del model
    release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 prefixes x Base in a temp folder; prints one input")
    args = ap.parse_args()
    cfg = load_config()
    size = cfg["run"]["size"]
    set_seed(cfg["project"]["seed"])
    pairs = build_pairs(load_prefixes(), load_vocabulary())
    if args.smoke:
        return smoke(cfg, pairs)
    res = ROOT / cfg["paths"]["results"]
    for name in (f"scores_{size}.parquet", f"scores_{size}_base_nobos.parquet"):
        if (res / name).exists():
            sys.exit(f"{res / name} exists: this run was already consolidated (never overwritten)")
    n_pref = len({p.prefix_id for p in pairs})
    print(f"{n_pref} prefixes, {len(pairs)} pairs per checkpoint; main run {6 * len(pairs)} rows + {len(pairs)} no-BOS rows")
    for key, repo in cfg["models"][size].items():
        run_job(key, repo, res / f"scores_{size}" / key, uses_bos(key, cfg), pairs, cfg)
    base = cfg["models"][size]["base"]
    run_job("base", base, res / f"scores_{size}_base_nobos" / "base", False, pairs, cfg)   # robustness (d), last
    print("\nRun finished. Next: scripts/04b_gate_and_consolidate.py (checks and consolidation).")


if __name__ == "__main__":
    main()
