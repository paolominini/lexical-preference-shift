"""Pilot run: 20 prefixes x 6 checkpoints (164 pairs each), batch size 1, SDPA, float16.

Per checkpoint (one model in memory at a time):
  main run with a simulated interruption + resume (same writer as the full run),
  reproducibility (10 pairs re-scored, must be bit-identical), BOS-flipped run (sensitivity to the first token),
  probability mass on the unused vocabulary rows at the word positions, and for the aligned checkpoints a
  bfloat16 run (the dtype their weights are stored in).
Outputs: results/scores_pilot_1.4b.parquet (+ part folders), reports/step_03_tables/*.csv. Analysis: 03b.

Usage: .venv/bin/python scripts/03_pilot.py
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from probing.config import is_aligned, load_config, local_dir  # noqa: E402
from probing.inputs import Encoded, context_text, encode_pair  # noqa: E402
from probing.io import PartWriter, read_parts  # noqa: E402
from probing.pools import Pair, build_pairs, load_prefixes, load_vocabulary  # noqa: E402
from probing.run import input_format, run_checkpoint  # noqa: E402
from probing.scoring import load_model, score, set_seed, target_positions  # noqa: E402

SIZE = "1.4b"
N_PER_POLARITY = 2           # per category: 2 SAFE-sourced + 2 NOT SAFE-sourced prefixes
INTERRUPT_AFTER = 8          # simulated interruption after 8 of 20 prefixes
N_REPRO = 10
N_TOKENIZER = 50277          # rows 50,277..50,303 of the 50,304 output rows are unused (tokenizer has 50,277)
RES = ROOT / "results"
TAB = ROOT / "reports/step_03_tables"
FINAL = RES / f"scores_pilot_{SIZE}.parquet"


def select_prefixes(seed: int) -> pd.DataFrame:
    """2 SAFE + 2 NOT SAFE-sourced prefixes per category, seeded, ids sorted inside each draw."""
    df = load_prefixes()
    rng = np.random.default_rng(seed)
    pick = []
    for cat in sorted(df["category"].unique()):
        for pol in ("SAFE", "NOT SAFE"):
            ids = sorted(df.loc[(df["category"] == cat) & (df["polarity"] == pol), "final_id"])
            pick += list(rng.choice(ids, N_PER_POLARITY, replace=False))
    return df[df["final_id"].isin(pick)].sort_values("final_id").reset_index(drop=True)


@torch.inference_mode()
def unused_mass(model, enc: Encoded) -> dict:
    """Softmax mass on the unused rows at each word position, and non-finite logits in the whole sequence."""
    dev = next(model.parameters()).device
    lg = model(input_ids=torch.tensor([enc.ids], device=dev)).logits[0].float()
    pos, _ = target_positions(enc)
    p = torch.softmax(lg[pos], dim=-1)
    mass = p[:, N_TOKENIZER:].sum(-1)
    return dict(mass_mean=float(mass.mean()), mass_max=float(mass.max()),
                max_row_prob=float(p[:, N_TOKENIZER:].max()),
                logp_shift_if_excluded=float(-torch.log1p(-mass.cpu().double()).sum()),
                nonfinite_logits=int((~torch.isfinite(lg)).sum()))


def release() -> None:
    """Return freed memory to the system. The caller must `del` its model first: only one model may be in memory."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main_run(model, tok, pairs: list[Pair], key: str, repo: str, cfg: dict) -> tuple[dict, list[dict]]:
    """Main condition with a simulated interruption, resume, and a no-op third session.

    If parts already exist (a previous process was stopped), the run resumes instead, and the timing comes from
    re-scoring all pairs in memory (nothing written), which is also a full bit-identity check.
    """
    writer = PartWriter(RES / f"scores_pilot_{SIZE}" / key)
    bos = is_aligned(key)
    if writer.done:
        prior = len(writer.done)
        b = run_checkpoint(model, tok, pairs, key, repo, cfg, writer, bos=bos, dtype="float16", every=5)
        c = run_checkpoint(model, tok, pairs, key, repo, cfg, PartWriter(writer.out_dir), bos=bos, dtype="float16")
        stored = read_parts(writer.out_dir).set_index(["prefix_id", "word"])["logp"]
        fmt, timing, n_diff = input_format(key, cfg), [], 0
        for pid in dict.fromkeys(p.prefix_id for p in pairs):
            grp = [p for p in pairs if p.prefix_id == pid]
            encs = [encode_pair(tok, context_text(grp[0].prefix, fmt, cfg["chat"]), p.word, bos=bos) for p in grp]
            t0 = perf_counter()
            new = score(model, encs, batch_size=cfg["run"]["batch_size"])
            timing.append(dict(model=key, prefix_id=pid, n_pairs=len(grp), seconds=perf_counter() - t0))
            n_diff += sum(ws.logp != float(stored.loc[(pid, p.word)]) for p, ws in zip(grp, new))
        resume = dict(model=key, session1_scored=prior, session2_scored=b.n_scored, session2_skipped=b.n_skipped,
                      session3_scored=c.n_scored, session3_skipped=c.n_skipped, overlap_s1_s2=0,
                      note=f"resumed after a real stop; full re-score differs on {n_diff}/{len(pairs)} pairs")
        return resume, timing
    a = run_checkpoint(model, tok, pairs, key, repo, cfg, writer, bos=bos, dtype="float16", every=5,
                       max_prefixes=INTERRUPT_AFTER)
    b = run_checkpoint(model, tok, pairs, key, repo, cfg, PartWriter(writer.out_dir), bos=bos, dtype="float16",
                       every=5)
    c = run_checkpoint(model, tok, pairs, key, repo, cfg, PartWriter(writer.out_dir), bos=bos, dtype="float16")
    resume = dict(model=key, session1_scored=a.n_scored, session2_scored=b.n_scored, session2_skipped=b.n_skipped,
                  session3_scored=c.n_scored, session3_skipped=c.n_skipped,
                  overlap_s1_s2=len(set(a.prefix_seconds) & set(b.prefix_seconds)))
    timing = [dict(model=key, prefix_id=pid, n_pairs=s.prefix_pairs[pid], seconds=s.prefix_seconds[pid])
              for s in (a, b) for pid in s.prefix_seconds]
    return resume, timing


def repro(model, tok, pairs: list[Pair], key: str, cfg: dict, seed: int) -> list[dict]:
    """Re-score 10 random pairs in reverse order; compare with the stored scores bit for bit."""
    stored = read_parts(RES / f"scores_pilot_{SIZE}" / key).set_index(["prefix_id", "word"])["logp"]
    idx = np.random.default_rng(seed).choice(len(pairs), N_REPRO, replace=False)[::-1]
    fmt = input_format(key, cfg)
    out = []
    for i in idx:
        p = pairs[i]
        enc = encode_pair(tok, context_text(p.prefix, fmt, cfg["chat"]), p.word, bos=is_aligned(key))
        new = score(model, [enc], batch_size=1)[0].logp
        old = float(stored.loc[(p.prefix_id, p.word)])
        out.append(dict(model=key, prefix_id=p.prefix_id, word_index=i, logp_stored=old, logp_rerun=new,
                        bit_identical=new == old))
    return out


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)
    if FINAL.exists():
        sys.exit(f"{FINAL} exists: the pilot was already run (never overwritten). Run 03b for the analysis.")
    TAB.mkdir(parents=True, exist_ok=True)
    sel = select_prefixes(seed)
    sel[["final_id", "category", "polarity", "corpus", "edited"]].to_csv(TAB / "pilot_prefixes.csv", index=False)
    pairs = build_pairs(sel, load_vocabulary())
    print(f"{len(sel)} prefixes, {len(pairs)} pairs per checkpoint")

    resume, timing, rep, mass, loads = [], [], [], [], []
    for key, repo in cfg["models"][SIZE].items():
        d = local_dir(repo, cfg["paths"]["models_dir"])
        tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
        assert len(tok) == N_TOKENIZER
        t0 = perf_counter()
        model = load_model(d, "float16", cfg["run"]["device"])
        loads.append(dict(model=key, dtype="float16", load_seconds=perf_counter() - t0))
        r, t = main_run(model, tok, pairs, key, repo, cfg)
        resume.append(r)
        timing += t
        rep += repro(model, tok, pairs, key, cfg, seed)
        fmt = input_format(key, cfg)
        # sensitivity: BOS flipped (aligned without, Base with)
        flip = PartWriter(RES / f"scores_pilot_{SIZE}_bosflip" / key)
        run_checkpoint(model, tok, pairs, key, repo, cfg, flip, bos=not is_aligned(key), dtype="float16")
        for p in pairs:
            enc = encode_pair(tok, context_text(p.prefix, fmt, cfg["chat"]), p.word, bos=is_aligned(key))
            mass.append(dict(model=key, prefix_id=p.prefix_id, word=p.word, polarity_word=p.polarity_word,
                             **unused_mass(model, enc)))
        del model
        release()
        if is_aligned(key):  # bfloat16 (the dtype the Archangel weights are stored in)
            t0 = perf_counter()
            model = load_model(d, "bfloat16", cfg["run"]["device"])
            loads.append(dict(model=key, dtype="bfloat16", load_seconds=perf_counter() - t0))
            bf = PartWriter(RES / f"scores_pilot_{SIZE}_bf16" / key)
            run_checkpoint(model, tok, pairs, key, repo, cfg, bf, bos=True, dtype="bfloat16")
            del model
            release()
        print(f"{key}: done (resume {r['session1_scored']}+{r['session2_scored']}, "
              f"third session scored {r['session3_scored']})")

    main_df = pd.concat([read_parts(p) for p in sorted((RES / f"scores_pilot_{SIZE}").iterdir())],
                        ignore_index=True)
    with open(FINAL, "xb") as fh:
        main_df.to_parquet(fh, index=False)
    pd.DataFrame(resume).to_csv(TAB / "resume_check.csv", index=False)
    pd.DataFrame(timing).to_csv(TAB / "timing_per_prefix.csv", index=False)
    pd.DataFrame(loads).to_csv(TAB / "load_times.csv", index=False)
    pd.DataFrame(rep).to_csv(TAB / "repro_check.csv", index=False)
    pd.DataFrame(mass).to_csv(TAB / "unused_rows_mass.csv", index=False)
    print(f"wrote {FINAL.relative_to(ROOT)} ({len(main_df)} rows) and run tables in {TAB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
