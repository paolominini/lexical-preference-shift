"""Checks on the parts of the full run: check the parts, re-score random rows with an independent path, then consolidate.

Checks (per set: main = six checkpoints with BOS, nobos = Base without BOS, never mixed):
  row counts, exact match with the expected (prefix, model, word) pairs, no duplicates, 0 non-finite, logp <= 0,
  bos / input_format / dtype / model_repo columns, n_tokens identical across models and prefixes for the same word.
Spot check: 5 random rows of the main set + 1 of the no-BOS set (seeded) are re-scored from scratch, (i) with `score`
(same code as the run) and (ii) by a manual path (no attention mask, full log-softmax, Python sum); (i) should be bit-identical.
A spot-check mismatch only warns (one extra row in gate_checks.csv): GPU kernels are not always bit-reproducible.
If the structural checks pass, the parts are consolidated into results/scores_<size>.parquet and
results/scores_<size>_base_nobos.parquet (written with mode 'xb': never overwrites). Tables: reports/step_04_tables/.
The word itself is never printed or saved in the tables.

Usage: .venv/bin/python scripts/04b_gate_and_consolidate.py
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from probing.config import load_config, local_dir, uses_bos  # noqa: E402
from probing.inputs import context_text, encode_pair  # noqa: E402
from probing.io import KEY, read_parts  # noqa: E402
from probing.pools import build_pairs, load_prefixes, load_vocabulary  # noqa: E402
from probing.run import input_format  # noqa: E402
from probing.scoring import load_model, score, target_positions  # noqa: E402

TAB = ROOT / "reports/step_04_tables"
N_SPOT = 5


def expected_frame(pairs, models: list[str]) -> pd.DataFrame:
    return pd.DataFrame([(p.prefix_id, m, p.word) for m in models for p in pairs], columns=KEY)


def check_set(name: str, df: pd.DataFrame, pairs, models: list[str], bos: bool, cfg: dict) -> list[dict]:
    """All structural checks for one set; returns rows (check, result, detail)."""
    exp = expected_frame(pairs, models)
    out = []

    def add(check: str, ok: bool, detail) -> None:
        out.append(dict(set=name, check=check, passed=bool(ok), detail=detail))

    add("row count", len(df) == len(exp), f"{len(df)} (expected {len(exp)})")
    add("no duplicate (prefix_id, model, word)", not df.duplicated(KEY).any(), int(df.duplicated(KEY).sum()))
    merged = exp.merge(df[KEY].drop_duplicates(), on=KEY, how="outer", indicator=True)
    add("no missing pair", (merged["_merge"] != "left_only").all(), int((merged["_merge"] == "left_only").sum()))
    add("no unexpected pair", (merged["_merge"] != "right_only").all(), int((merged["_merge"] == "right_only").sum()))
    add("non-finite logp = 0", np.isfinite(df["logp"]).all(), int((~np.isfinite(df["logp"])).sum()))
    add("logp <= 0", (df["logp"] <= 0).all(), int((df["logp"] > 0).sum()))
    add("bos column", (df["bos"] == bos).all(), f"all {bos}")
    add("dtype column", (df["dtype"] == cfg["run"]["dtype"]).all(), f"all {cfg['run']['dtype']}")
    fmt_ok = all((df.loc[df["model"] == m, "input_format"] == input_format(m, cfg)).all() for m in models)
    add("input_format per model (base raw, aligned chat)", fmt_ok, "")
    repo_ok = all((df.loc[df["model"] == m, "model_repo"] == cfg["models"][cfg["run"]["size"]][m]).all() for m in models)
    add("model_repo per model", repo_ok, "")
    nt = df.groupby("word")["n_tokens"].nunique()
    add("n_tokens constant per word (all models, prefixes)", (nt == 1).all(), int((nt != 1).sum()))
    add("n_tokens >= 1", (df["n_tokens"] >= 1).all(), int(df["n_tokens"].min()))
    return out


def manual_logp(model, enc) -> float:
    """Independent path: no attention mask, full-sequence logits, float32 log-softmax, Python sum."""
    dev = next(model.parameters()).device
    with torch.inference_mode():
        lg = model(input_ids=torch.tensor([enc.ids], device=dev)).logits[0].float()
    pos, tgt = target_positions(enc)
    lp = torch.log_softmax(lg, dim=-1).cpu()
    return float(sum(float(lp[p, t]) for p, t in zip(pos, tgt)))


def spot_check(df_main: pd.DataFrame, df_nobos: pd.DataFrame, pairs, cfg: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = [("main", df_main.iloc[i]) for i in rng.choice(len(df_main), N_SPOT, replace=False)]
    pick.append(("nobos", df_nobos.iloc[int(rng.integers(len(df_nobos)))]))
    lookup = {(p.prefix_id, p.word): p for p in pairs}
    rows = []
    for key in dict.fromkeys(r["model"] for _, r in pick):
        d = local_dir(cfg["models"][cfg["run"]["size"]][key], cfg["paths"]["models_dir"])
        tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
        model = load_model(d, cfg["run"]["dtype"], cfg["run"]["device"])
        for setname, r in pick:
            if r["model"] != key:
                continue
            p = lookup[(r["prefix_id"], r["word"])]
            enc = encode_pair(tok, context_text(p.prefix, input_format(key, cfg), cfg["chat"]), p.word, bos=bool(r["bos"]))
            new = score(model, [enc], batch_size=1)[0].logp
            man = manual_logp(model, enc)
            rows.append(dict(set=setname, prefix_id=r["prefix_id"], model=key, polarity_word=r["polarity_word"],
                             n_tokens=int(r["n_tokens"]), bos=bool(r["bos"]), logp_stored=r["logp"], logp_rerun=new,
                             bit_identical=new == r["logp"], logp_manual=man, manual_abs_diff=abs(man - r["logp"])))
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    size = cfg["run"]["size"]
    res = ROOT / cfg["paths"]["results"]
    final_main, final_nobos = res / f"scores_{size}.parquet", res / f"scores_{size}_base_nobos.parquet"
    if final_main.exists() or final_nobos.exists():
        sys.exit("a consolidated file exists (never overwritten)")
    pairs = build_pairs(load_prefixes(), load_vocabulary())
    models = list(cfg["models"][size])
    df_main = pd.concat([read_parts(res / f"scores_{size}" / m) for m in models], ignore_index=True)
    df_nobos = read_parts(res / f"scores_{size}_base_nobos" / "base")
    rows = check_set("main", df_main, pairs, models, True, cfg) + check_set("nobos", df_nobos, pairs, ["base"], False, cfg)
    # same words, same tokens: Base with and without BOS must tokenize the words identically
    a = df_main[df_main["model"] == "base"].set_index(KEY[0::2])["n_tokens"]
    b = df_nobos.set_index(KEY[0::2])["n_tokens"]
    rows.append(dict(set="both", check="n_tokens Base BOS == Base no-BOS", passed=bool((a.sort_index() == b.sort_index()).all()), detail=""))
    TAB.mkdir(parents=True, exist_ok=True)
    chk = pd.DataFrame(rows)
    chk.to_csv(TAB / "gate_checks.csv", index=False)
    print(chk[["set", "check", "passed", "detail"]].to_string(index=False))
    spot = spot_check(df_main, df_nobos, pairs, cfg, cfg["project"]["seed"])
    spot.to_csv(TAB / "spot_check.csv", index=False)
    print(spot.drop(columns=["logp_stored", "logp_rerun", "logp_manual"]).to_string(index=False))
    ok = chk["passed"].all()          # structural checks block; the spot check only warns (GPU kernels may not be bit-reproducible)
    print(f"\nall structural checks passed: {chk['passed'].all()}; spot check bit-identical: {spot['bit_identical'].all()}; "
          f"manual path max abs diff {spot['manual_abs_diff'].max():.2e}")
    if not spot["bit_identical"].all():
        diff = float((spot["logp_rerun"] - spot["logp_stored"]).abs().max())
        warn = dict(set="main+nobos", check="spot check bit-identical (non-blocking)", passed=False,
                    detail=f"warning: max abs diff {diff:.2e}")
        pd.concat([chk, pd.DataFrame([warn])], ignore_index=True).to_csv(TAB / "gate_checks.csv", index=False)
        print(f"\nWARNING: re-scored rows are not bit-identical to the run (max abs diff {diff:.2e}). "
              "This is not an error: consolidating anyway; the difference is recorded in gate_checks.csv.")
    if not ok:
        sys.exit("checks NOT passed: nothing consolidated")
    for df, path in ((df_main, final_main), (df_nobos, final_nobos)):
        with open(path, "xb") as fh:
            df.sort_values(KEY).reset_index(drop=True).to_parquet(fh, index=False)
        print(f"wrote {path.relative_to(ROOT)} ({len(df)} rows)")


if __name__ == "__main__":
    main()
