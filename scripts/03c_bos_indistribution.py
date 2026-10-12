"""BOS test with a "\n" control, Base only.

The file name ("in-distribution") is kept so that it matches its saved outputs.

For each of the 500 prefixes (Base RAW, batch 1, float16, log-softmax float32): sum of the log-probs of prefix tokens
2..n given the previous ones, (a) without and (b) with a leading <|endoftext|>, and (c) with a leading "\n" (control: one
extra context token without the document-boundary meaning). Token 1 is excluded in all three, so the same tokens are
scored. If Base is more at ease with the BOS, (b) > (a); if the gain is EOD-specific, also (b) > (c).

Outputs: results/bos_control_1.4b.parquet (per prefix, never overwritten),
reports/step_03_tables/bos_control_by_category.csv. The first run (a, b only) is kept in
results/bos_indistribution_1.4b.parquet; this run checks that (a) and (b) reproduce it. Prints a short summary.

Usage: .venv/bin/python scripts/03c_bos_indistribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from probing.config import load_config, local_dir  # noqa: E402
from probing.inputs import BOS_ID  # noqa: E402
from probing.pools import load_prefixes  # noqa: E402
from probing.scoring import load_model, set_seed  # noqa: E402

SIZE = "1.4b"
OUT = ROOT / f"results/bos_control_{SIZE}.parquet"
TAB = ROOT / "reports/step_03_tables/bos_control_by_category.csv"
FIRST_RUN = ROOT / f"results/bos_indistribution_{SIZE}.parquet"


@torch.inference_mode()
def tail_logp(model, ids: list[int], skip: int) -> float:
    """Sum of log P(ids[k] | ids[:k]) for k > skip (skip = index of the first prefix token)."""
    dev = next(model.parameters()).device
    lg = model(input_ids=torch.tensor([ids], device=dev)).logits[0]
    pos = torch.arange(skip, len(ids) - 1, device=dev)  # position k-1 predicts token k, k = skip+1..n-1
    lp = torch.log_softmax(lg[pos].float(), dim=-1)
    tgt = torch.tensor(ids[skip + 1:], device=dev)
    return float(lp.gather(1, tgt.unsqueeze(1)).sum())


def summarise(g: pd.DataFrame) -> dict:
    """Per-token differences (b-a), (b-c), (c-a): mean, median, share > 0 and Wilcoxon p for each contrast."""
    out: dict = dict(n=len(g))
    for name, hi, lo in [("b_a", "logp_with_bos", "logp_no_bos"), ("b_c", "logp_with_bos", "logp_newline"),
                         ("c_a", "logp_newline", "logp_no_bos")]:
        d = (g[hi] - g[lo]) / g["n_scored_tokens"]
        out |= {f"mean_{name}": d.mean(), f"median_{name}": d.median(), f"share_{name}_gt0": (d > 0).mean(),
                f"wilcoxon_p_{name}": wilcoxon(g[hi], g[lo]).pvalue}
    return out


def main() -> None:
    if OUT.exists():
        sys.exit(f"{OUT} exists (never overwritten)")
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    repo = cfg["models"][SIZE]["base"]
    d = local_dir(repo, cfg["paths"]["models_dir"])
    tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
    model = load_model(d, "float16", cfg["run"]["device"])
    nl = tok.encode("\n", add_special_tokens=False)
    assert len(nl) == 1, nl
    rows = []
    for pid, cat, pfx in load_prefixes()[["final_id", "category", "prefix"]].itertuples(index=False):
        ids = tok.encode(pfx, add_special_tokens=False)
        a = tail_logp(model, ids, skip=0)
        b = tail_logp(model, [BOS_ID] + ids, skip=1)
        c = tail_logp(model, nl + ids, skip=1)
        rows.append(dict(prefix_id=pid, category=cat, model="base", model_repo=repo, n_scored_tokens=len(ids) - 1,
                         logp_no_bos=a, logp_with_bos=b, logp_newline=c, dtype="float16"))
    df = pd.DataFrame(rows)
    with open(OUT, "xb") as fh:
        df.to_parquet(fh, index=False)
    tab = pd.DataFrame([dict(category=c, **summarise(g)) for c, g in df.groupby("category")]
                       + [dict(category="ALL", **summarise(df))])
    tab.to_csv(TAB, index=False)
    cols = ["logp_no_bos", "logp_with_bos", "logp_newline"]
    print(f"non-finite: {int((~np.isfinite(df[cols].to_numpy())).sum())}")
    if FIRST_RUN.exists():
        old = pd.read_parquet(FIRST_RUN).set_index("prefix_id").loc[df["prefix_id"]]
        for col in cols[:2]:
            print(f"max |{col} - first run|: {np.abs(df[col].to_numpy() - old[col].to_numpy()).max():.2e}")
    print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
