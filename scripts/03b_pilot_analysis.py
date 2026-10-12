"""Pilot analysis: reads the pilot scores and run tables, writes reports/step_03_tables/*.csv.

Checks: NaN/inf, plausibility, timing and full-run estimate, BOS sensitivity, float16 vs bfloat16,
mass on the unused vocabulary rows, reproducibility and resume. Prints a short summary (<= 20 lines).

Usage: .venv/bin/python scripts/03b_pilot_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from probing.io import read_parts  # noqa: E402

SIZE = "1.4b"
RES = ROOT / "results"
TAB = ROOT / "reports/step_03_tables"
N_PAIRS_FULL = 4100
ALIGNED = ["sft", "sft_dpo", "sft_ppo", "sft_kto", "kto"]
ORDER = ["base"] + ALIGNED


def read_variant(name: str) -> pd.DataFrame:
    """All parts of a variant folder (main / bosflip / bf16), every model."""
    root = RES / name
    return pd.concat([read_parts(d) for d in sorted(root.iterdir()) if d.is_dir()], ignore_index=True)


def deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Delta(c|S) = mean logP over SAFE words - mean over NOT SAFE words, per prefix and model."""
    m = df.groupby(["model", "prefix_id", "category", "polarity_word"])["logp"].mean().unstack("polarity_word")
    return (m["SAFE"] - m["NOT SAFE"]).rename("delta").reset_index()


def d_vs_base(delta: pd.DataFrame, base_delta: pd.DataFrame) -> pd.DataFrame:
    """D_{M,Base}(s) = Delta_M(s) - Delta_Base(s) for the aligned models."""
    b = base_delta.set_index("prefix_id")["delta"]
    out = delta[delta["model"] != "base"].copy()
    out["D"] = out["delta"] - out["prefix_id"].map(b)
    return out


def nan_check(main, flip, bf, mass) -> pd.DataFrame:
    rows = []
    for name, df in (("main_fp16", main), ("bos_flip_fp16", flip), ("bf16", bf)):
        for m, g in df.groupby("model"):
            rows.append(dict(variant=name, model=m, n_rows=len(g), nonfinite_logp=int((~np.isfinite(g["logp"])).sum()),
                             positive_logp=int((g["logp"] > 0).sum())))
    nf = mass.groupby("model")["nonfinite_logits"].sum()
    for m, v in nf.items():
        rows.append(dict(variant="full_logits_fp16", model=m, n_rows=int((mass["model"] == m).sum()),
                         nonfinite_logp=int(v), positive_logp=0))
    return pd.DataFrame(rows)


def plausibility(main: pd.DataFrame, delta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Means by model x category x polarity, extremes, and the share neutral > derogatory."""
    by = main.groupby(["model", "category", "polarity_word"])["logp"].mean().unstack("polarity_word")
    allc = main.groupby(["model", "polarity_word"])["logp"].mean().unstack("polarity_word")
    allc["category"] = "ALL"
    tab = pd.concat([by.reset_index(), allc.reset_index()], ignore_index=True)
    tab = tab.rename(columns={"SAFE": "mean_logp_neutral", "NOT SAFE": "mean_logp_derog"})
    tab["neutral_minus_derog"] = tab["mean_logp_neutral"] - tab["mean_logp_derog"]
    ext = []
    for m, g in main.groupby("model"):
        for kind, r in (("min", g.loc[g["logp"].idxmin()]), ("max", g.loc[g["logp"].idxmax()])):
            ext.append(dict(model=m, extreme=kind, logp=r["logp"], word=r["word"], polarity_word=r["polarity_word"],
                            n_tokens=r["n_tokens"], prefix_id=r["prefix_id"]))
    share = []
    for (m, pid), g in main.groupby(["model", "prefix_id"]):
        pos = g.loc[g["polarity_word"] == "SAFE", "logp"].to_numpy()
        neg = g.loc[g["polarity_word"] == "NOT SAFE", "logp"].to_numpy()
        share.append(dict(model=m, prefix_id=pid, n_cmp=pos.size * neg.size,
                          n_neutral_higher=int((pos[:, None] > neg[None, :]).sum())))
    sh = pd.DataFrame(share).groupby("model")[["n_cmp", "n_neutral_higher"]].sum()
    sh["share_neutral_higher"] = sh["n_neutral_higher"] / sh["n_cmp"]
    sh["share_prefixes_delta_pos"] = delta.groupby("model")["delta"].apply(lambda s: (s > 0).mean())
    sh["mean_delta"] = delta.groupby("model")["delta"].mean()
    return tab, pd.DataFrame(ext), sh.reset_index()


def timing() -> tuple[pd.DataFrame, float]:
    t = pd.read_csv(TAB / "timing_per_prefix.csv")
    loads = pd.read_csv(TAB / "load_times.csv")
    g = t.groupby("model").agg(pairs=("n_pairs", "sum"), seconds=("seconds", "sum"), prefixes=("prefix_id", "count"))
    g["sec_per_pair"] = g["seconds"] / g["pairs"]
    g["sec_per_prefix"] = g["seconds"] / g["prefixes"]
    g["load_seconds_fp16"] = loads[loads["dtype"] == "float16"].set_index("model")["load_seconds"]
    g["s4_minutes"] = (g["load_seconds_fp16"] + N_PAIRS_FULL * g["sec_per_pair"]) / 60
    return g.reindex(ORDER).reset_index(), float(g["s4_minutes"].sum())


def compare(main: pd.DataFrame, other: pd.DataFrame, label: str, base_delta_main: pd.DataFrame,
            base_delta_other: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Change in logP and in Delta per model, and the change in D compared with the size of D."""
    k = ["prefix_id", "model", "word"]
    j = main.merge(other[k + ["logp"]], on=k, suffixes=("", "_v"))
    j["dlogp"] = j["logp_v"] - j["logp"]
    dm, dv = deltas(main), deltas(other)
    dd = dm.merge(dv[["model", "prefix_id", "delta"]], on=["model", "prefix_id"], suffixes=("", "_v"))
    dd["ddelta"] = dd["delta_v"] - dd["delta"]
    Dm = d_vs_base(dm, base_delta_main).set_index(["model", "prefix_id"])["D"]
    rows = []
    for m in [x for x in ORDER if x in set(j["model"])]:
        jm, dm_ = j[j["model"] == m], dd[dd["model"] == m]
        r = dict(variant=label, model=m, n_pairs=len(jm), mean_dlogp=jm["dlogp"].mean(),
                 median_abs_dlogp=jm["dlogp"].abs().median(), max_abs_dlogp=jm["dlogp"].abs().max(),
                 mean_ddelta=dm_["ddelta"].mean(), median_abs_ddelta=dm_["ddelta"].abs().median(),
                 max_abs_ddelta=dm_["ddelta"].abs().max())
        rows.append(r)
    out = pd.DataFrame(rows)
    # effect on D for every aligned model: variant Delta replaces main Delta on the changed side
    eff = []
    for m in ALIGNED:
        D = Dm.loc[m]
        if label.startswith("bos") or label == "bf16":
            dv_m = dv[dv["model"] == m].set_index("prefix_id")["delta"] if m in set(dv["model"]) else None
            if dv_m is not None:
                Dv = dv_m - base_delta_main.set_index("prefix_id")["delta"]
                eff.append(dict(variant=label, changed_side=m, model=m, mean_D=D.mean(), median_abs_D=D.abs().median(),
                                mean_D_variant=Dv.mean(), mean_change_D=(Dv - D).mean(),
                                median_abs_change_D=(Dv - D).abs().median(),
                                sign_agreement=float((np.sign(Dv) == np.sign(D)).mean())))
            if base_delta_other is not None:
                Dv = dm[dm["model"] == m].set_index("prefix_id")["delta"] - base_delta_other.set_index("prefix_id")["delta"]
                eff.append(dict(variant=label, changed_side="base", model=m, mean_D=D.mean(),
                                median_abs_D=D.abs().median(), mean_D_variant=Dv.mean(),
                                mean_change_D=(Dv - D).mean(), median_abs_change_D=(Dv - D).abs().median(),
                                sign_agreement=float((np.sign(Dv) == np.sign(D)).mean())))
    return out, pd.DataFrame(eff)


def main() -> None:
    main_df = pd.read_parquet(RES / f"scores_pilot_{SIZE}.parquet")
    flip = read_variant(f"scores_pilot_{SIZE}_bosflip")
    bf = read_variant(f"scores_pilot_{SIZE}_bf16")
    mass = pd.read_csv(TAB / "unused_rows_mass.csv")
    assert len(main_df) == 164 * 6 and not main_df.duplicated(["prefix_id", "model", "word"]).any()

    nan = nan_check(main_df, flip, bf, mass)
    nan.to_csv(TAB / "nan_inf_check.csv", index=False)

    delta = deltas(main_df)
    base_delta = delta[delta["model"] == "base"]
    plaus, ext, share = plausibility(main_df, delta)
    plaus.to_csv(TAB / "plausibility_by_category.csv", index=False)
    ext.to_csv(TAB / "extremes.csv", index=False)
    share.to_csv(TAB / "neutral_vs_derog_share.csv", index=False)
    delta.merge(d_vs_base(delta, base_delta)[["model", "prefix_id", "D"]], how="left",
                on=["model", "prefix_id"]).to_csv(TAB / "delta_per_prefix.csv", index=False)
    by_ntok = main_df.groupby(["model", "polarity_word", "n_tokens"])["logp"].agg(["mean", "count"]).reset_index()
    by_ntok.to_csv(TAB / "logp_by_ntokens.csv", index=False)

    tim, s4 = timing()
    tim.to_csv(TAB / "timing_summary.csv", index=False)

    base_flip_delta = deltas(flip)
    base_flip_delta = base_flip_delta[base_flip_delta["model"] == "base"]
    bos_tab, bos_eff = compare(main_df, flip, "bos_flip", base_delta, base_flip_delta)
    bos_tab.to_csv(TAB / "bos_sensitivity_logp_delta.csv", index=False)
    bos_eff.to_csv(TAB / "bos_sensitivity_D.csv", index=False)

    bf_tab, bf_eff = compare(main_df, bf, "bf16", base_delta, None)
    bf_tab.to_csv(TAB / "fp16_vs_bf16_logp_delta.csv", index=False)
    bf_eff.to_csv(TAB / "fp16_vs_bf16_D.csv", index=False)

    um = mass.groupby("model").agg(mass_mean=("mass_mean", "mean"), mass_max=("mass_max", "max"),
                                   max_row_prob=("max_row_prob", "max"),
                                   logp_shift_mean=("logp_shift_if_excluded", "mean"),
                                   logp_shift_max=("logp_shift_if_excluded", "max")).reindex(ORDER).reset_index()
    um.to_csv(TAB / "unused_rows_mass_summary.csv", index=False)

    rep = pd.read_csv(TAB / "repro_check.csv")
    res = pd.read_csv(TAB / "resume_check.csv")

    print(f"non-finite logP total: {int(nan['nonfinite_logp'].sum())}")
    print(share[["model", "share_neutral_higher", "share_prefixes_delta_pos", "mean_delta"]].round(3).to_string(index=False))
    print(tim[["model", "sec_per_pair", "load_seconds_fp16", "s4_minutes"]].round(3).to_string(index=False))
    print(f"full-run estimate (6 checkpoints, batch 1, loading included): {s4:.1f} min")
    print(bos_eff.groupby("changed_side")[["median_abs_change_D", "median_abs_D"]].median().round(3).to_string())
    print(f"bf16 max |dlogp| {bf_tab['max_abs_dlogp'].max():.3f}; repro bit-identical "
          f"{int(rep['bit_identical'].sum())}/{len(rep)}; third session scored {int(res['session3_scored'].sum())}")
    print(f"unused rows mass max {um['mass_max'].max():.2e}")


if __name__ == "__main__":
    main()
