"""Replication across sizes: H1-H3 per size and a descriptive cross-size reading.

Runs only after `12_ingest.py` has passed the reproduction check. The 1.4B point is always the Mac main run
(results/delta_1.4b*.parquet); 2.8b, 6.9b, 12b come from the server working copies (results/delta_<size>*.parquet).
Methods are reused unchanged from 07 (H1), 08 (H2, H2-KTO) and 09 (H3 Shift per category).
Scope per size: H1 full as in 07 (Bonferroni x5 within size); H2 sign/order of the SFT-family gaps above the 0.04-0.06 band only
(p kept in the CSV, not read); H2-KTO and H3 descriptive. Across sizes: descriptive only (n = 4 sizes, no trend test).
Writes:
  reports/step_12_tables/h1_by_size.csv          H1 per size and reference (07 columns + verdict)
  reports/step_12_tables/h2_gaps_by_size.csv     6 SFT-family pairs per size: mean D_{M1,M2} [CI], above the band, sign vs 1.4B
  reports/step_12_tables/h2_order_by_size.csv    order of the four SFT-family models by mean D vs Base, per size
  reports/step_12_tables/h2kto_by_size.csv       Direct KTO - SFT-KTO per size: D, delta_pos, delta_neg (descriptive)
  reports/step_12_tables/h3_shift_by_size.csv    Shift per model x category per size (09 shift_table)
  reports/step_12_tables/h3_family_by_size.csv   SFT-family summary per category per size (09 family_summary)
  reports/step_12_tables/h3_order_rho.csv        Spearman rho of the category order vs 1.4B, per model and family mean (n = 5, no p)
  reports/step_12_tables/scale_D_by_size.csv     mean D vs Base per model x size + difference vs 1.4B (paired bootstrap CI, no p)
  reports/step_12_tables/gate_checks_scale.csv
  results/figures/scale_D_by_size.{png,pdf}
Usage: .venv/bin/python scripts/12_scale.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probing.config import load_config  # noqa: E402
from probing.derived import identity_error  # noqa: E402
from probing.scale import category_order_rho, h1_verdict, load_script  # noqa: E402
from probing.stats import bootstrap_mean_ci  # noqa: E402

TAB = ROOT / "reports/step_12_tables"
SIZES = ["1.4b", "2.8b", "6.9b", "12b"]
PARAMS = {"1.4b": 1.4, "2.8b": 2.8, "6.9b": 6.9, "12b": 12.0}   # billions of parameters (x axis of the figure)
REF = "1.4b"

d07 = load_script(ROOT / "scripts/07_rq1.py")
d08 = load_script(ROOT / "scripts/08_rq2.py")
d09 = load_script(ROOT / "scripts/09_rq3.py")
LABEL, FAMILY, MODELS = d07.LABEL, d08.FAMILY, d09.MODELS
N_BOOT = d07.N_BOOT              # 10,000 resamples for model-level means (as 07/08); Shift per category uses 09's 1,000
CLEARLY_ABOVE = d08.CLEARLY_ABOVE
CAT_COLORS = d09.CAT_COLORS
INK, MUTED = d07.INK, d07.MUTED


def load_size(res: Path, size: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D tables of one size: Mac files for 1.4B (main run), server working copies otherwise."""
    return pd.read_parquet(res / f"delta_{size}.parquet"), pd.read_parquet(res / f"delta_{size}_base_nobos.parquet")


def h3_frame(dm: pd.DataFrame, dn: pd.DataFrame) -> pd.DataFrame:
    """Long table (aligned model x prefix) with y = D (main) and y_nobos, as 09 load_frame without the toxicity columns."""
    main = dm[dm["model"] != "base"][["prefix_id", "category", "model", "D"]].rename(columns={"D": "y"})
    nob = dn[["prefix_id", "model", "D"]].rename(columns={"D": "y_nobos"})
    return main.merge(nob, on=["prefix_id", "model"], how="left", validate="one_to_one").sort_values(["model", "prefix_id"])


def h2_tables(dm: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 6 SFT-family pairs (08 pairwise_table) and Direct KTO - SFT-KTO (08 h2kto_table) for one size."""
    al = dm[dm["model"] != "base"]
    delta, ind = d08.wide(al, "delta"), d08.wide(al, "I_M").astype(bool)
    lp_pos, lp_neg = d08.wide(al, "logp_pos"), d08.wide(al, "logp_neg")
    fried = d08.friedman_table(delta)
    pw = d08.pairwise_table(delta, ind, float(fried["p"].iloc[0]), seed)
    kto = d08.h2kto_table(delta, ind, d08.components(*d08.KTO_PAIR, lp_pos, lp_neg, delta), seed)
    return pw, kto


def scale_table(ds: dict[str, pd.DataFrame], h1: pd.DataFrame, aligned: list[str], seed: int) -> pd.DataFrame:
    """Mean D vs Base (with BOS) per model x size, plus the paired difference vs the 1.4B (Mac) run: mean and bootstrap 95% CI
    over the 500 prefixes (same prefixes in every size). Descriptive, no p."""
    rows = []
    for size in SIZES:
        for m in aligned:
            r = h1[(h1["size"] == size) & (h1["reference"] == "base_bos") & (h1["model"] == m)].iloc[0]
            row = dict(size=size, params_b=PARAMS[size], run="mac" if size == REF else "server", model=m, label=LABEL[m],
                       mean_D=r["mean"], ci_boot_lo=r["ci_boot_lo"], ci_boot_hi=r["ci_boot_hi"], exp_mean=float(np.exp(r["mean"])),
                       d_z=r["d_z"], share_pos=r["share_pos"])
            if size != REF:
                a = ds[REF].loc[ds[REF]["model"] == m].set_index("prefix_id")["D"].sort_index()
                b = ds[size].loc[ds[size]["model"] == m].set_index("prefix_id")["D"].sort_index()
                assert a.index.equals(b.index)
                d = (b - a).to_numpy()
                lo, hi = bootstrap_mean_ci(d, N_BOOT, seed)
                row.update(diff_vs_1_4b=float(d.mean()), diff_ci_lo=lo, diff_ci_hi=hi, share_prefix_higher=float((d > 0).mean()))
            rows.append(row)
    return pd.DataFrame(rows)


def max_diff(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], cols: list[str]) -> float:
    """Max absolute difference of `cols` between two tables matched on `keys` (used to check 1.4B against the tables of 07-09)."""
    m = a[keys + cols].merge(b[keys + cols], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    assert len(m) == len(a) == len(b), (len(m), len(a), len(b))
    return float(max((m[f"{c}_a"] - m[f"{c}_b"]).abs().max() for c in cols))


def figure_scale(sc: pd.DataFrame, aligned: list[str], path: Path) -> None:
    """Mean D vs Base per model across sizes (log x), 95% bootstrap CIs, direct labels at the right end; 1.4B = Mac run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    markers = ["o", "s", "^", "D", "v"]
    dodge = np.linspace(-0.03, 0.03, len(aligned))
    ends = []
    for i, m in enumerate(aligned):
        g = sc[sc["model"] == m].sort_values("params_b")
        x = g["params_b"].to_numpy() * 10 ** dodge[i]
        ax.errorbar(x, g["mean_D"], yerr=[g["mean_D"] - g["ci_boot_lo"], g["ci_boot_hi"] - g["mean_D"]], color=CAT_COLORS[i],
                    marker=markers[i], ms=7, lw=2, elinewidth=1, capsize=3, label=LABEL[m])
        ends.append([float(g["mean_D"].iloc[-1]), LABEL[m]])
    ax.set_ylim(top=float(sc["ci_boot_hi"].max()) + 0.15)        # room for the one-row legend
    ends.sort()
    gap = 0.035 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for j in range(1, len(ends)):                       # keep end labels apart
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + gap)
    for y, lab in ends:
        ax.annotate(lab, (13.6, y), va="center", fontsize=9, color=INK, annotation_clip=False)
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(list(PARAMS.values()), ["1.4B\n(Mac, main run)", "2.8B", "6.9B", "12B"])
    ax.minorticks_off()
    ax.set_xlim(1.2, 13.5)
    ax.set_xlabel("model size (parameters, log scale)")
    ax.set_ylabel("mean D = Δ_M − Δ_Base (nat)")
    ax.set_title("Mean D vs Base per post-trained model across sizes (500 prefixes; 95% bootstrap CI)", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=5)
    d07._style(ax)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    res = ROOT / cfg["paths"]["results"]
    seed = int(cfg["project"]["seed"])
    aligned = list(cfg["models"][REF])[1:]
    labels = cfg["display_labels"]
    gate = pd.read_csv(TAB / "reproduction_gate.csv")
    if not gate["passed"].all():
        sys.exit("reproduction check not passed (reports/step_12_tables/reproduction_gate.csv): run 12_ingest.py first; no analysis")
    (res / "figures").mkdir(exist_ok=True)

    checks = []
    h1s, pws, ktos, shs, fams, ds = [], [], [], [], [], {}
    for size in SIZES:
        dm, dn = load_size(res, size)
        ok = (len(dm) == 3000 and len(dn) == 2500 and identity_error(dm.dropna(subset=["D"])) < 1e-9 and identity_error(dn) < 1e-9
              and dm.loc[dm["model"] != "base", "D"].notna().all())
        checks.append(dict(check=f"{size}: D tables complete (3000 / 2500 rows), identity D = delta_pos - delta_neg < 1e-9", value=ok, pass_=ok))
        ds[size] = dm
        h1 = d07.h1_table(dm, dn, aligned, seed)
        h1["verdict"] = h1_verdict(h1)
        sig = (h1["wilcoxon_p_bonf5"] < 0.05) & ((h1["ci_boot_lo"] > 0) | (h1["ci_boot_hi"] < 0))
        h1["direction"] = np.where(sig, np.where(h1["mean"] > 0, "positive", "negative (reversed)"), "n.s.")
        h1s.append(h1.assign(size=size))
        pw, kto = h2_tables(dm, seed)
        pws.append(pw.assign(size=size))
        ktos.append(kto.assign(size=size))
        sh = d09.shift_table(h3_frame(dm, dn), seed)
        shs.append(sh.assign(size=size))
        fams.append(d09.family_summary(sh).assign(size=size))

    h1 = pd.concat(h1s, ignore_index=True)
    pw = pd.concat(pws, ignore_index=True)
    kto = pd.concat(ktos, ignore_index=True)
    sh = pd.concat(shs, ignore_index=True)
    fam = pd.concat(fams, ignore_index=True)

    # H2: sign / order only. Sign is read where |mean| > 0.06 (above the band); sign agreement vs the 1.4B gap.
    ref_sign = pw[pw["size"] == REF].set_index("pair")["mean"].pipe(np.sign)
    pw["above_band"] = pw["mean"].abs() > CLEARLY_ABOVE
    pw["sign"] = np.where(pw["above_band"], np.sign(pw["mean"]).map({1.0: "+", -1.0: "-"}), "within band")
    pw["same_sign_as_1_4b"] = np.sign(pw["mean"]).to_numpy() == pw["pair"].map(ref_sign).to_numpy()
    h2_cols = ["size", "pair", "m1", "m2", "mean", "ci_boot_lo", "ci_boot_hi", "exp_mean", "d_z", "share_pos", "above_band", "sign",
               "same_sign_as_1_4b", "wilcoxon_p", "wilcoxon_p_bonf6"]
    h2 = pw[h2_cols]
    order = (h1[(h1["reference"] == "base_bos") & h1["model"].isin(FAMILY)].sort_values(["size", "mean"], ascending=[True, False])
             .groupby("size", sort=False)["model"].apply(lambda s: " > ".join(LABEL[m] for m in s)).rename("family_order_by_mean_D")
             .reindex(SIZES).reset_index())
    order["n_gaps_above_band"] = order["size"].map(h2.groupby("size")["above_band"].sum())
    order["n_gaps_same_sign_as_1_4b"] = order["size"].map(h2.groupby("size")["same_sign_as_1_4b"].sum())

    kto_cols = ["size", "quantity", "pair", "mean", "ci_boot_lo", "ci_boot_hi", "exp_mean", "d_z", "share_pos"]
    kto_d = kto[kto_cols]

    # H3: Shift per category; category order vs 1.4B (Spearman, n = 5, no p).
    sh["category_label"] = sh["category"].map(labels)
    fam["category_label"] = fam["category"].map(labels)
    w = sh.pivot_table(index=["size", "category"], columns="model", values="mean", observed=True)
    w["family_mean"] = w[FAMILY].mean(axis=1)
    rho = pd.DataFrame([dict(size=s, series=c, label=LABEL.get(c, "SFT-family mean"),
                             rho_vs_1_4b=category_order_rho(w.loc[REF, c], w.loc[s, c]))
                        for s in SIZES for c in MODELS + ["family_mean"]])

    sc = scale_table(ds, h1, aligned, seed)

    # Checks: the 1.4B rows must equal the main-run tables of 07-09 (same functions, same seed).
    t7 = pd.read_csv(ROOT / "reports/step_07_tables/h1_tests.csv")
    t8 = pd.read_csv(ROOT / "reports/step_08_tables/h2_pairwise.csv")
    t8k = pd.read_csv(ROOT / "reports/step_08_tables/h2kto_tests.csv")
    t9 = pd.read_csv(ROOT / "reports/step_09_tables/h3_shift_by_category.csv")
    for name, a, b, keys, cols in [
            ("1.4B H1 == S7 h1_tests.csv", h1[h1["size"] == REF], t7, ["reference", "model"], ["mean", "ci_boot_lo", "ci_boot_hi", "wilcoxon_p"]),
            ("1.4B H2 pairs == S8 h2_pairwise.csv", pw[pw["size"] == REF], t8, ["pair"], ["mean", "ci_boot_lo", "ci_boot_hi", "wilcoxon_p_bonf6"]),
            ("1.4B H2-KTO == S8 h2kto_tests.csv", kto[kto["size"] == REF], t8k, ["quantity"], ["mean", "ci_boot_lo", "ci_boot_hi"]),
            ("1.4B Shift == S9 h3_shift_by_category.csv", sh[sh["size"] == REF].astype({"model": str}), t9, ["model", "category"],
             ["mean", "ci_boot_lo", "ci_boot_hi", "mean_nobos"])]:
        v = max_diff(a, b, keys, cols)
        checks.append(dict(check=name, value=v, pass_=v < 1e-12))
    v = float((rho.loc[rho["size"] == REF, "rho_vs_1_4b"] - 1).abs().max())
    checks.append(dict(check="rho of 1.4B vs itself = 1", value=v, pass_=v < 1e-12))
    for size in SIZES:
        n = int((h1["size"] == size).sum()), int((pw["size"] == size).sum()), int((sh["size"] == size).sum())
        checks.append(dict(check=f"{size}: rows H1 / H2 pairs / Shift = 10 / 6 / 25", value=str(n), pass_=n == (10, 6, 25)))
    gates = pd.DataFrame(checks).rename(columns={"pass_": "pass"})

    for name, df in [("h1_by_size", h1), ("h2_gaps_by_size", h2), ("h2_order_by_size", order), ("h2kto_by_size", kto_d),
                     ("h3_shift_by_size", sh), ("h3_family_by_size", fam), ("h3_order_rho", rho), ("scale_D_by_size", sc),
                     ("gate_checks_scale", gates)]:
        df.to_csv(TAB / f"{name}.csv", index=False)
    figure_scale(sc, aligned, res / "figures/scale_D_by_size")

    pd.set_option("display.width", 200)
    print(f"checks: {int(gates['pass'].sum())}/{len(gates)} pass")
    if not gates["pass"].all():
        print(gates[~gates["pass"]].to_string(index=False))
    print("\nmean D vs Base (BOS) [diff vs 1.4B Mac]:")
    print(sc.assign(cell=lambda t: t["mean_D"].map("{:+.2f}".format) + t["diff_vs_1_4b"].map(lambda x: "" if pd.isna(x) else f" [{x:+.2f}]"))
          .pivot(index="label", columns="size", values="cell")[SIZES].to_string())
    print("\nH1 direction (BOS | no BOS):")
    v = h1.pivot_table(index=["reference", "model"], columns="size", values="direction", aggfunc="first")[SIZES]
    print(v.to_string())
    print("\nH2 family order and gaps above the band:")
    print(order.to_string(index=False))
    print("\nH3 rho of category order vs 1.4B:")
    print(rho.pivot(index="label", columns="size", values="rho_vs_1_4b")[SIZES].round(2).to_string())


if __name__ == "__main__":
    main()
