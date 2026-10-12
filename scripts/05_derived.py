"""Derived quantities (descriptive only; tests and CIs are in 07-09).

Reads results/scores_<size>.parquet (main, Base with BOS) and results/scores_<size>_base_nobos.parquet (robustness d);
<size> = config `run.size` (main run 1.4b).
Writes, never overwriting:
  results/delta_<size>.parquet            per (model, prefix): delta, delta_pos, delta_neg, D, I_M (six models; Base is the reference)
  results/delta_<size>_base_nobos.parquet same for the five aligned models against Base without BOS (never mixed with the above)
  reports/step_05_tables/*.csv          summary, D main vs no-BOS, per-word shift, probability ratios, drop-one-word (Transgender Woman),
                                        identity/coherence checks, worked example.
Usage: .venv/bin/python scripts/05_derived.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probing.config import load_config  # noqa: E402
from probing.derived import KEYS, NEG, POS, derived_vs_base, identity_error, pair_differences, pool_means  # noqa: E402

TAB = ROOT / "reports/step_05_tables"
TW = "Transgender Woman"


def build_delta_main(scores: pd.DataFrame) -> pd.DataFrame:
    """Six models x 500 prefixes. Base keeps delta and I_M only (D, delta_pos, delta_neg are relative to Base: NaN)."""
    base = scores[scores["model"] == "base"]
    pm = pool_means(scores)
    dv = derived_vs_base(scores, base)[KEYS + ["delta_pos", "delta_neg", "D"]]
    out = pm.merge(dv, on=KEYS, how="left", validate="one_to_one")
    out["I_M"] = out["delta"] > 0
    cols = ["prefix_id", "category", "model", "delta", "delta_pos", "delta_neg", "D", "I_M", "logp_pos", "logp_neg"]
    return out[cols].sort_values(["model", "prefix_id"]).reset_index(drop=True)


def build_delta_nobos(scores: pd.DataFrame, base_nobos: pd.DataFrame) -> pd.DataFrame:
    """Five aligned models vs Base scored without BOS. `delta_ref` is Delta of Base without BOS."""
    out = derived_vs_base(scores, base_nobos)
    cols = ["prefix_id", "category", "model", "delta", "delta_ref", "delta_pos", "delta_neg", "D"]
    return out[cols].sort_values(["model", "prefix_id"]).reset_index(drop=True)


def write_new(df: pd.DataFrame, path: Path) -> None:
    """Never overwrite results silently. An existing file is kept, after checking that it equals the recomputed one."""
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), df, check_exact=False, rtol=1e-12, atol=1e-12)
        print(f"{path.name} already exists and equals the recomputed table: kept")
        return
    df.to_parquet(path, index=False)


def summary_table(dm: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Model x category (and ALL): means of Delta, delta+, delta-, D (= Shift_M(c)), shares I_M = 1 and D > 0."""
    rows = []
    for model in order:
        g_model = dm[dm["model"] == model]
        for cat, g in [("ALL", g_model)] + list(g_model.groupby("category")):
            rows.append(dict(model=model, category=cat, n_prefixes=len(g), mean_delta=g["delta"].mean(),
                             mean_delta_pos=g["delta_pos"].mean(), mean_delta_neg=g["delta_neg"].mean(),
                             mean_D=g["D"].mean(), share_I_M=g["I_M"].mean(),
                             share_D_pos=(g["D"] > 0).mean() if model != "base" else np.nan))
    return pd.DataFrame(rows)


def d_side_by_side(dm: pd.DataFrame, dn: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rows = []
    for model in order[1:]:
        for cat in ["ALL"] + sorted(dm["category"].unique()):
            a = dm[(dm["model"] == model) & ((dm["category"] == cat) | (cat == "ALL"))]
            b = dn[(dn["model"] == model) & ((dn["category"] == cat) | (cat == "ALL"))]
            rows.append(dict(model=model, category=cat, mean_D_main=a["D"].mean(), mean_D_nobos=b["D"].mean(),
                             nobos_minus_main=b["D"].mean() - a["D"].mean(),
                             share_D_pos_main=(a["D"] > 0).mean(), share_D_pos_nobos=(b["D"] > 0).mean()))
    return pd.DataFrame(rows)


def per_word_table(scores: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Mean (logP_M - logP_Base) by model, category, word, plus the word's additive contribution to mean D.

    D(s) = mean_pos diff - mean_neg diff, so each word contributes +diff/n_pos (neutral) or -diff/n_neg (derogatory);
    the contributions of the words of a pool sum to the pool's mean shift, and over the whole pool to mean D.
    """
    d = pair_differences(scores, scores[scores["model"] == "base"])
    g = d.groupby(["model", "category", "polarity_word", "word"]).agg(
        n_tokens=("n_tokens", "first"), mean_logp_model=("logp", "mean"), mean_logp_base=("logp_base", "mean"),
        mean_shift=("diff", "mean"), share_shift_pos=("diff", lambda x: (x > 0).mean())).reset_index()
    n_pool = g.groupby(["model", "category", "polarity_word"])["word"].transform("count")
    g["contribution_to_D"] = np.where(g["polarity_word"] == POS, 1, -1) * g["mean_shift"] / n_pool
    g["model"] = pd.Categorical(g["model"], order[1:], ordered=True)
    return g.sort_values(["model", "category", "polarity_word", "word"], ascending=[True, True, False, True]).reset_index(drop=True)


def ratio_table(summary: pd.DataFrame, d_both: pd.DataFrame) -> pd.DataFrame:
    """Mean Delta and mean D also as e^x (exp of the mean, i.e. a geometric-type ratio)."""
    r = summary[["model", "category", "mean_delta", "mean_D"]].copy()
    r["ratio_delta"] = np.exp(r["mean_delta"])
    r["ratio_D"] = np.exp(r["mean_D"])
    nb = d_both[["model", "category", "mean_D_nobos"]].copy()
    nb["ratio_D_nobos"] = np.exp(nb["mean_D_nobos"])
    return r.merge(nb, on=["model", "category"], how="left")


def drop_one_word(scores: pd.DataFrame, category: str, order: list[str]) -> pd.DataFrame:
    """Mean D of `category` recomputed with each probe word removed in turn (descriptive preview of robustness check c, one category only)."""
    d = pair_differences(scores, scores[scores["model"] == "base"])
    d = d[d["category"] == category]
    words = sorted(d["word"].unique())
    rows = []
    for model in order[1:]:
        dm = d[d["model"] == model]
        for w in [None] + words:
            h = dm if w is None else dm[dm["word"] != w]
            comp = h.groupby(["prefix_id", "polarity_word"])["diff"].mean().unstack()
            rows.append(dict(model=model, dropped="(none)" if w is None else w,
                             polarity="" if w is None else dm.loc[dm["word"] == w, "polarity_word"].iloc[0],
                             mean_D=(comp[POS] - comp[NEG]).mean()))
    return pd.DataFrame(rows)


def drop_two_words(scores: pd.DataFrame, category: str, models: list[str]) -> pd.DataFrame:
    """Mean D of `category` with every pair of probe words removed (is the result carried by one or two words?)."""
    d = pair_differences(scores, scores[scores["model"] == "base"])
    d = d[(d["category"] == category) & d["model"].isin(models)]
    rows = []
    for a, b in itertools.combinations(sorted(d["word"].unique()), 2):
        comp = d[~d["word"].isin([a, b])].groupby(["model", "prefix_id", "polarity_word"])["diff"].mean().unstack()
        for model, v in (comp[POS] - comp[NEG]).groupby("model").mean().items():
            rows.append(dict(model=model, dropped_1=a, dropped_2=b, mean_D=v))
    return pd.DataFrame(rows)


def worked_example(scores: pd.DataFrame, prefixes: pd.DataFrame, category: str, models: list[str]) -> str:
    """One prefix followed from word log-probs to Delta, delta+, delta-, D. Derogatory words are masked."""
    pid = scores.loc[scores["category"] == category, "prefix_id"].iloc[0]
    s = scores[scores["prefix_id"] == pid]
    words = s[s["model"] == "base"].sort_values(["polarity_word", "word"], ascending=[False, True])
    label, neg_i = {}, 0
    for w, p in zip(words["word"], words["polarity_word"]):
        if p == POS:
            label[w] = w
        else:
            neg_i += 1
            label[w] = f"[derog-{neg_i}]"
    piv = s.pivot(index="word", columns="model", values="logp").loc[words["word"], models]
    pol = words.set_index("word")["polarity_word"]
    text = prefixes.set_index("final_id").loc[pid, "prefix"]
    lines = [f"Prefix {pid} ({category}): ...{text[-80:]} [slot]", "", "| word | pool | " + " | ".join(models) + " |",
             "|---|---|" + "---|" * len(models)]
    for w in piv.index:
        lines.append(f"| {label[w]} | {'neutral' if pol[w] == POS else 'derogatory'} | " + " | ".join(f"{v:.3f}" for v in piv.loc[w]) + " |")
    dv = derived_vs_base(s, s[s["model"] == "base"]).set_index("model")
    pm = pool_means(s).set_index("model")
    lines += ["", "| model | mean logP neutral | mean logP derog. | Delta | delta+ | delta- | D = Delta_M - Delta_Base | delta+ - delta- |",
              "|---|---|---|---|---|---|---|---|"]
    for m in models:
        if m == "base":
            lines.append(f"| base | {pm.loc[m, 'logp_pos']:.3f} | {pm.loc[m, 'logp_neg']:.3f} | {pm.loc[m, 'delta']:.3f} | - | - | - | - |")
        else:
            r = dv.loc[m]
            lines.append(f"| {m} | {pm.loc[m, 'logp_pos']:.3f} | {pm.loc[m, 'logp_neg']:.3f} | {r['delta']:.3f} | {r['delta_pos']:.3f} | "
                         f"{r['delta_neg']:.3f} | {r['D']:.3f} | {r['delta_pos'] - r['delta_neg']:.3f} |")
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    res = ROOT / cfg["paths"]["results"]
    size = cfg["run"]["size"]
    order = list(cfg["models"][size])
    TAB.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(res / f"scores_{size}.parquet")
    nobos = pd.read_parquet(res / f"scores_{size}_base_nobos.parquet")
    prefixes = pd.read_csv(ROOT / cfg["paths"]["prefixes"])

    dm = build_delta_main(scores)
    scores_aligned = scores[scores["model"] != "base"]
    dn = build_delta_nobos(scores_aligned, nobos)
    err_main = identity_error(dm.dropna(subset=["D"]))
    err_nobos = identity_error(dn)
    assert err_main < 1e-9 and err_nobos < 1e-9, (err_main, err_nobos)
    assert len(dm) == 3000 and len(dn) == 2500 and not dm[["delta", "logp_pos", "logp_neg"]].isna().any().any()
    assert dm.loc[dm["model"] != "base", ["delta_pos", "delta_neg", "D"]].notna().all().all()
    write_new(dm, res / f"delta_{size}.parquet")
    write_new(dn, res / f"delta_{size}_base_nobos.parquet")

    summ = summary_table(dm, order)
    summ.to_csv(TAB / "summary_model_category.csv", index=False)
    both = d_side_by_side(dm, dn, order)
    both.to_csv(TAB / "D_model_category_main_vs_nobos.csv", index=False)
    pw = per_word_table(scores, order)
    pw.to_csv(TAB / "per_word_shift.csv", index=False)
    ratio_table(summ, both).to_csv(TAB / "ratios_exp.csv", index=False)
    drop = drop_one_word(scores, TW, order)
    drop.to_csv(TAB / "drop_one_word_transgender_woman.csv", index=False)
    drop_two_words(scores, TW, ["sft", "sft_dpo", "sft_ppo", "sft_kto", "kto"]).to_csv(TAB / "drop_two_words_transgender_woman.csv", index=False)

    # Check: mean D per model and category must equal the 04c tables (computed by an independent code path).
    s4 = pd.read_csv(ROOT / "reports/step_04_tables/provisional_D_by_model_category.csv")
    s5 = {"base_BOS": summ[summ["model"] != "base"][["model", "category", "mean_D"]],
          "base_noBOS": both[["model", "category", "mean_D_nobos"]].rename(columns={"mean_D_nobos": "mean_D"})}
    cmp_ = []
    for ref, frame in s5.items():
        a = s4[s4["reference"] == ref][["model", "category", "mean_D"]].merge(frame, on=["model", "category"], suffixes=("_s4", "_s5"),
                                                                              validate="one_to_one")
        assert len(a) == 30, len(a)
        cmp_.append((a["mean_D_s4"] - a["mean_D_s5"]).abs().max())
    chk = pd.DataFrame([dict(check="identity D = delta_pos - delta_neg, main (2500 rows)", max_abs_error=err_main),
                        dict(check="identity D = delta_pos - delta_neg, no-BOS (2500 rows)", max_abs_error=err_nobos),
                        dict(check="mean D vs S4 table (BOS), all model x category cells", max_abs_error=cmp_[0]),
                        dict(check="mean D vs S4 table (no BOS), all model x category cells", max_abs_error=cmp_[1])])
    chk.to_csv(TAB / "identity_and_s4_checks.csv", index=False)
    ex = worked_example(scores, prefixes, "Female Homosexual", order)
    (TAB / "worked_example.md").write_text(ex + "\n")

    pd.set_option("display.width", 200)
    print(chk.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
    print("\nmean D (BOS | no BOS), ALL:")
    print(both[both["category"] == "ALL"][["model", "mean_D_main", "mean_D_nobos"]].round(3).to_string(index=False))
    print("\n" + ex)


if __name__ == "__main__":
    main()
