"""RQ3 / H3: does the alignment shift D_{M,Base}(s) = Shift_M(c,s) differ across the five categories, also after
adjusting for prefix toxicity (Detoxify) and source polarity? One analysis per post-trained model; no pooled model x category test.

Unit = prefix (100 per category, 500 per model). y = D_{M,Base}(s), reference Base with BOS (main); Base without BOS as robustness.

Outcome rules (fixed before running):
  per model M, "differ"   = KW p < 0.05 AND ANCOVA (y ~ C(category) + toxicity + notsafe, Type II) category p < 0.05
                            AND range of raw AND adjusted category means > 0.06 nat;
               "do not differ" = (KW and ANCOVA both n.s.) OR (both ranges <= 0.06); otherwise "mixed".
  H3 (SFT family, headline) supported = all four "differ" and no toxicity-coding variant (rank, 3 bins) turns an adjusted category
  test n.s.; not supported = none of the four "differ"; mixed otherwise. Direct KTO: own per-model verdict, reported separately.

Reads (no re-scoring): results/delta_1.4b.parquet, results/delta_1.4b_base_nobos.parquet, results/toxicity.parquet,
data/final_prefix_set.csv (read only), reports/step_05_tables/{summary_model_category, D_model_category_main_vs_nobos,
per_word_shift}.csv, reports/step_07_tables/components_vs_base.csv (checks and the two-word check).
Writes:
  reports/step_09_tables/h3_shift_by_category.csv       Shift_M(c) [bootstrap 95% CI, 1000 resamples of prefixes], e^mean, SD, IQR...
  reports/step_09_tables/h3_family_summary.csv          per category: four SFT-family Shift values, mean, min-max; Direct KTO apart
  reports/step_09_tables/h3_tests.csv                   per model x reference: KW, ANCOVA, interaction, rank / 3-bin variants, verdicts
  reports/step_09_tables/h3_adjusted_means.csv          raw vs adjusted category means per model, variant and reference
  reports/step_09_tables/source_word_shares.csv         share of each category x polarity cell by source word
  reports/step_09_tables/two_word_neutral_check.csv     neutral words: KTO - SFT and KTO - SFT-KTO mean shift
  reports/step_09_tables/identity_attack_descriptive.csv  identity_attack per category + Spearman rho with y per model (no p)
  reports/step_09_tables/gate_checks.csv
  results/figures/rq3_shift_by_category, rq3_shift_distribution, rq3_toxicity_scatter (.png, .pdf)
Usage: .venv/bin/python scripts/09_rq3.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probing.config import load_config  # noqa: E402
from probing.stats import ancova_category, bootstrap_mean_ci, kruskal_summary  # noqa: E402

TAB = ROOT / "reports/step_09_tables"
N_BOOT = 1000           # resamples of the category's prefixes (07/08 use 10,000)
ALPHA = 0.05
PRECISION = 0.05        # nat; category means closer than this are not distinguished in the wording (precision scale of 08_rq2.py)
CLEARLY_ABOVE = 0.06    # range edge of the outcome rule (upper edge of the 0.04-0.06 band)
FAMILY = ["sft", "sft_dpo", "sft_ppo", "sft_kto"]
MODELS = FAMILY + ["kto"]
LABEL = {"sft": "SFT", "sft_dpo": "SFT-DPO", "sft_ppo": "SFT-PPO", "sft_kto": "SFT-KTO", "kto": "KTO (direct)"}
SHORT = {"African - Afro American": "African", "Female Homosexual": "Female Homosexual", "Homosexual Man": "Homosexual Man",
         "Sex Worker, Prostitute": "Sex Worker", "Transgender Woman": "Transgender Woman"}
ABBR = {"African - Afro American": "AF", "Female Homosexual": "FH", "Homosexual Man": "HM", "Sex Worker, Prostitute": "SW",
        "Transgender Woman": "TW"}
COVARIATES = {"linear": ["toxicity", "notsafe"], "rank": ["tox_rank", "notsafe"], "bin3": ["C(tox_bin)", "notsafe"]}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#8a8a8a"
CAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots 1-5, fixed order (alphabetical categories)


def load_frame(res: Path, prefixes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long table (model x prefix) with y (main and no-BOS reference), toxicity codings, polarity, identity_attack, source word."""
    dm = pd.read_parquet(res / "delta_1.4b.parquet")
    dn = pd.read_parquet(res / "delta_1.4b_base_nobos.parquet")
    tox = pd.read_parquet(res / "toxicity.parquet")
    main = dm[dm["model"] != "base"][["prefix_id", "category", "model", "D"]].rename(columns={"D": "y"})
    nob = dn[["prefix_id", "model", "D"]].rename(columns={"D": "y_nobos"})
    t = tox[["prefix_id", "category", "source_polarity", "toxicity", "identity_attack"]].rename(columns={"category": "category_tox"})
    t["notsafe"] = (t["source_polarity"] == "NOT SAFE").astype(float)
    t["tox_rank"] = stats.rankdata(t["toxicity"]) / len(t)
    t["tox_bin"] = np.select([t["toxicity"] < 0.05, t["toxicity"] < 0.5], ["<0.05", "0.05-0.5"], ">=0.5")
    p = prefixes[["final_id", "category", "polarity", "source_word"]].rename(
        columns={"final_id": "prefix_id", "category": "category_set", "polarity": "polarity_set"})
    df = main.merge(nob, on=["prefix_id", "model"], how="left").merge(t, on="prefix_id", how="left").merge(p, on="prefix_id", how="left")
    for c in ["category", "prefix_id", "model"]:
        df[c] = df[c].astype(str)
    return df.sort_values(["model", "prefix_id"]).reset_index(drop=True), dm


def shift_table(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shift_M(c) per model x category: mean [bootstrap 95% CI], e^mean, share > 0, SD, median, IQR; no-BOS mean as a plain column."""
    rows = []
    for (m, cat), g in df.groupby(["model", "category"]):
        y = g["y"].to_numpy()
        lo, hi = bootstrap_mean_ci(y, N_BOOT, seed)
        q1, q3 = np.quantile(y, [0.25, 0.75])
        rows.append(dict(model=m, category=cat, n_prefixes=len(y), mean=y.mean(), ci_boot_lo=lo, ci_boot_hi=hi,
                         exp_mean=float(np.exp(y.mean())), share_pos=float((y > 0).mean()), sd=y.std(ddof=1), median=float(np.median(y)),
                         iqr=q3 - q1, mean_nobos=g["y_nobos"].mean()))
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], MODELS)
    return out.sort_values(["model", "category"]).reset_index(drop=True)


def family_summary(sh: pd.DataFrame) -> pd.DataFrame:
    """Per category: the four SFT-family Shift values, their mean and min-max (with model names); Direct KTO in its own column."""
    w = sh.pivot(index="category", columns="model", values="mean")
    fam = w[FAMILY]
    out = pd.DataFrame({f"shift_{m}": fam[m] for m in FAMILY})
    out["family_mean"] = fam.mean(axis=1)
    out["family_min"], out["family_min_model"] = fam.min(axis=1), fam.idxmin(axis=1).astype(str)
    out["family_max"], out["family_max_model"] = fam.max(axis=1), fam.idxmax(axis=1).astype(str)
    out["shift_kto_direct"] = w["kto"]
    return out.reset_index()


def verdict(kw_p: float, an_p: float, range_raw: float, range_adj: float) -> str:
    """Per-model H3 outcome rule (see the module docstring)."""
    if kw_p < ALPHA and an_p < ALPHA and range_raw > CLEARLY_ABOVE and range_adj > CLEARLY_ABOVE:
        return "differ"
    if (kw_p >= ALPHA and an_p >= ALPHA) or (range_raw <= CLEARLY_ABOVE and range_adj <= CLEARLY_ABOVE):
        return "do not differ"
    return "mixed"


def model_tests(g: pd.DataFrame, yname: str, reference: str, variants: bool) -> tuple[dict, list[dict]]:
    """KW + main ANCOVA (+ interaction, rank / 3-bin variants if `variants`) for one model; returns the test row and adjusted means rows."""
    d = g.rename(columns={yname: "yy"})
    cats = sorted(d["category"].unique())
    kw = kruskal_summary([d.loc[d["category"] == c, "yy"].to_numpy() for c in cats])
    lin = ancova_category(d, "yy", "category", COVARIATES["linear"], slope_var="toxicity" if variants else None)
    raw, adj = lin["raw_means"], lin["adjusted_means"]
    t_cat, t_tox, t_ns = lin["terms"]["C(category)"], lin["terms"]["toxicity"], lin["terms"]["notsafe"]
    co = lin["coefs"]
    resid_sd = pd.Series(lin["resid"]).groupby(d["category"].to_numpy()).std(ddof=1)
    cat_sd = d.groupby("category")["yy"].std(ddof=1)
    row = dict(reference=reference, model=g["model"].iloc[0], n=len(d), kw_H=kw["H"], kw_df=kw["df"], kw_p=kw["p"],
               kw_epsilon2=kw["epsilon2"], range_raw=max(raw.values()) - min(raw.values()),
               max_category_raw=SHORT[max(raw, key=raw.get)], min_category_raw=SHORT[min(raw, key=raw.get)],
               ancova_F_category=t_cat["F"], ancova_df_category=t_cat["df"], ancova_df_resid=lin["df_resid"],
               ancova_p_category=t_cat["p"], partial_eta2_category=t_cat["partial_eta2"],
               range_adj=max(adj.values()) - min(adj.values()), max_category_adj=SHORT[max(adj, key=adj.get)],
               min_category_adj=SHORT[min(adj, key=adj.get)], max_abs_adj_minus_raw=max(abs(adj[c] - raw[c]) for c in cats),
               tox_coef=co.loc["toxicity", "coef"], tox_ci_lo=co.loc["toxicity", "ci_lo"], tox_ci_hi=co.loc["toxicity", "ci_hi"],
               tox_p=t_tox["p"], partial_eta2_tox=t_tox["partial_eta2"], notsafe_coef=co.loc["notsafe", "coef"],
               notsafe_ci_lo=co.loc["notsafe", "ci_lo"], notsafe_ci_hi=co.loc["notsafe", "ci_hi"], notsafe_p=t_ns["p"],
               partial_eta2_notsafe=t_ns["partial_eta2"], r2=lin["r2"], resid_skew=float(stats.skew(lin["resid"])),
               resid_sd_max_over_min=float(resid_sd.max() / resid_sd.min()), category_sd_min=float(cat_sd.min()),
               category_sd_max=float(cat_sd.max()))
    adj_rows = [dict(reference=reference, model=row["model"], variant="linear", category=c, raw_mean=raw[c], adjusted_mean=adj[c],
                     toxicity_slope_within_category=lin.get("group_slopes", {}).get(c, np.nan)) for c in cats]
    if variants:
        row.update(interaction_F=lin["interaction_F"], interaction_df=lin["interaction_df"], interaction_p=lin["interaction_p"])
        for v in ["rank", "bin3"]:
            a = ancova_category(d, "yy", "category", COVARIATES[v])
            row.update({f"{v}_F_category": a["terms"]["C(category)"]["F"], f"{v}_p_category": a["terms"]["C(category)"]["p"],
                        f"{v}_partial_eta2_category": a["terms"]["C(category)"]["partial_eta2"],
                        f"{v}_range_adj": max(a["adjusted_means"].values()) - min(a["adjusted_means"].values())})
            adj_rows += [dict(reference=reference, model=row["model"], variant=v, category=c, raw_mean=a["raw_means"][c],
                              adjusted_mean=a["adjusted_means"][c], toxicity_slope_within_category=np.nan) for c in cats]
        row["variants_all_significant"] = bool(row["rank_p_category"] < ALPHA and row["bin3_p_category"] < ALPHA)
    row["verdict"] = verdict(row["kw_p"], row["ancova_p_category"], row["range_raw"], row["range_adj"])
    return row, adj_rows


def tests_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """h3_tests (one row per model x reference, H3 family outcome on the main rows) and h3_adjusted_means."""
    rows, adj = [], []
    for ref, yname, variants in [("base_bos", "y", True), ("base_nobos", "y_nobos", False)]:
        for m in MODELS:
            r, a = model_tests(df[df["model"] == m], yname, ref, variants)
            rows.append(r)
            adj += a
    t = pd.DataFrame(rows)
    main_fam = t[(t["reference"] == "base_bos") & t["model"].isin(FAMILY)]
    if (main_fam["verdict"] == "differ").all() and main_fam["variants_all_significant"].all():
        outcome = "supported"
    elif (main_fam["verdict"] != "differ").all():
        outcome = "not supported"
    else:
        outcome = "mixed"
    t["h3_sft_family_outcome"] = np.where((t["reference"] == "base_bos") & t["model"].isin(FAMILY), outcome, None)
    a = pd.DataFrame(adj)
    a["adj_minus_raw"] = a["adjusted_mean"] - a["raw_mean"]
    grp = a.groupby(["reference", "model", "variant"])
    a["rank_raw"] = grp["raw_mean"].rank(ascending=False).astype(int)
    a["rank_adj"] = grp["adjusted_mean"].rank(ascending=False).astype(int)
    return t, a


def source_word_shares(prefixes: pd.DataFrame) -> pd.DataFrame:
    """Share of each category x polarity cell (and of the category) by source word: which words drive a category result."""
    c = prefixes.groupby(["category", "polarity", "source_word"]).size().rename("n_prefixes").reset_index()
    c["share_of_cell"] = c["n_prefixes"] / c.groupby(["category", "polarity"])["n_prefixes"].transform("sum")
    c["share_of_category"] = c["n_prefixes"] / c.groupby("category")["n_prefixes"].transform("sum")
    return c.sort_values(["category", "polarity", "n_prefixes"], ascending=[True, False, False]).reset_index(drop=True)


def two_word_check(per_word: pd.DataFrame) -> pd.DataFrame:
    """Neutral words: mean shift vs Base per model and the two labelled differences (Base cancels in both)."""
    w = per_word[per_word["polarity_word"] == "SAFE"].pivot_table(index=["category", "word", "n_tokens"], columns="model",
                                                                    values="mean_shift").reset_index()
    out = w[["category", "word", "n_tokens"]].copy()
    out["n_words"] = out["word"].str.split().str.len()
    for m in ["sft", "sft_kto", "kto"]:
        out[f"shift_{m}"] = w[m]
    out["KTO - SFT"] = w["kto"] - w["sft"]
    out["KTO - SFT-KTO"] = w["kto"] - w["sft_kto"]
    return out


def identity_attack_table(df: pd.DataFrame) -> pd.DataFrame:
    """identity_attack per category (mean, median) and Spearman rho with y per model; descriptive only, no p (not in the ANCOVA)."""
    rows = []
    for cat, g in [("ALL", df)] + list(df.groupby("category")):
        one = g[g["model"] == "sft"]
        row = dict(category=cat, n_prefixes=len(one), mean_identity_attack=one["identity_attack"].mean(),
                   median_identity_attack=one["identity_attack"].median(), mean_toxicity=one["toxicity"].mean())
        for m in MODELS:
            gm = g[g["model"] == m]
            row[f"rho_{m}"] = float(stats.spearmanr(gm["identity_attack"], gm["y"]).statistic)
        rows.append(row)
    return pd.DataFrame(rows)


def kw_hand(y: np.ndarray, groups: np.ndarray) -> float:
    """Kruskal-Wallis H from the textbook formula with tie correction (check 3b)."""
    n = len(y)
    r = stats.rankdata(y)
    h = 12 / (n * (n + 1)) * sum(r[groups == c].sum() ** 2 / (groups == c).sum() for c in np.unique(groups)) - 3 * (n + 1)
    _, t = np.unique(y, return_counts=True)
    return h / (1 - (t ** 3 - t).sum() / (n ** 3 - n))


def ancova_F_hand(g: pd.DataFrame, yname: str) -> float:
    """Category F from the residual sums of squares with and without the category dummies (numpy lstsq; check 3a)."""
    y = g[yname].to_numpy()
    cov = np.column_stack([np.ones(len(g)), g["toxicity"], g["notsafe"]])
    dummies = pd.get_dummies(g["category"], drop_first=True).to_numpy(float)
    full = np.column_stack([cov, dummies])
    rss = [float(((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum()) for X in (cov, full)]
    df1, df2 = dummies.shape[1], len(y) - full.shape[1]
    return ((rss[0] - rss[1]) / df1) / (rss[1] / df2)


def gate_checks(df: pd.DataFrame, dm: pd.DataFrame, prefixes: pd.DataFrame, sh: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """Checks 1-3 and the estimability / count checks (check 4, a seeded re-run, is done from the shell)."""
    s5 = pd.read_csv(ROOT / "reports/step_05_tables/summary_model_category.csv")
    s5 = s5[s5["category"] != "ALL"].set_index(["model", "category"])["mean_D"]
    s7 = pd.read_csv(ROOT / "reports/step_07_tables/components_vs_base.csv")
    s7 = s7[s7["category"] != "ALL"].set_index(["model", "category"])["mean_D"]
    nb = pd.read_csv(ROOT / "reports/step_05_tables/D_model_category_main_vs_nobos.csv")
    nb = nb[nb["category"] != "ALL"].set_index(["model", "category"])["mean_D_nobos"]
    k = sh.assign(model=sh["model"].astype(str)).set_index(["model", "category"])
    offset = (k["mean_nobos"] - k["mean"]).groupby(level="category").agg(lambda s: s.max() - s.min())
    one = df[df["model"] == "sft"]
    base_ids = set(dm.loc[dm["model"] == "base", "prefix_id"].astype(str))
    tox_ids, set_ids = set(one["prefix_id"]), set(prefixes["final_id"].astype(str))
    per_cat = df.groupby(["model", "category"]).size()
    cells = one.groupby(["category", "source_polarity"])["toxicity"].agg(["min", "max"])
    main = df.copy()
    g3a = max(abs(ancova_F_hand(main[main["model"] == m], "y") - tests.query("reference == 'base_bos' and model == @m")["ancova_F_category"].iloc[0])
              for m in MODELS)
    g3b = max(abs(kw_hand(main.loc[main["model"] == m, "y"].to_numpy(), main.loc[main["model"] == m, "category"].to_numpy())
                  - tests.query("reference == 'base_bos' and model == @m")["kw_H"].iloc[0]) for m in MODELS)
    chk = [("1a. 25 Shift_M(c) means vs S5 summary_model_category mean_D", float((k["mean"] - s5.reindex(k.index)).abs().max()), 1e-12),
           ("1b. 25 Shift_M(c) means vs S7 components_vs_base mean_D", float((k["mean"] - s7.reindex(k.index)).abs().max()), 1e-12),
           ("1c. 25 no-BOS means vs S5 D_model_category_main_vs_nobos", float((k["mean_nobos"] - nb.reindex(k.index)).abs().max()), 1e-12),
           ("1d. no-BOS - BOS offset: max spread across the five models within a category", float(offset.max()), 1e-12),
           ("2a. prefix ids: delta (aligned, Base) vs toxicity vs prefix set, mismatches",
            float(len(tox_ids ^ set_ids) + len(base_ids ^ set_ids) + sum(len(set(df.loc[df["model"] == m, "prefix_id"]) ^ set_ids) for m in MODELS)), 0.0),
           ("2b. category labels delta = toxicity = prefix set, mismatches",
            float(((df["category"] != df["category_tox"]) | (df["category"] != df["category_set"])).sum()), 0.0),
           ("2c. source polarity toxicity = prefix set, mismatches", float((df["source_polarity"] != df["polarity_set"]).sum()), 0.0),
           ("2d. missing y, y_nobos or toxicity", float(df[["y", "y_nobos", "toxicity"]].isna().sum().sum()), 0.0),
           ("2e. cells (model x category) with n != 100", float((per_cat != 100).sum() + (len(per_cat) != 25)), 0.0),
           ("3a. ANCOVA category F: numpy RSS vs statsmodels Type II, 5 models", g3a, 1e-9),
           ("3b. KW H: textbook formula vs scipy, 5 models", g3b, 1e-9),
           ("3c. S6 statement reproduced: category x polarity cells NOT spanning toxicity (<0.05 and >0.9) [finding, non-blocking]",
            float(((cells["min"] >= 0.05) | (cells["max"] <= 0.9)).sum()), 0.0),
           ("3d. (added after 3c) cells whose toxicity range does NOT contain the evaluation point (mean toxicity)",
            float(((cells["min"] > one["toxicity"].mean()) | (cells["max"] < one["toxicity"].mean())).sum()), 0.0)]
    out = pd.DataFrame(chk, columns=["check", "max_abs_error", "tolerance"])
    out["pass"] = out["max_abs_error"] <= out["tolerance"]
    out["blocking"] = ~out["check"].str.contains("non-blocking")
    assert out.loc[out["blocking"], "pass"].all(), out
    return out


def _style(ax, grid_axis: str = "y") -> None:
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK)
    (ax.yaxis if grid_axis == "y" else ax.xaxis).grid(True, color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=150)
    fig.savefig(path.with_suffix(".pdf"))


def figure_shift(sh: pd.DataFrame, path: Path) -> None:
    """Shift_M(c) [95% bootstrap CI] per category: SFT family in blue (one marker per model), Direct KTO in orange."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = sorted(sh["category"].unique())
    markers = {"sft": "o", "sft_dpo": "s", "sft_ppo": "^", "sft_kto": "D", "kto": "v"}
    offs = dict(zip(MODELS, [0.3, 0.15, 0.0, -0.15, -0.36]))
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.axvline(0, color=MUTED, lw=1)
    for m in MODELS:
        s = sh[sh["model"] == m].set_index("category").loc[cats]
        y = np.arange(len(cats))[::-1] + offs[m]
        color = ORANGE if m == "kto" else BLUE
        ax.errorbar(s["mean"], y, xerr=[s["mean"] - s["ci_boot_lo"], s["ci_boot_hi"] - s["mean"]], fmt=markers[m], ms=6.5,
                    color=color, ecolor=color, elinewidth=1.1, capsize=0, markeredgecolor="white", label=LABEL[m])
    for i in range(1, len(cats)):
        ax.axhline(i - 0.5, color="#eeeeee", lw=0.8)
    ax.set_yticks(np.arange(len(cats))[::-1], [SHORT[c] for c in cats])
    ax.set_xlabel("Shift_M(c) = mean D_{M,Base}(s) over the category's 100 prefixes (nat), 95% bootstrap CI")
    ax.set_title("RQ3: alignment shift by category (SFT family in blue, Direct KTO in orange)", fontsize=10, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, ncol=1)
    _style(ax, grid_axis="x")
    note = ("D > 0: the model separates neutral from derogatory words more than Base (RAW, with BOS) after the same prefixes. "
            "CI: 1000 bootstrap resamples of the category's prefixes.")
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(note, 125)), fontsize=8, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, path)
    plt.close(fig)


def figure_distribution(df: pd.DataFrame, path: Path) -> None:
    """Violins of D_{M,Base}(s) per category, one panel per model; diamond = category mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = sorted(df["category"].unique())
    fig, axes = plt.subplots(1, 5, figsize=(13, 4.4), sharey=True)
    for ax, m in zip(axes, MODELS):
        g = df[df["model"] == m]
        data = [g.loc[g["category"] == c, "y"].to_numpy() for c in cats]
        color = ORANGE if m == "kto" else BLUE
        vp = ax.violinplot(data, positions=range(len(cats)), widths=0.8, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(color)
            b.set_alpha(0.3)
            b.set_edgecolor(color)
        ax.plot(range(len(cats)), [d.mean() for d in data], "D", ms=5.5, color=INK, markeredgecolor="white")
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.set_xticks(range(len(cats)), [ABBR[c] for c in cats])
        ax.set_title(LABEL[m], fontsize=10, color=INK)
        _style(ax)
    axes[0].set_ylabel("D_{M,Base}(s) (nat)")
    note = ("One violin per category (n = 100 prefixes); diamond = Shift_M(c). AF = African, FH = Female Homosexual, HM = Homosexual Man, "
            "SW = Sex Worker, TW = Transgender Woman.")
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(note, 160)), fontsize=8, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, path)
    plt.close(fig)


def figure_toxicity(df: pd.DataFrame, path: Path) -> None:
    """Prefix toxicity vs D_{M,Base}(s): grey points, one least-squares line per category (descriptive), one panel per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = sorted(df["category"].unique())
    fig, axes = plt.subplots(1, 5, figsize=(13, 4.6), sharey=True)
    xs = np.array([0.0, 1.0])
    for ax, m in zip(axes, MODELS):
        g = df[df["model"] == m]
        ax.scatter(g["toxicity"], g["y"], s=7, color=MUTED, alpha=0.35, lw=0)
        for c, color in zip(cats, CAT_COLORS):
            gc = g[g["category"] == c]
            b, a = np.polyfit(gc["toxicity"], gc["y"], 1)
            ax.plot(xs, a + b * xs, color=color, lw=2, label=SHORT[c])
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.set_title(LABEL[m], fontsize=10, color=INK)
        ax.set_xlabel("prefix toxicity (Detoxify)")
        _style(ax)
    axes[0].set_ylabel("D_{M,Base}(s) (nat)")
    axes[0].set_ylim(-2.5, 3.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=9)
    note = ("Lines: per-category least-squares fit of D on toxicity (descriptive; the ANCOVA also adjusts for source polarity). "
            "y-axis cut at -2.5 / 3.0 for readability; a few points fall outside.")
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(note, 160)), fontsize=8, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    _save(fig, path)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    res = ROOT / cfg["paths"]["results"]
    seed = int(cfg["project"]["seed"])
    TAB.mkdir(parents=True, exist_ok=True)
    (res / "figures").mkdir(exist_ok=True)
    prefixes = pd.read_csv(ROOT / cfg["paths"]["prefixes"])
    df, dm = load_frame(res, prefixes)
    per_word = pd.read_csv(ROOT / "reports/step_05_tables/per_word_shift.csv")

    sh = shift_table(df, seed)
    fam = family_summary(sh)
    tests, adj = tests_tables(df)
    gates = gate_checks(df, dm, prefixes, sh, tests)
    for name, t in [("h3_shift_by_category", sh), ("h3_family_summary", fam), ("h3_tests", tests), ("h3_adjusted_means", adj),
                    ("source_word_shares", source_word_shares(prefixes)), ("two_word_neutral_check", two_word_check(per_word)),
                    ("identity_attack_descriptive", identity_attack_table(df)), ("gate_checks", gates)]:
        t.to_csv(TAB / f"{name}.csv", index=False)
    figure_shift(sh, res / "figures/rq3_shift_by_category")
    figure_distribution(df, res / "figures/rq3_shift_distribution")
    figure_toxicity(df, res / "figures/rq3_toxicity_scatter")

    print(f"checks: {int(gates['pass'].sum())}/{len(gates)} pass; non-blocking failures: "
          f"{gates.loc[~gates['pass'] & ~gates['blocking'], 'check'].tolist()}")
    main_t = tests[tests["reference"] == "base_bos"]
    cols = ["model", "kw_H", "kw_p", "kw_epsilon2", "range_raw", "ancova_F_category", "ancova_p_category", "range_adj",
            "interaction_p", "rank_p_category", "bin3_p_category", "verdict"]
    print(main_t[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"H3 (SFT family): {main_t['h3_sft_family_outcome'].iloc[0]}; Direct KTO verdict: {main_t.loc[main_t['model'] == 'kto', 'verdict'].iloc[0]}")
    nb = tests[tests["reference"] == "base_nobos"]
    print("no-BOS verdicts:", dict(zip(nb["model"], nb["verdict"])))


if __name__ == "__main__":
    main()
