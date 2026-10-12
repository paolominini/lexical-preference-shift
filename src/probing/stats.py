"""Inferential helpers for the research-question scripts 07-09: paired Wilcoxon summary, bootstrap CI, McNemar, Friedman, Bonferroni,
length check, Kruskal-Wallis, ANCOVA by category.

All functions take plain arrays / DataFrames and return dicts of numbers, so the scripts only arrange them into tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import mcnemar


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, seed: int, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling the units of `x` (prefixes: each prefix is one unit). Deterministic for a given seed."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation: (sum of ranks of positive d - sum of ranks of negative d) / total rank sum."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired_summary(d: np.ndarray, n_boot: int, seed: int) -> dict:
    """Summary of paired differences d (e.g. D_{M,Base}(s) over prefixes): location, CIs, effect sizes, two-sided tests.

    Wilcoxon signed-rank (two-sided) is the main test; the sign test (binomial on d > 0) drops its symmetry assumption;
    skewness is reported as the assumption check. d_z = mean / SD (ddof = 1).
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    lo, hi = bootstrap_mean_ci(d, n_boot, seed)
    t_half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    w = stats.wilcoxon(d, alternative="two-sided")
    n_pos, n_nonzero = int((d > 0).sum()), int((d != 0).sum())
    return dict(n=n, mean=mean, ci_boot_lo=lo, ci_boot_hi=hi, ci_t_lo=mean - t_half, ci_t_hi=mean + t_half,
                median=float(np.median(d)), sd=sd, d_z=mean / sd, rank_biserial=rank_biserial(d),
                share_pos=n_pos / n, wilcoxon_W=float(w.statistic), wilcoxon_p=float(w.pvalue),
                sign_test_p=float(stats.binomtest(n_pos, n_nonzero, 0.5).pvalue), skewness=float(stats.skew(d)))


def mcnemar_paired(ref: np.ndarray, model: np.ndarray) -> dict:
    """Exact McNemar test on a paired binary outcome (I_M of Base = `ref` vs I_M of M = `model`, same prefixes)."""
    ref, model = np.asarray(ref, dtype=bool), np.asarray(model, dtype=bool)
    both = int((ref & model).sum())
    only_ref = int((ref & ~model).sum())
    only_model = int((~ref & model).sum())
    neither = int((~ref & ~model).sum())
    res = mcnemar([[both, only_ref], [only_model, neither]], exact=True)
    return dict(mc_both=both, mc_only_ref=only_ref, mc_only_model=only_model, mc_neither=neither,
                mcnemar_p=float(res.pvalue), share_I_ref=float(ref.mean()), share_I_model=float(model.mean()))


def friedman_summary(x: np.ndarray) -> dict:
    """Friedman test on an n x k array (rows = blocks, e.g. prefixes; columns = related samples, e.g. models).

    Kendall's W = chi2 / (n (k - 1)) is the effect size (0 = no agreement between blocks on the ordering, 1 = full agreement).
    Mean ranks are the within-block ranks (1 = lowest) averaged over blocks, one per column.
    """
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    res = stats.friedmanchisquare(*x.T)
    ranks = np.apply_along_axis(stats.rankdata, 1, x).mean(axis=0)
    return dict(n=n, k=k, chi2=float(res.statistic), df=k - 1, p=float(res.pvalue),
                kendall_w=float(res.statistic / (n * (k - 1))), mean_ranks=ranks.tolist())


def bonferroni(p: np.ndarray, m: int) -> np.ndarray:
    """Bonferroni-adjusted p-values for a family of m tests: min(1, p * m)."""
    return np.minimum(1.0, np.asarray(p, dtype=float) * m)


def kruskal_summary(groups: list[np.ndarray]) -> dict:
    """Kruskal-Wallis H test across k independent groups; effect size epsilon^2 = H / (n - 1) (0 = no separation, 1 = full)."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    n = sum(len(g) for g in groups)
    res = stats.kruskal(*groups)
    return dict(n=n, k=len(groups), H=float(res.statistic), df=len(groups) - 1, p=float(res.pvalue),
                epsilon2=float(res.statistic / (n - 1)))


def ancova_category(df: pd.DataFrame, y: str, group: str, covariates: list[str], slope_var: str | None = None) -> dict:
    """OLS ANCOVA `y ~ C(group) + covariates` (covariates are formula terms, e.g. "toxicity", "C(tox_bin)").

    Returns: Type II F, df and p of every term, partial eta^2 = SS_term / (SS_term + SS_resid), coefficients with 95% CIs
    (`coefs`), R^2, residuals, and adjusted group means = prediction with the group dummies set to each group and every other
    design column at its sample mean. If `slope_var` (a numeric covariate) is given, also the homogeneity-of-slopes check:
    nested F of the additive model vs the model with C(group):slope_var added, and the within-group slopes of slope_var.
    """
    rhs = " + ".join([f"C({group})"] + covariates)
    fit = smf.ols(f"{y} ~ {rhs}", data=df).fit()
    a2 = anova_lm(fit, typ=2)
    ss_res = a2.loc["Residual", "sum_sq"]
    terms = {t: dict(F=float(r["F"]), df=float(r["df"]), p=float(r["PR(>F)"]), partial_eta2=float(r["sum_sq"] / (r["sum_sq"] + ss_res)))
             for t, r in a2.drop(index="Residual").iterrows()}
    names = fit.model.exog_names
    xbar = fit.model.exog.mean(axis=0)
    levels = sorted(df[group].unique())
    adjusted = {}
    for lev in levels:
        x = xbar.copy()
        for j, nm in enumerate(names):
            if nm.startswith(f"C({group})[T."):
                x[j] = float(nm == f"C({group})[T.{lev}]")
        adjusted[lev] = float(x @ fit.params.to_numpy())
    ci = fit.conf_int()
    coefs = pd.DataFrame({"coef": fit.params, "ci_lo": ci[0], "ci_hi": ci[1], "p": fit.pvalues})
    out = dict(terms=terms, coefs=coefs, r2=float(fit.rsquared), df_resid=float(fit.df_resid), resid=fit.resid.to_numpy(),
               adjusted_means=adjusted, raw_means=df.groupby(group)[y].mean().to_dict())
    if slope_var is not None:
        full = smf.ols(f"{y} ~ {rhs} + C({group}):{slope_var}", data=df).fit()
        nested = anova_lm(fit, full).iloc[1]
        out.update(interaction_F=float(nested["F"]), interaction_df=float(nested["df_diff"]), interaction_p=float(nested["Pr(>F)"]))
        others = [c for c in covariates if c != slope_var]
        sep = smf.ols(f"{y} ~ " + " + ".join([f"C({group})", f"C({group}):{slope_var}"] + others), data=df).fit()
        out["group_slopes"] = {lev: float(sep.params[f"C({group})[{lev}]:{slope_var}"]) for lev in levels}
    return out


def varying_pools(per_word: pd.DataFrame, pool_keys: list[str]) -> pd.Series:
    """Boolean per pool: does n_tokens take more than one value among the pool's words?"""
    return per_word.groupby(pool_keys)["n_tokens"].nunique() > 1


def within_pool_length_slope(per_word: pd.DataFrame, pool_keys: list[str], y: str = "mean_shift") -> dict:
    """Slope of `y` on n_tokens within pools (both demeaned per pool), over the pools where n_tokens varies.

    Descriptive: words are the units, so n is tiny and length is confounded with word identity. Also counts the pools where
    the longest words have a lower mean `y` than the shortest ones.
    """
    vary = varying_pools(per_word, pool_keys)
    keep = per_word.set_index(pool_keys).index.isin(vary[vary].index)
    w = per_word[keep]
    x_dm = w["n_tokens"] - w.groupby(pool_keys)["n_tokens"].transform("mean")
    y_dm = w[y] - w.groupby(pool_keys)[y].transform("mean")
    longer_lower = 0
    for _, g in w.groupby(pool_keys):
        hi = g.loc[g["n_tokens"] == g["n_tokens"].max(), y].mean()
        lo = g.loc[g["n_tokens"] == g["n_tokens"].min(), y].mean()
        longer_lower += int(hi < lo)
    return dict(slope=float((x_dm * y_dm).sum() / (x_dm ** 2).sum()), n_words=len(w), n_pools=int(vary.sum()),
                n_pools_longer_lower=longer_lower)
