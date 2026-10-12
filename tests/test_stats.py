"""Tests for the inferential helpers on small hand-made inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

from probing.stats import (ancova_category, bonferroni, bootstrap_mean_ci, friedman_summary, kruskal_summary, mcnemar_paired,
                           paired_summary, rank_biserial, within_pool_length_slope)


def test_d_z_and_cis_hand_case() -> None:
    d = np.array([1.0, 2.0, 3.0, 4.0, -1.0])
    s = paired_summary(d, n_boot=2000, seed=0)
    assert s["mean"] == pytest.approx(1.8)
    assert s["sd"] == pytest.approx(np.std(d, ddof=1))
    assert s["d_z"] == pytest.approx(1.8 / np.std(d, ddof=1))
    assert s["share_pos"] == pytest.approx(0.8)
    assert s["ci_boot_lo"] <= s["mean"] <= s["ci_boot_hi"]
    assert s["ci_t_lo"] < s["mean"] < s["ci_t_hi"]


def test_bootstrap_deterministic() -> None:
    x = np.random.default_rng(1).normal(size=200)
    assert bootstrap_mean_ci(x, 1000, seed=42) == bootstrap_mean_ci(x, 1000, seed=42)
    lo, hi = bootstrap_mean_ci(x, 1000, seed=42)
    assert lo < x.mean() < hi


def test_wilcoxon_and_sign_test_match_scipy() -> None:
    d = np.random.default_rng(2).normal(0.3, 1.0, size=100)
    s = paired_summary(d, n_boot=100, seed=0)
    w = stats.wilcoxon(d)
    assert s["wilcoxon_W"] == w.statistic and s["wilcoxon_p"] == pytest.approx(w.pvalue)
    assert s["sign_test_p"] == pytest.approx(stats.binomtest(int((d > 0).sum()), 100).pvalue)


def test_rank_biserial_hand_case() -> None:
    # |d| ranks: 1 -> 1, -2 -> 2, 3 -> 3; positive rank sum 4, negative 2, total 6
    assert rank_biserial(np.array([1.0, -2.0, 3.0])) == pytest.approx((4 - 2) / 6)
    assert rank_biserial(np.array([1.0, 2.0])) == pytest.approx(1.0)


def test_mcnemar_counts_and_exact_p() -> None:
    ref = np.array([1] * 10 + [1] * 2 + [0] * 8 + [0] * 5, dtype=bool)
    mod = np.array([1] * 10 + [0] * 2 + [1] * 8 + [0] * 5, dtype=bool)
    m = mcnemar_paired(ref, mod)
    assert (m["mc_both"], m["mc_only_ref"], m["mc_only_model"], m["mc_neither"]) == (10, 2, 8, 5)
    assert m["mcnemar_p"] == pytest.approx(stats.binomtest(2, 10, 0.5).pvalue)
    assert m["share_I_ref"] == pytest.approx(12 / 25) and m["share_I_model"] == pytest.approx(18 / 25)


def test_length_slope_recovers_slope_and_skips_constant_pools() -> None:
    # pool A: y = 5 - 0.5 * n; pool B: y = -3 - 0.5 * n (different level, same slope); pool C: constant n, wild y (ignored)
    rows = [("A", n, 5 - 0.5 * n) for n in (1, 2, 2, 3)] + [("B", n, -3 - 0.5 * n) for n in (2, 3)] + [("C", 2, y) for y in (10.0, -10.0)]
    pw = pd.DataFrame(rows, columns=["pool", "n_tokens", "mean_shift"])
    r = within_pool_length_slope(pw, ["pool"])
    assert r["slope"] == pytest.approx(-0.5)
    assert (r["n_pools"], r["n_words"], r["n_pools_longer_lower"]) == (2, 6, 2)


def test_friedman_matches_scipy() -> None:
    x = np.random.default_rng(3).normal(size=(60, 4)) + np.array([0.0, 0.2, 0.1, -0.1])
    f = friedman_summary(x)
    ref = stats.friedmanchisquare(*x.T)
    assert f["chi2"] == pytest.approx(ref.statistic) and f["p"] == pytest.approx(ref.pvalue)
    assert (f["n"], f["k"], f["df"]) == (60, 4, 3)
    assert sum(f["mean_ranks"]) == pytest.approx(4 * 5 / 2)


def test_kendall_w_concordant_and_block_shift_invariant() -> None:
    # every block ranks the columns the same way -> W = 1 and mean ranks 1..k
    x = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.7, 2.0, 9.0], [-3.0, -2.0, -1.0, 0.0]])
    f = friedman_summary(x)
    assert f["kendall_w"] == pytest.approx(1.0) and f["mean_ranks"] == [1.0, 2.0, 3.0, 4.0]
    # adding a per-block constant (Delta -> D = Delta - Delta_Base) leaves the statistic unchanged
    y = np.random.default_rng(4).normal(size=(50, 4))
    shift = np.random.default_rng(5).normal(size=(50, 1)) * 10
    assert friedman_summary(y + shift)["chi2"] == pytest.approx(friedman_summary(y)["chi2"])


def test_bonferroni_scales_and_caps() -> None:
    assert bonferroni(np.array([0.001, 0.01, 0.5]), 6).tolist() == pytest.approx([0.006, 0.06, 1.0])


def test_kruskal_matches_scipy_and_epsilon2_hand_value() -> None:
    rng = np.random.default_rng(6)
    g = [rng.normal(m, 1.0, size=30) for m in (0.0, 0.3, 0.8)]
    k = kruskal_summary(g)
    ref = stats.kruskal(*g)
    assert k["H"] == pytest.approx(ref.statistic) and k["p"] == pytest.approx(ref.pvalue)
    assert (k["n"], k["k"], k["df"]) == (90, 3, 2)
    assert k["epsilon2"] == pytest.approx(ref.statistic / 89)


def test_adjusted_means_equal_raw_means_when_covariate_balanced() -> None:
    # every group has the same covariate values (and the same 0/1 split), so the adjustment moves nothing
    rng = np.random.default_rng(7)
    x, z = np.tile(np.linspace(0, 1, 20), 3), np.tile(np.r_[np.zeros(10), np.ones(10)], 3)
    grp = np.repeat(["a", "b", "c"], 20)
    y = np.repeat([0.0, 0.5, 1.2], 20) - 0.7 * x + 0.3 * z + rng.normal(0, 0.1, 60)
    df = pd.DataFrame(dict(y=y, grp=grp, x=x, z=z))
    a = ancova_category(df, "y", "grp", ["x", "z"])
    for lev in "abc":
        assert a["adjusted_means"][lev] == pytest.approx(a["raw_means"][lev], abs=1e-12)
    assert a["coefs"].loc["x", "coef"] == pytest.approx(-0.7, abs=0.1)


def test_interaction_nested_f_matches_anova_lm_and_detects_planted_slope() -> None:
    rng = np.random.default_rng(8)
    x = rng.uniform(0, 1, 150)
    grp = np.repeat(["a", "b", "c"], 50)
    slope = np.where(grp == "c", 2.0, -0.5)          # planted: group c has a different slope
    df = pd.DataFrame(dict(y=slope * x + rng.normal(0, 0.2, 150), grp=grp, x=x))
    a = ancova_category(df, "y", "grp", ["x"], slope_var="x")
    ref = anova_lm(smf.ols("y ~ C(grp) + x", df).fit(), smf.ols("y ~ C(grp) + x + C(grp):x", df).fit()).iloc[1]
    assert a["interaction_F"] == pytest.approx(ref["F"]) and a["interaction_p"] == pytest.approx(ref["Pr(>F)"])
    assert a["interaction_df"] == 2 and a["interaction_p"] < 1e-6
    assert a["group_slopes"]["c"] == pytest.approx(2.0, abs=0.2) and a["group_slopes"]["a"] == pytest.approx(-0.5, abs=0.2)
