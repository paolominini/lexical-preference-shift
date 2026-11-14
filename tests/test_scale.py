"""Fast tests for the larger-size replication helpers (scale.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probing.config import ROOT
from probing.scale import category_order_rho, compare_column, h1_verdict, load_script, mean_d_gate


def _runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = pd.DataFrame({"model": ["a"] * 3 + ["b"] * 3, "prefix_id": list("xyz") * 2, "D": [0.1, 0.2, 0.3, -0.1, 0.0, 0.1]})
    new = ref.copy()
    new["D"] = new["D"] + np.array([0.01, -0.01, 0.0, 0.05, 0.05, 0.05])
    return ref, new


def test_compare_column_identical_and_shifted() -> None:
    ref, new = _runs()
    same = compare_column(ref, ref, ["model", "prefix_id"], "D")
    assert same["n"] == 6 and same["max_abs_diff"] == 0 and same["pearson_r"] == pytest.approx(1.0)
    c = compare_column(ref, new, ["model", "prefix_id"], "D")
    assert c["max_abs_diff"] == pytest.approx(0.05) and c["mean_abs_diff"] == pytest.approx(0.17 / 6)


def test_compare_column_rejects_different_keys() -> None:
    ref, new = _runs()
    with pytest.raises(ValueError):
        compare_column(ref, new.iloc[:-1], ["model", "prefix_id"], "D")


def test_mean_d_gate_tolerance() -> None:
    ref, new = _runs()
    g = mean_d_gate(ref, new, ["a", "b"], tol=0.04).set_index("model")
    assert g.loc["a", "diff"] == pytest.approx(0.0) and g.loc["a", "passed"]
    assert g.loc["b", "diff"] == pytest.approx(0.05) and not g.loc["b", "passed"]


def test_h1_verdict_needs_all_three_conditions() -> None:
    h1 = pd.DataFrame({"mean": [0.5, 0.5, 0.5, -0.5], "ci_boot_lo": [0.4, -0.1, 0.4, -0.6],
                       "wilcoxon_p_bonf5": [1e-10, 1e-10, 0.2, 1e-10]})
    assert list(h1_verdict(h1)) == ["supported", "not supported", "not supported", "not supported"]


def test_category_order_rho() -> None:
    a = pd.Series([1, 2, 3, 4, 5], index=list("abcde"))
    assert category_order_rho(a, a * 10) == pytest.approx(1.0)
    assert category_order_rho(a, -a.iloc[::-1]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        category_order_rho(a, a.rename(index={"e": "f"}))


def test_load_script_runs_no_main() -> None:
    mod = load_script(ROOT / "scripts/05_derived.py")
    assert callable(mod.build_delta_main) and callable(mod.write_new)
