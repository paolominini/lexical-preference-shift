"""Tests for the derived quantities on a tiny hand-made table."""
from __future__ import annotations

import pandas as pd
import pytest

from probing.derived import derived_vs_base, identity_error, pool_means


def _scores() -> pd.DataFrame:
    rows = []
    # one prefix, two neutral words (a, b), two derogatory words (x, y)
    vals = {"base": {"a": -10.0, "b": -12.0, "x": -14.0, "y": -16.0}, "m": {"a": -11.0, "b": -13.0, "x": -18.0, "y": -20.0}}
    for model, words in vals.items():
        for w, lp in words.items():
            rows.append(dict(prefix_id="P0", category="C", model=model, word=w,
                             polarity_word="SAFE" if w in "ab" else "NOT SAFE", logp=lp))
    return pd.DataFrame(rows)


def test_hand_computed_values() -> None:
    s = _scores()
    pm = pool_means(s).set_index("model")
    assert pm.loc["base", "delta"] == pytest.approx(-11.0 - (-15.0))      # 4
    assert pm.loc["m", "delta"] == pytest.approx(-12.0 - (-19.0))         # 7
    d = derived_vs_base(s, s[s["model"] == "base"]).iloc[0]
    assert d["delta_pos"] == pytest.approx(-1.0)
    assert d["delta_neg"] == pytest.approx(-4.0)
    assert d["D"] == pytest.approx(3.0)
    assert identity_error(derived_vs_base(s, s[s["model"] == "base"])) < 1e-12
