"""Resumable run loop: interruption + resume gives the same rows as one uninterrupted run."""
import pandas as pd

from probing.io import PartWriter, read_parts
from probing.run import run_checkpoint


def _pairs(pairs, n_prefixes=3):
    ids = list(dict.fromkeys(p.prefix_id for p in pairs))[:n_prefixes]
    return [p for p in pairs if p.prefix_id in ids]


def test_resume_equals_single_run(tmp_path, tiny_model, tok, cfg, pairs):
    sub = _pairs(pairs)
    a = PartWriter(tmp_path / "a")
    s1 = run_checkpoint(tiny_model, tok, sub, "sft", "repo", cfg, a, bos=True, dtype="float32", every=1,
                        max_prefixes=1)
    s2 = run_checkpoint(tiny_model, tok, sub, "sft", "repo", cfg, PartWriter(a.out_dir), bos=True, dtype="float32")
    s3 = run_checkpoint(tiny_model, tok, sub, "sft", "repo", cfg, PartWriter(a.out_dir), bos=True, dtype="float32")
    assert s1.n_scored + s2.n_scored == len(sub) and s2.n_skipped == s1.n_scored
    assert s3.n_scored == 0 and s3.n_skipped == len(sub)
    b = PartWriter(tmp_path / "b")
    run_checkpoint(tiny_model, tok, sub, "sft", "repo", cfg, b, bos=True, dtype="float32")
    key = ["prefix_id", "word"]
    ra = read_parts(a.out_dir).sort_values(key).reset_index(drop=True)
    rb = read_parts(b.out_dir).sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(ra, rb)
    assert len(ra) == len(sub) and not ra.duplicated(key).any()


def test_format_and_bos_recorded(tmp_path, tiny_model, tok, cfg, pairs):
    sub = _pairs(pairs, 1)
    w = PartWriter(tmp_path / "c")
    run_checkpoint(tiny_model, tok, sub, "base", "repo", cfg, w, bos=False, dtype="float32")
    df = read_parts(w.out_dir)
    assert set(df["input_format"]) == {"raw"} and set(df["bos"]) == {False} and set(df["model"]) == {"base"}
