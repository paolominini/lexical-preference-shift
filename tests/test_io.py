"""Test 9: resumable writer (never overwrite, resume skips done pairs)."""
import pandas as pd
import pytest

from probing.io import COLUMNS, PartWriter, read_parts


def row(pid, word, logp=-5.0):
    return dict(prefix_id=pid, category="c", model="base", model_repo="r", word=word, polarity_word="SAFE",
                n_tokens=2, logp=logp, input_format="raw", bos=False, dtype="float16")


def test_write_and_resume(tmp_path):
    w = PartWriter(tmp_path)
    w.write([row("A", "x"), row("A", "y")])
    w2 = PartWriter(tmp_path)  # a new run resumes
    assert w2.is_done("A", "base", "x") and not w2.is_done("B", "base", "x")
    p = w2.write([row("B", "x")])
    assert p.name == "part_0001.parquet"
    df = read_parts(tmp_path)
    assert list(df.columns) == COLUMNS and len(df) == 3


def test_refuses_duplicates(tmp_path):
    w = PartWriter(tmp_path)
    w.write([row("A", "x")])
    with pytest.raises(ValueError):
        PartWriter(tmp_path).write([row("A", "x", -1.0)])
    with pytest.raises(ValueError):
        w.write([row("C", "x"), row("C", "x")])
    assert len(read_parts(tmp_path)) == 1


def test_refuses_wrong_columns(tmp_path):
    with pytest.raises(ValueError):
        PartWriter(tmp_path).write([{"prefix_id": "A"}])


def test_never_overwrites_file(tmp_path):
    w = PartWriter(tmp_path)
    (tmp_path / "part_0000.parquet").write_bytes(b"")  # simulate a file appearing on disk
    w._next_path = lambda: tmp_path / "part_0000.parquet"
    with pytest.raises(FileExistsError):
        w.write([row("A", "x")])


def test_empty_dir_schema(tmp_path):
    assert list(read_parts(tmp_path).columns) == COLUMNS
    assert isinstance(read_parts(tmp_path), pd.DataFrame)
