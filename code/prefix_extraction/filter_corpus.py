#!/usr/bin/env python3
"""
filter_corpus.py - keeps only the rows whose text contains a vocabulary word (singular form),
so that very large corpora (Civil Comments, SBIC) shrink to a file small enough to upload.
The original file is never modified; all columns of the kept rows are preserved.
The corpora are not redistributed with this repository (download them from their original sources).

  python filter_corpus.py --input civil_comments.csv --text-col text --vocab Vocabulary_Final.xlsx --out civil_comments_filtered.csv
  python filter_corpus.py --input SBIC.v2.agg.trn.csv --text-col post --vocab Vocabulary_Final.xlsx --out sbic_trn_filtered.csv
"""
import argparse, re
import pandas as pd


def norm(w):
    return re.sub(r"[\s-]+", " ", str(w).lower().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunksize", type=int, default=200_000)
    a = ap.parse_args()

    v = pd.read_excel(a.vocab, sheet_name="Final_Probing_Set")
    words = sorted({norm(w) for w in v["WORD"]}, key=len, reverse=True)
    alt = "|".join(re.escape(w).replace(r"\ ", r"[\s-]+").replace(r"\-", r"[\s-]?") for w in words)
    pat = re.compile(r"(?<![A-Za-z0-9])(?:" + alt + r")(?![A-Za-z0-9])", re.I)

    first, kept, total = True, 0, 0
    for ch in pd.read_csv(a.input, chunksize=a.chunksize, low_memory=False):
        total += len(ch)
        m = ch[a.text_col].astype(str).str.contains(pat)
        ch[m].to_csv(a.out, mode="w" if first else "a", header=first, index=False)
        first, kept = False, kept + int(m.sum())
    print(f"{kept} of {total} rows kept -> {a.out}")


if __name__ == "__main__":
    main()
