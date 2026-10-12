"""Tokenization audit of the six 1.4B checkpoints.

Loads only tokenizers and JSON configs (no weights). Writes the report to reports/step_01_tokenization.md and the
per-word / per-checkpoint tables to reports/step_01_tables/. Prints at most ~20 lines on screen.

Usage: .venv/bin/python scripts/01_tokenization_audit.py --size 1.4b
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer, PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parents[1]
NEUTRAL = "The quick brown fox"
ROLE_TOKENS = ["<|user|>", "<|assistant|>"]
CFG_KEYS = ["tokenizer_class", "add_bos_token", "add_eos_token", "bos_token", "eos_token", "pad_token",
            "padding_side", "model_max_length", "clean_up_tokenization_spaces"]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def local_dir(repo: str, models_dir: str | None) -> Path:
    """Newest local snapshot folder of a repo in the Hugging Face cache (never downloads)."""
    from huggingface_hub import scan_cache_dir
    kw = {"cache_dir": models_dir} if models_dir else {}
    for r in scan_cache_dir(**kw).repos:
        if r.repo_id == repo:
            return Path(max(r.revisions, key=lambda v: v.last_modified).snapshot_path)
    raise FileNotFoundError(f"{repo} not found in the local cache")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def vocab_hash(tok: PreTrainedTokenizerBase) -> str:
    items = sorted(tok.get_vocab().items())
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:12]


def build_ids(tok: PreTrainedTokenizerBase, text: str, bos: bool) -> list[int]:
    """Explicit input ids: no reliance on tokenizer defaults."""
    ids = tok.encode(text, add_special_tokens=False)
    return ([tok.bos_token_id] if bos else []) + ids


def chat_text(cfg: dict, prefix: str) -> str:
    return cfg["chat"]["template"].format(instruction=cfg["chat"]["instruction"], prefix=prefix)


def load_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    prefixes = pd.read_csv(ROOT / "data/final_prefix_set.csv")
    vocab = pd.read_excel(ROOT / "data/Vocabulary_Final.xlsx", sheet_name="Final_Probing_Set")
    return prefixes, vocab


def audit_checkpoint(key: str, repo: str, cfg: dict, prefixes: pd.DataFrame, vocab: pd.DataFrame) -> dict:
    d = local_dir(repo, cfg["paths"]["models_dir"])
    tcfg, mcfg = read_json(d / "tokenizer_config.json"), read_json(d / "config.json")
    tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
    aligned = key != "base"
    default_ids = tok(NEUTRAL)["input_ids"]
    raw_ids = tok(NEUTRAL, add_special_tokens=False)["input_ids"]
    roles = {r: tok.encode(r, add_special_tokens=False) for r in ROLE_TOKENS}
    word_ids = {w: tok.encode(" " + w, add_special_tokens=False) for w in vocab.WORD.unique()}
    # round trip: full string (prefix + " " + word) must end with the ids of " word" alone
    bad, n = [], 0
    for cat, grp in prefixes.groupby("category"):
        pool = vocab[vocab.CATEGORY.str.strip() == cat.strip()].WORD.unique()
        for pid, pfx in zip(grp.final_id, grp.prefix):
            for w in pool:
                full = build_ids(tok, pfx + " " + w, bos=aligned)
                n += 1
                if full[-len(word_ids[w]):] != word_ids[w]:
                    bad.append((pid, w))
    return dict(
        key=key, repo=repo, dir=str(d), tok=tok,
        cfg={k: tcfg.get(k) for k in CFG_KEYS},
        model_cfg={k: mcfg.get(k) for k in ["architectures", "dtype", "torch_dtype", "vocab_size", "bos_token_id",
                                            "eos_token_id"]},
        default_ids=default_ids, raw_ids=raw_ids, bos_default=default_ids[:1] == [tok.bos_token_id] and
        len(default_ids) > len(raw_ids),
        roles=roles, roles_special={r: r in tok.all_special_tokens for r in ROLE_TOKENS},
        vocab_size=len(tok), vocab_hash=vocab_hash(tok), word_ids=word_ids, roundtrip_bad=bad, roundtrip_n=n,
    )


def show_tokens(tok: PreTrainedTokenizerBase, ids: list[int]) -> str:
    return " ".join(f"{i}:{tok.convert_ids_to_tokens(i)!r}" for i in ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1.4b")
    args = ap.parse_args()
    cfg = load_config()
    prefixes, vocab = load_pairs()
    res = {k: audit_checkpoint(k, r, cfg, prefixes, vocab) for k, r in cfg["models"][args.size].items()}
    keys = list(res)

    # tables (files, not screen)
    tdir = ROOT / "reports/step_01_tables"
    tdir.mkdir(exist_ok=True)
    words = pd.DataFrame({k: {w: json.dumps(v) for w, v in res[k]["word_ids"].items()} for k in keys})
    words["identical_across_six"] = words.nunique(axis=1) == 1
    words["n_tokens"] = [len(res["base"]["word_ids"][w]) for w in words.index]
    words.index.name = "word"
    words.to_csv(tdir / "word_token_ids.csv")
    cfgtab = pd.DataFrame({k: {**res[k]["cfg"], **{f"config.{a}": b for a, b in res[k]["model_cfg"].items()}}
                           for k in keys})
    cfgtab.to_csv(tdir / "tokenizer_and_model_config.csv")

    # one worked example (first prefix of the first category), RAW for base, CHAT + explicit BOS for aligned
    ex = prefixes.iloc[0]
    ex_txt = {k: (ex.prefix if k == "base" else chat_text(cfg, ex.prefix)) for k in keys}
    example = {k: build_ids(res[k]["tok"], ex_txt[k] + " " + "WORD", bos=(k != "base")) for k in keys}

    all_words_identical = bool(words.identical_across_six.all())
    vocab_same = len({res[k]["vocab_hash"] for k in keys}) == 1
    rt_bad = {k: len(res[k]["roundtrip_bad"]) for k in keys}
    ntok = words.n_tokens.value_counts().sort_index().to_dict()

    L = ["# Tokenization audit (" + args.size + ")", "",
         "Tokenizers and JSON configs only (no weights). Tables: `reports/step_01_tables/`.", "",
         "## A. Tokenizer defaults (transformers behaviour, not a policy)", "",
         "| key | add_bos_token (config) | bos added by default | default ids of neutral string | vocab | vocab hash |",
         "|---|---|---|---|---|---|"]
    for k in keys:
        r = res[k]
        L.append(f"| {k} | {r['cfg']['add_bos_token']} | {r['bos_default']} | {r['default_ids']} | "
                 f"{r['vocab_size']} | {r['vocab_hash']} |")
    L += ["", "## B. Role tokens", "", "| key | `<|user|>` ids | `<|assistant|>` ids | special tokens? |", "|---|---|---|---|"]
    for k in keys:
        r = res[k]
        L.append(f"| {k} | {r['roles']['<|user|>']} | {r['roles']['<|assistant|>']} | "
                 f"{any(r['roles_special'].values())} |")
    L += ["", "## C. config.json / tokenizer_config.json (stored dtype etc.)", "",
          "| key | tokenizer_class | pad_token | stored dtype (`dtype` or `torch_dtype`) | config vocab_size | bos_token_id | eos_token_id |",
          "|---|---|---|---|---|---|---|"]
    for k in keys:
        r, m = res[k]["cfg"], res[k]["model_cfg"]
        L.append(f"| {k} | {r['tokenizer_class']} | {r['pad_token']} | {m['dtype'] or m['torch_dtype']} | {m['vocab_size']} | "
                 f"{m['bos_token_id']} | {m['eos_token_id']} |")
    L += ["", "## D. Probe words", "",
          f"- 41 vocabulary rows, {len(words)} distinct words (queer and homosexual are in two pools).",
          f"- Token ids of `\" word\"` identical across the six tokenizers: **{all_words_identical}**.",
          f"- Number of tokens per word (n_tokens: n_words): {ntok}.",
          f"- Round-trip (full string `prefix + \" \" + word` ends with the ids of `\" word\"` alone), "
          f"{res['base']['roundtrip_n']} pairs per checkpoint, failures: {rt_bad}.",
          "", "## E. Worked example (explicit input; aligned start with BOS, Base without)", "",
          f"Prefix {ex.final_id}; `WORD` is a placeholder here.", ""]
    for k in (keys[0], keys[1]):
        L += [f"**{k}** ({'RAW' if k == 'base' else 'CHAT'}), {len(example[k])} tokens:", "```",
              show_tokens(res[k]["tok"], example[k]), "```"]
    L += ["", "## F. Reading", "",
          "- Same vocabulary, same word ids in all six tokenizers, so the only input differences are BOS and the CHAT frame.",
          "- Role markers are plain text split into several ordinary tokens (see B), not special tokens.",
          "", "## G. Key numbers and notes", "",
          f"1. Vocabulary: {sorted({res[k]['vocab_size'] for k in keys})} tokens, same hash in all six: {vocab_same}.",
          f"2. BOS added by default: {[k for k in keys if res[k]['bos_default']] or 'none'}.",
          f"3. Words with identical ids in the six tokenizers: {int(words.identical_across_six.sum())}/{len(words)}.",
          f"4. Failed round-trips: {sum(rt_bad.values())} of {res['base']['roundtrip_n'] * len(keys)} checks.",
          f"5. Stored dtype in the configs: {sorted({str(res[k]['model_cfg']['dtype'] or res[k]['model_cfg']['torch_dtype']) for k in keys})}.",
          "",
          "- The configs store the dtype as `torch_dtype` (not `dtype`): Base float16, the five aligned checkpoints bfloat16.",
          "- `add_bos_token` is absent from all six tokenizer configs: there is no default to rely on, so BOS is added explicitly.",
          "- Base has no `pad_token`; the aligned checkpoints pad with `<|endoftext|>` (id 0, same as BOS/EOS): padding is built by hand.",
          "- config.vocab_size = 50304 but the tokenizer has 50277 tokens: 27 unused embedding rows enter the softmax denominator, identically in the six models.",
          "- The CHAT frame adds 18 tokens before the prefix, 19 with BOS (4 for `<|user|>`, 5 for `<|assistant|>`, plus instruction and newlines)."]
    (ROOT / "reports/step_01_tokenization.md").write_text("\n".join(L) + "\n")

    print(f"vocab same: {vocab_same} | words identical: {int(words.identical_across_six.sum())}/{len(words)} "
          f"| round-trip failures: {rt_bad}")
    for k in keys:
        r = res[k]
        print(f"{k:8s} add_bos_token={r['cfg']['add_bos_token']} bos_default={r['bos_default']} "
              f"dtype={r['model_cfg']['dtype'] or r['model_cfg']['torch_dtype']} user_ids={r['roles']['<|user|>']}")
    print("n_tokens per word:", ntok)


if __name__ == "__main__":
    main()
