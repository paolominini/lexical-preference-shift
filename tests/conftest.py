"""Shared fixtures: real tokenizer and data, a tiny random GPT-NeoX (float32, CPU) for fast logic tests."""
from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

from probing.config import is_aligned, load_config, local_dir, uses_bos
from probing.inputs import Encoded, context_text, encode_pair
from probing.pools import build_pairs, load_prefixes, load_vocabulary


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def tok(cfg):
    """Base tokenizer (identical vocabulary and word ids in all six)."""
    return AutoTokenizer.from_pretrained(local_dir(cfg["models"][cfg["run"]["size"]]["base"]), local_files_only=True)


@pytest.fixture(scope="session")
def prefixes():
    return load_prefixes()


@pytest.fixture(scope="session")
def vocab():
    return load_vocabulary()


@pytest.fixture(scope="session")
def pairs(prefixes, vocab):
    return build_pairs(prefixes, vocab)


@pytest.fixture(scope="session")
def tiny_model():
    """2-layer GPT-NeoX with random weights (fixed seed), same vocab size as Pythia (50,304)."""
    torch.manual_seed(0)
    conf = GPTNeoXConfig(vocab_size=50304, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         intermediate_size=128, max_position_embeddings=256, initializer_range=0.2)
    return GPTNeoXForCausalLM(conf).to(torch.float32).eval()


def sample_encs(tok, cfg, pairs, n_prefixes_per_cat: int = 1) -> list[Encoded]:
    """Encodings of real pairs in both formats: mixed lengths and 1-4 token words."""
    seen: dict[str, set] = {}
    out = []
    for p in pairs:
        ids = seen.setdefault(p.category, set())
        if p.prefix_id not in ids and len(ids) >= n_prefixes_per_cat:
            continue
        ids.add(p.prefix_id)
        for key in ("base", "sft"):
            fmt = "chat" if is_aligned(key) else "raw"
            out.append(encode_pair(tok, context_text(p.prefix, fmt, cfg["chat"]), p.word, bos=uses_bos(key, cfg)))
    return out


@pytest.fixture(scope="session")
def encs(tok, cfg, pairs) -> list[Encoded]:
    return sample_encs(tok, cfg, pairs)
