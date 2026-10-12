"""Explicit input construction: RAW for Base, CHAT for the aligned checkpoints.

Never relies on tokenizer defaults: every encode uses `add_special_tokens=False` and BOS is inserted by hand.
"""
from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

BOS_ID = 0  # <|endoftext|> in all six tokenizers (checked: identical vocabularies)


@dataclass(frozen=True)
class Encoded:
    """Token ids of context + word. The word occupies ids[word_start:]."""
    ids: tuple[int, ...]
    word_start: int

    @property
    def n_word_tokens(self) -> int:
        return len(self.ids) - self.word_start


def context_text(prefix: str, fmt: str, chat_cfg: dict | None = None) -> str:
    """The text before the word: the prefix itself (RAW) or the TuluV2 frame + prefix (CHAT)."""
    if fmt == "raw":
        return prefix
    if fmt == "chat":
        return chat_cfg["template"].format(instruction=chat_cfg["instruction"], prefix=prefix)
    raise ValueError(f"unknown input format: {fmt}")


def encode_pair(tok: PreTrainedTokenizerBase, context: str, word: str, bos: bool) -> Encoded:
    """Ids of `[BOS] + context + " " + word`, with the word boundary checked against the full string.

    The word is tokenized alone as `" " + word`. The concatenation must equal the tokenization of the
    full string, otherwise BPE merged across the boundary and the word position would be ambiguous.
    """
    if context.endswith(" "):
        raise ValueError("context must not end with a space (the word carries its own leading space)")
    head = ([BOS_ID] if bos else []) + tok.encode(context, add_special_tokens=False)
    word_ids = tok.encode(" " + word, add_special_tokens=False)
    full = ([BOS_ID] if bos else []) + tok.encode(context + " " + word, add_special_tokens=False)
    if head + word_ids != full:
        raise ValueError(f"word boundary mismatch for {word!r}")
    return Encoded(tuple(full), len(head))
