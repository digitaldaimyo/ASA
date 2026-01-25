"""Simple tokenization helpers for offline demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class SimpleTokenizer:
    vocab: Dict[str, int]
    inv_vocab: Dict[int, str]
    unk_token: str = "<unk>"

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "SimpleTokenizer":
        chars = sorted({ch for text in texts for ch in text})
        vocab = {"<unk>": 0, "<pad>": 1}
        for ch in chars:
            if ch not in vocab:
                vocab[ch] = len(vocab)
        inv_vocab = {idx: tok for tok, idx in vocab.items()}
        return cls(vocab=vocab, inv_vocab=inv_vocab)

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(ch, self.vocab[self.unk_token]) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.inv_vocab.get(i, self.unk_token) for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
