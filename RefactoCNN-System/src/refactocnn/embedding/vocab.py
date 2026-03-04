from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List
from ..constants import SPECIAL_TOKENS

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, sequences: Iterable[List[str]], min_freq: int = 1) -> "Vocab":
        freq: Dict[str, int] = {}
        for seq in sequences:
            for t in seq:
                freq[t] = freq.get(t, 0) + 1
        itos = [SPECIAL_TOKENS["PAD"], SPECIAL_TOKENS["UNK"]]
        for t, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= min_freq and t not in itos:
                itos.append(t)
        stoi = {t:i for i,t in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, seq: List[str]) -> List[int]:
        unk = self.stoi[SPECIAL_TOKENS["UNK"]]
        return [self.stoi.get(t, unk) for t in seq]

    @property
    def pad_id(self) -> int:
        return self.stoi[SPECIAL_TOKENS["PAD"]]
