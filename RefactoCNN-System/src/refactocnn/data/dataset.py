from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset

@dataclass
class SegmentItem:
    x: torch.Tensor
    y: int
    meta: Dict[str, Any]

class SegmentDataset(Dataset):
    def __init__(self, items: List[SegmentItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        return {"x": it.x, "y": int(it.y), "meta": it.meta}
