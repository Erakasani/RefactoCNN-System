from __future__ import annotations
from typing import Any, Dict, List
import torch

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs = torch.stack([b["x"] for b in batch], dim=0)  # [B, D]
    ys = torch.tensor([b["y"] for b in batch], dtype=torch.long)
    metas = [b["meta"] for b in batch]
    return {"x": xs, "y": ys, "meta": metas}
