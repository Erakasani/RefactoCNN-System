from __future__ import annotations
import torch

def pad_truncate(ids: list[int], max_len: int, pad_id: int) -> list[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))

def mean_pool(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # emb: [B,L,D], mask: [B,L] 1 for real tokens
    mask = mask.unsqueeze(-1)  # [B,L,1]
    summed = (emb * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom
