from __future__ import annotations
import torch

def fuse(token_vec: torch.Tensor, ast_vec: torch.Tensor | None) -> torch.Tensor:
    if ast_vec is None:
        return token_vec
    return torch.cat([token_vec, ast_vec], dim=-1)
