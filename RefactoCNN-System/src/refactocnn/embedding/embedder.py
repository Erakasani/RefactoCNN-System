from __future__ import annotations
import torch
import torch.nn as nn

class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, L]
        return self.emb(ids)  # [B, L, D]
