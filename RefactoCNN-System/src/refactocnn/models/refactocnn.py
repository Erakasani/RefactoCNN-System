from __future__ import annotations
import torch
import torch.nn as nn
from .init import he_init

class RefactoCNN(nn.Module):
    """CNN classifier over a fixed-length vector, treated as a 1D signal of length D.
    Input: x [B, D]
    Reshape: [B, 1, D]
    """
    def __init__(self, input_dim: int, conv1_filters: int = 64, conv2_filters: int = 128,
                 kernel_size: int = 3, dropout: float = 0.5, num_classes: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, conv1_filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(conv1_filters, conv2_filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # compute flattened size: conv2_filters * (input_dim//2)
        flat = conv2_filters * (input_dim // 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )
        he_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B,1,D]
        h = self.net(x)
        out = self.fc(h)
        return out
