from __future__ import annotations
import torch

def make_plateau_scheduler(optimizer, factor: float = 0.5, patience: int = 3):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience, verbose=True
    )
