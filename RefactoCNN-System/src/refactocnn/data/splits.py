from __future__ import annotations
from typing import List, Tuple
import random

def stratified_split_indices(labels: List[int], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified split for binary/multi-class labels.
    Keeps class proportions roughly stable across train/val/test.

    Returns: (train_idx, val_idx, test_idx)
    """
    rng = random.Random(seed)
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)
    for y in by_class:
        rng.shuffle(by_class[y])

    train_idx, val_idx, test_idx = [], [], []
    for y, idxs in by_class.items():
        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train+n_val])
        test_idx.extend(idxs[n_train+n_val:])
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx
