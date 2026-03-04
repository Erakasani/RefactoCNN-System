from __future__ import annotations
import torch
import torch.nn as nn

def make_loss(class_weight):
    if class_weight is None:
        return nn.CrossEntropyLoss()
    w = torch.tensor(class_weight, dtype=torch.float)
    return nn.CrossEntropyLoss(weight=w)
