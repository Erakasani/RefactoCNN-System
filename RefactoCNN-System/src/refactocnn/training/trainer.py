from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from ..utils.metrics import compute_binary_metrics


@dataclass
class TrainResult:
    best_epoch: int
    best_val_f1: float
    best_state_dict: Dict[str, torch.Tensor]


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    optimizer,
    scheduler,
    device: str,
    epochs: int,
    early_stop_patience: int,
    tb_writer=None,
) -> TrainResult:
    """Train with early stopping on validation F1.

    Returns the best epoch, best val F1, and an in-memory copy of the best weights.
    """

    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * x.size(0)

        train_loss = total_loss / max(1, len(train_loader.dataset))

        val = evaluate_model(model, val_loader, device)
        val_loss = val["loss"]
        scheduler.step(val_loss)

        if tb_writer:
            tb_writer.add_scalar("loss/train", train_loss, epoch)
            tb_writer.add_scalar("loss/val", val_loss, epoch)
            tb_writer.add_scalar("metrics/val_f1", val["metrics"]["f1"], epoch)
            tb_writer.add_scalar("metrics/val_mcc", val["metrics"]["mcc"], epoch)

        cur_f1 = float(val["metrics"]["f1"])
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return TrainResult(best_epoch=best_epoch, best_val_f1=best_f1, best_state_dict=best_state)


@torch.no_grad()
def evaluate_model(model, loader: DataLoader, device: str) -> Dict[str, Any]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    ys, preds = [], []
    total_loss = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        p = torch.argmax(logits, dim=-1)
        ys.extend(y.cpu().tolist())
        preds.extend(p.cpu().tolist())

    if len(ys):
        metrics = compute_binary_metrics(ys, preds)
    else:
        metrics = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "mcc": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    return {"loss": total_loss / max(1, len(loader.dataset)), "metrics": metrics, "y_true": ys, "y_pred": preds}
