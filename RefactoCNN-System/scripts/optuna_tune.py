from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List, Tuple, Optional

import optuna
import torch
import yaml
from torch.utils.data import DataLoader

from src.refactocnn.data.dataset import SegmentDataset, SegmentItem
from src.refactocnn.data.splits import stratified_split_indices
from src.refactocnn.data.collate import collate_batch
from src.refactocnn.embedding.embedder import TokenEmbedder
from src.refactocnn.embedding.pooling import mean_pool
from src.refactocnn.models.refactocnn import RefactoCNN
from src.refactocnn.training.checkpoints import save_checkpoint
from src.refactocnn.training.losses import make_loss
from src.refactocnn.training.scheduler import make_plateau_scheduler
from src.refactocnn.training.trainer import evaluate_model
from src.refactocnn.utils.logging import save_json
from src.refactocnn.utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = idx[:n_train]
    val = idx[n_train:n_train + n_val]
    test = idx[n_train + n_val:]
    return train, val, test


def compute_class_weight(labels: List[int]) -> Optional[List[float]]:
    import numpy as np
    labels = np.asarray(labels)
    c0 = int((labels == 0).sum())
    c1 = int((labels == 1).sum())
    if c0 == 0 or c1 == 0:
        return None
    w0 = (c0 + c1) / (2.0 * c0)
    w1 = (c0 + c1) / (2.0 * c1)
    return [float(w0), float(w1)]


def suggest_from_space(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "loguniform":
            out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
        elif t == "uniform":
            out[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=False)
        elif t == "categorical":
            out[name] = trial.suggest_categorical(name, list(spec["choices"]))
        elif t == "int":
            out[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        else:
            raise ValueError(f"Unknown search space type: {t} for {name}")
    return out


def build_items(cfg: dict, device: str) -> Tuple[List[SegmentItem], List[int], List[int], List[int]]:
    """Build pooled feature vectors once to speed up tuning.

    Note: This baseline follows the paper's 'embedding generation' stage by producing fixed vectors
    (token-only or token+AST fusion). The CNN then operates on these fixed representations.
    """
    pack = torch.load(cfg["paths"]["features_pt"], map_location="cpu")
    encoded = pack["encoded"]
    vocab_tok = pack["vocab_tok"]
    vocab_ast = pack["vocab_ast"]
    fuse_ast = bool(cfg["data"].get("fuse_ast", True))
    embed_dim = int(cfg["data"].get("embed_dim", 64))

    # deterministic embed init for fair tuning
    torch.manual_seed(int(cfg["data"].get("random_seed", 42)))
    emb_tok = TokenEmbedder(len(vocab_tok.itos), embed_dim, vocab_tok.pad_id).to(device)
    emb_ast = TokenEmbedder(len(vocab_ast.itos), embed_dim, vocab_ast.pad_id).to(device)

    items: List[SegmentItem] = []
    for row in encoded:
        tok = torch.tensor(row["tok_ids"], dtype=torch.long).unsqueeze(0).to(device)
        tok_mask = (tok != vocab_tok.pad_id).float()
        tok_vec = mean_pool(emb_tok(tok), tok_mask).squeeze(0).detach().cpu()

        if fuse_ast:
            ast = torch.tensor(row["ast_ids"], dtype=torch.long).unsqueeze(0).to(device)
            ast_mask = (ast != vocab_ast.pad_id).float()
            ast_vec = mean_pool(emb_ast(ast), ast_mask).squeeze(0).detach().cpu()
            x = torch.cat([tok_vec, ast_vec], dim=-1)
        else:
            x = tok_vec

        items.append(SegmentItem(x=x, y=int(row["label"]), meta=row.get("meta", {})))

    train_idx, val_idx, test_idx = split_indices(
        len(items), float(cfg["data"]["train_ratio"]), float(cfg["data"]["val_ratio"]), int(cfg["data"]["random_seed"])
    )
    return items, train_idx, val_idx, test_idx


def train_one_trial(
    trial: optuna.Trial,
    cfg: dict,
    params: dict,
    items: List[SegmentItem],
    train_idx: List[int],
    val_idx: List[int],
    device: str,
) -> float:
    bs = int(params["batch_size"])
    train_ds = SegmentDataset([items[i] for i in train_idx])
    val_ds = SegmentDataset([items[i] for i in val_idx])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_batch)

    input_dim = train_ds[0]["x"].numel()
    model = RefactoCNN(
        input_dim=input_dim,
        conv1_filters=int(params["conv1_filters"]),
        conv2_filters=int(params["conv2_filters"]),
        kernel_size=int(params["kernel_size"]),
        dropout=float(params["dropout"]),
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = make_plateau_scheduler(optimizer, factor=0.5, patience=3)

    cw = None
    if params.get("class_weight") == "auto":
        cw = compute_class_weight([items[i].y for i in train_idx])
    loss_fn = make_loss(cw)

    best_f1 = -1.0
    bad_epochs = 0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        val = evaluate_model(model, val_loader, device)
        scheduler.step(val["loss"])
        cur_f1 = float(val["metrics"]["f1"])

        trial.report(cur_f1, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["train"]["early_stop_patience"]):
                break

    return best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/optuna_search.yaml")
    ap.add_argument("--storage", default=None, help="Optuna storage URL (e.g., sqlite:///optuna.db). If omitted, uses in-memory.")
    ap.add_argument("--study", default=None, help="Override study name")
    ap.add_argument("--trials", type=int, default=None, help="Override number of trials")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["data"]["random_seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items, train_idx, val_idx, test_idx = build_items(cfg, device=device)

    opt_cfg = cfg["optuna"]
    study_name = args.study or opt_cfg.get("study_name", "refactocnn_optuna")
    n_trials = args.trials or int(opt_cfg.get("n_trials", 30))

    sampler_name = str(opt_cfg.get("sampler", "tpe")).lower()
    pruner_name = str(opt_cfg.get("pruner", "median")).lower()

    sampler = optuna.samplers.TPESampler(seed=int(cfg["data"]["random_seed"])) if sampler_name == "tpe" else optuna.samplers.RandomSampler(seed=int(cfg["data"]["random_seed"]))
    pruner = optuna.pruners.MedianPruner() if pruner_name == "median" else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        direction=str(opt_cfg.get("direction", "maximize")),
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    space = cfg["search_space"]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_from_space(trial, space)
        return train_one_trial(trial, cfg, params, items, train_idx, val_idx, device)

    study.optimize(objective, n_trials=n_trials, timeout=opt_cfg.get("timeout_sec", None))

    best_params = dict(study.best_trial.params)

    # Re-train on train+val, evaluate on test
    combined_idx = list(train_idx) + list(val_idx)
    bs = int(best_params.get("batch_size", 64))

    train_ds = SegmentDataset([items[i] for i in combined_idx])
    test_ds = SegmentDataset([items[i] for i in test_idx])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_batch)

    input_dim = train_ds[0]["x"].numel()
    model = RefactoCNN(
        input_dim=input_dim,
        conv1_filters=int(best_params.get("conv1_filters", 64)),
        conv2_filters=int(best_params.get("conv2_filters", 128)),
        kernel_size=int(best_params.get("kernel_size", 3)),
        dropout=float(best_params.get("dropout", 0.5)),
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(best_params.get("lr", 1e-3)),
        weight_decay=float(best_params.get("weight_decay", 0.0)),
    )
    scheduler = make_plateau_scheduler(optimizer, factor=0.5, patience=3)

    cw = None
    if best_params.get("class_weight") == "auto":
        cw = compute_class_weight([items[i].y for i in combined_idx])
    loss_fn = make_loss(cw)

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        probe = evaluate_model(model, test_loader, device)
        scheduler.step(probe["loss"])
        cur_f1 = float(probe["metrics"]["f1"])

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["train"]["early_stop_patience"]):
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["reports_dir"], exist_ok=True)

    ckpt_path = os.path.join(cfg["paths"]["models_dir"], "best_optuna.pt")
    save_checkpoint(ckpt_path, model, optimizer, epoch=-1, best_metric=best_f1, config={"base": cfg, "best_params": best_params})

    final_test = evaluate_model(model, test_loader, device)

    save_json(os.path.join(cfg["paths"]["reports_dir"], "optuna_best_params.json"), best_params)
    save_json(os.path.join(cfg["paths"]["reports_dir"], "optuna_test_metrics.json"), final_test["metrics"])

    print("Study:", study_name)
    print("Best value (val F1):", study.best_value)
    print("Best params:", best_params)
    print("Saved checkpoint:", ckpt_path)
    print("Test metrics:", final_test["metrics"])


if __name__ == "__main__":
    main()
