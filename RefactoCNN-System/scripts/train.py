from __future__ import annotations
import argparse, os, yaml, random
import torch
from torch.utils.data import DataLoader
from src.refactocnn.utils.seed import set_seed
from src.refactocnn.utils.logging import make_tb_writer, save_json
from src.refactocnn.data.dataset import SegmentDataset, SegmentItem
from src.refactocnn.data.splits import stratified_split_indices
from src.refactocnn.data.collate import collate_batch
from src.refactocnn.embedding.embedder import TokenEmbedder
from src.refactocnn.embedding.pooling import mean_pool
from src.refactocnn.models.refactocnn import RefactoCNN
from src.refactocnn.training.trainer import train_model, evaluate_model
from src.refactocnn.training.scheduler import make_plateau_scheduler
from src.refactocnn.training.losses import make_loss
from src.refactocnn.training.checkpoints import save_checkpoint

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = idx[:n_train]
    val = idx[n_train:n_train+n_val]
    test = idx[n_train+n_val:]
    return train, val, test

def load_split_files(splits_dir: str):
    """If train/val/test split files exist, load them as integer indices."""
    train_p = os.path.join(splits_dir, "train.txt")
    val_p = os.path.join(splits_dir, "val.txt")
    test_p = os.path.join(splits_dir, "test.txt")
    if not (os.path.exists(train_p) and os.path.exists(val_p) and os.path.exists(test_p)):
        return None
    def _read(p):
        with open(p, "r", encoding="utf-8") as f:
            return [int(x.strip()) for x in f if x.strip()]
    return _read(train_p), _read(val_p), _read(test_p)

def compute_class_weight(labels):
    # returns [w0,w1] inversely proportional to frequency
    import numpy as np
    labels = np.asarray(labels)
    c0 = (labels == 0).sum()
    c1 = (labels == 1).sum()
    if c0 == 0 or c1 == 0:
        return None
    w0 = (c0 + c1) / (2.0 * c0)
    w1 = (c0 + c1) / (2.0 * c1)
    return [float(w0), float(w1)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["data"]["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pack = torch.load(cfg["paths"]["features_pt"], map_location="cpu")
    encoded = pack["encoded"]
    vocab_tok = pack["vocab_tok"]
    vocab_ast = pack["vocab_ast"]
    max_tokens = pack["max_tokens"]
    max_ast_nodes = pack["max_ast_nodes"]
    fuse_ast = bool(cfg["data"].get("fuse_ast", True))
    embed_dim = int(cfg["data"]["embed_dim"])

    # Embedders
    emb_tok = TokenEmbedder(len(vocab_tok.itos), embed_dim, vocab_tok.pad_id).to(device)
    emb_ast = TokenEmbedder(len(vocab_ast.itos), embed_dim, vocab_ast.pad_id).to(device)

    # Build fixed vectors by mean pooling embeddings per segment
    items = []
    for row in encoded:
        tok = torch.tensor(row["tok_ids"], dtype=torch.long).unsqueeze(0).to(device)
        tok_mask = (tok != vocab_tok.pad_id).float()
        tok_vec = mean_pool(emb_tok(tok), tok_mask).squeeze(0).detach().cpu()

        ast_vec = None
        if fuse_ast:
            ast = torch.tensor(row["ast_ids"], dtype=torch.long).unsqueeze(0).to(device)
            ast_mask = (ast != vocab_ast.pad_id).float()
            ast_vec = mean_pool(emb_ast(ast), ast_mask).squeeze(0).detach().cpu()
            x = torch.cat([tok_vec, ast_vec], dim=-1)
        else:
            x = tok_vec

        items.append(SegmentItem(x=x, y=int(row["label"]), meta=row["meta"]))

    labels = [it.y for it in items]
    splits = load_split_files(cfg["paths"].get("splits_dir", "data/splits"))
    if splits is not None:
        train_idx, val_idx, test_idx = splits
    else:
        train_idx, val_idx, test_idx = stratified_split_indices(labels, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["random_seed"])

    train_ds = SegmentDataset([items[i] for i in train_idx])
    val_ds = SegmentDataset([items[i] for i in val_idx])
    test_ds = SegmentDataset([items[i] for i in test_idx])

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_batch)

    input_dim = train_ds[0]["x"].numel()
    model = RefactoCNN(
        input_dim=input_dim,
        conv1_filters=cfg["model"]["conv1_filters"],
        conv2_filters=cfg["model"]["conv2_filters"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = make_plateau_scheduler(optimizer, factor=cfg["train"]["lr_factor"], patience=cfg["train"]["lr_patience"])

    cw = None
    if cfg["train"]["class_weight"] == "auto":
        cw = compute_class_weight([items[i].y for i in train_idx])
    loss_fn = make_loss(cw)

    tb = make_tb_writer(cfg["paths"]["logs_dir"])

    res = train_model(
        model, train_loader, val_loader, loss_fn, optimizer, scheduler, device,
        epochs=cfg["train"]["epochs"], early_stop_patience=cfg["train"]["early_stop_patience"],
        tb_writer=tb
    )

    # restore best weights before final eval + saving
    model.load_state_dict(res.best_state_dict, strict=True)

    # final eval
    test_eval = evaluate_model(model, test_loader, device)

    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    best_path = os.path.join(cfg["paths"]["models_dir"], "best.pt")
    save_checkpoint(best_path, model, optimizer, epoch=res.best_epoch, best_metric=res.best_val_f1, config=cfg)

    os.makedirs(cfg["paths"]["reports_dir"], exist_ok=True)
    save_json(os.path.join(cfg["paths"]["reports_dir"], "test_metrics.json"), test_eval["metrics"])
    print("Saved:", best_path)
    print("Test metrics:", test_eval["metrics"])

if __name__ == "__main__":
    main()
