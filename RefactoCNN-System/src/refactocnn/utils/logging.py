from __future__ import annotations
import os, json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def make_tb_writer(logs_dir: str, run_name: str | None = None) -> SummaryWriter:
    os.makedirs(logs_dir, exist_ok=True)
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    return SummaryWriter(log_dir=os.path.join(logs_dir, run_name))

def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
