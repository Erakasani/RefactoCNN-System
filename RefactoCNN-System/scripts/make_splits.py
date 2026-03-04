from __future__ import annotations
import argparse, os, torch
from src.refactocnn.data.splits import stratified_split_indices

def write_list(path: str, idxs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for i in idxs:
            f.write(str(int(i)) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.pt")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pack = torch.load(args.features, map_location="cpu")
    labels = [int(r.get("label", 0)) for r in pack["encoded"]]
    train_idx, val_idx, test_idx = stratified_split_indices(labels, args.train_ratio, args.val_ratio, args.seed)

    write_list(os.path.join(args.out_dir, "train.txt"), train_idx)
    write_list(os.path.join(args.out_dir, "val.txt"), val_idx)
    write_list(os.path.join(args.out_dir, "test.txt"), test_idx)
    print(f"Wrote splits to {args.out_dir} (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

if __name__ == "__main__":
    main()
