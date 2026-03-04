from __future__ import annotations
import argparse, json, os
import torch
from tqdm import tqdm
from src.refactocnn.preprocessing.robust_parse import preprocess_snippet
from src.refactocnn.embedding.vocab import Vocab
from src.refactocnn.embedding.pooling import pad_truncate
from src.refactocnn.utils.seed import set_seed

def load_segments(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--max_ast_nodes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    segs = list(load_segments(args.segments))
    preps = [preprocess_snippet(s["code"], max_ast_nodes=args.max_ast_nodes) for s in tqdm(segs, desc="Preprocess")]
    tok_seqs = [p["tokens"] for p in preps]
    ast_seqs = [p["flat_ast"] for p in preps]

    vocab_tok = Vocab.build(tok_seqs, min_freq=1)
    vocab_ast = Vocab.build(ast_seqs, min_freq=1)

    # store encoded sequences (ids) for later embedding with nn.Embedding during training
    encoded = []
    for i, (s, p) in enumerate(zip(segs, preps)):
        s.setdefault("id", i)
        tok_ids = pad_truncate(vocab_tok.encode(p["tokens"]), args.max_tokens, vocab_tok.pad_id)
        ast_ids = pad_truncate(vocab_ast.encode(p["flat_ast"]), args.max_ast_nodes, vocab_ast.pad_id)
        encoded.append({
            "id": int(s.get("id", 0)),
            "tok_ids": tok_ids,
            "ast_ids": ast_ids,
            "label": int(s.get("label", 0)),
            "meta": {k: s.get(k) for k in ["file","signature","start_line","end_line"]},
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "encoded": encoded,
        "vocab_tok": vocab_tok,
        "vocab_ast": vocab_ast,
        "max_tokens": args.max_tokens,
        "max_ast_nodes": args.max_ast_nodes,
    }, args.out)
    print(f"Saved features package to {args.out}")

if __name__ == "__main__":
    main()
