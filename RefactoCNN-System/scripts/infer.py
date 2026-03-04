from __future__ import annotations
import argparse, os, yaml, torch
from tqdm import tqdm
from src.refactocnn.utils.io import safe_read_text, discover_java_files
from src.refactocnn.preprocessing.ast_parser import extract_methods
from src.refactocnn.embedding.embedder import TokenEmbedder
from src.refactocnn.models.refactocnn import RefactoCNN
from src.refactocnn.embedding.vocab import Vocab
from src.refactocnn.inference.pipeline import infer_segment
from src.refactocnn.inference.report import write_jsonl, write_csv, write_html

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def collect_segments(path: str):
    files = discover_java_files(path)
    for fp in files:
        code = safe_read_text(fp)
        for m in extract_methods(code):
            m["file"] = fp
            yield m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Java file or project dir")
    ap.add_argument("--ckpt", required=True, help="Checkpoint .pt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="artifacts/reports/infer_report.jsonl")
    ap.add_argument("--out_csv", default="artifacts/reports/infer_report.csv")
    ap.add_argument("--out_html", default="artifacts/reports/infer_report.html")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pack = torch.load(cfg["paths"]["features_pt"], map_location="cpu")
    vocab_tok: Vocab = pack["vocab_tok"]
    vocab_ast: Vocab = pack["vocab_ast"]

    embed_dim = int(cfg["data"]["embed_dim"])
    fuse_ast = bool(cfg["data"].get("fuse_ast", True))
    max_tokens = int(cfg["data"]["max_tokens"])
    max_ast_nodes = int(cfg["data"]["max_ast_nodes"])

    input_dim = embed_dim * (2 if fuse_ast else 1)
    model = RefactoCNN(
        input_dim=input_dim,
        conv1_filters=cfg["model"]["conv1_filters"],
        conv2_filters=cfg["model"]["conv2_filters"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    emb_tok = TokenEmbedder(len(vocab_tok.itos), embed_dim, vocab_tok.pad_id).to(device)
    emb_ast = TokenEmbedder(len(vocab_ast.itos), embed_dim, vocab_ast.pad_id).to(device)

    results = []
    for seg in tqdm(list(collect_segments(args.path)), desc="Infer"):
        out = infer_segment(model, emb_tok, emb_ast, vocab_tok, vocab_ast, seg,
                            max_tokens=max_tokens, max_ast_nodes=max_ast_nodes,
                            fuse_ast=fuse_ast, device=device)
        results.append(out)

    write_jsonl(args.out, results)
    write_csv(args.out_csv, results)
    write_html(args.out_html, results)
    print("Wrote:", args.out)
    print("Wrote:", args.out_csv)
    print("Wrote:", args.out_html)

if __name__ == "__main__":
    main()
