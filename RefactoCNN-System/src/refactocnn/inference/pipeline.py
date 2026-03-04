from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from ..preprocessing.robust_parse import preprocess_snippet
from ..embedding.pooling import pad_truncate, mean_pool
from ..suggestion_engine.mapper import map_prediction_to_suggestion

@torch.no_grad()
def infer_segment(model, embedder_tok, embedder_ast, vocab_tok, vocab_ast,
                  segment: Dict, max_tokens: int, max_ast_nodes: int, fuse_ast: bool,
                  device: str) -> Dict:
    prep = preprocess_snippet(segment["code"], max_ast_nodes=max_ast_nodes)
    tok_ids = vocab_tok.encode(prep["tokens"])
    tok_ids = pad_truncate(tok_ids, max_tokens, vocab_tok.pad_id)
    tok_t = torch.tensor(tok_ids, dtype=torch.long).unsqueeze(0).to(device)
    tok_mask = (tok_t != vocab_tok.pad_id).float()
    tok_emb = embedder_tok(tok_t)
    tok_vec = mean_pool(tok_emb, tok_mask)

    ast_vec = None
    if fuse_ast:
        ast_ids = vocab_ast.encode(prep["flat_ast"])
        ast_ids = pad_truncate(ast_ids, max_ast_nodes, vocab_ast.pad_id)
        ast_t = torch.tensor(ast_ids, dtype=torch.long).unsqueeze(0).to(device)
        ast_mask = (ast_t != vocab_ast.pad_id).float()
        ast_emb = embedder_ast(ast_t)
        ast_vec = mean_pool(ast_emb, ast_mask)

    x = tok_vec if ast_vec is None else torch.cat([tok_vec, ast_vec], dim=-1)  # [1,D]
    logits = model(x)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    prob_refactor = float(probs[1])
    pred = int(prob_refactor >= 0.5)
    sug = map_prediction_to_suggestion(pred, prob_refactor, segment, prep["tokens"])

    out = {
        "file": segment.get("file"),
        "signature": segment.get("signature"),
        "start_line": segment.get("start_line"),
        "end_line": segment.get("end_line"),
        "prob_no_refactor": float(probs[0]),
        "prob_refactor": prob_refactor,
        "pred_label": pred,
        "suggestion": sug["suggestion"],
        "confidence": float(sug["confidence"]),
        "rules_fired": sug.get("rules_fired", []),
        "reason": sug.get("reason",""),
        "code_preview": (segment.get("code","")[:160].replace("\n"," ").strip()),
    }
    return out
