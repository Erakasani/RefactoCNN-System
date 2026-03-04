from __future__ import annotations
from typing import Dict, List, Optional
from .tokenizer import tokenize_java
from .ast_parser import parse_ast
from .ast_flatten import flatten_ast_preorder

def preprocess_snippet(code: str, max_ast_nodes: int = 256) -> Dict:
    tokens = tokenize_java(code)
    tree = parse_ast(code)
    flat_ast = flatten_ast_preorder(tree, max_nodes=max_ast_nodes) if tree is not None else []
    return {"tokens": tokens, "flat_ast": flat_ast, "ast_ok": tree is not None}
