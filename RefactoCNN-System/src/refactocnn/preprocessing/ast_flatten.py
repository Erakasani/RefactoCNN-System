from __future__ import annotations
from typing import Any, List
import javalang

def flatten_ast_preorder(tree: Any, max_nodes: int = 256) -> List[str]:
    """Flatten AST as a sequence of node type names in preorder."""
    if tree is None:
        return []
    out: List[str] = []
    stack = [tree]
    while stack and len(out) < max_nodes:
        node = stack.pop()
        if isinstance(node, (list, tuple)):
            # push reversed for preorder
            for item in reversed(node):
                stack.append(item)
            continue
        if isinstance(node, javalang.ast.Node):
            out.append(type(node).__name__)
            # children yields (attr, child)
            children = list(node.children)
            # children may contain None/lists/nodes
            for child in reversed(children):
                stack.append(child)
        else:
            # primitives ignored
            continue
    return out
