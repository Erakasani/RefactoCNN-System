from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import javalang

def parse_ast(code: str) -> Optional[Any]:
    try:
        tree = javalang.parse.parse(code)
        return tree
    except Exception:
        return None

def extract_methods(code: str) -> List[Dict]:
    """Best-effort method segmentation using javalang.
    Returns a list of {name, signature, start_line, end_line, code}.
    End line is approximated by scanning braces from start_line.
    """
    methods: List[Dict] = []
    try:
        tree = javalang.parse.parse(code)
    except Exception:
        return methods

    # Build line index for brace scanning
    lines = code.splitlines()
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        if not getattr(node, "position", None):
            continue
        start_line = node.position.line
        name = node.name
        params = []
        for p in (node.parameters or []):
            try:
                ptype = getattr(p.type, "name", "var")
            except Exception:
                ptype = "var"
            params.append(f"{ptype} {p.name}")
        ret = getattr(node.return_type, "name", "void") if node.return_type else "void"
        signature = f"{ret} {name}({', '.join(params)})"

        # approximate end_line by brace matching from start_line-1
        end_line = _scan_method_end(lines, start_line)
        snippet = "\n".join(lines[start_line-1:end_line]) if end_line >= start_line else "\n".join(lines[start_line-1:])
        methods.append({
            "name": name,
            "signature": signature,
            "start_line": start_line,
            "end_line": end_line,
            "code": snippet,
        })
    return methods

def _scan_method_end(lines: List[str], start_line: int) -> int:
    # naive brace scan: find first '{' after start line, then balance
    i = start_line - 1
    brace = 0
    found_open = False
    for idx in range(i, len(lines)):
        line = lines[idx]
        for ch in line:
            if ch == '{':
                brace += 1
                found_open = True
            elif ch == '}':
                if found_open:
                    brace -= 1
                    if brace == 0:
                        return idx + 1
    return min(len(lines), start_line + 50)  # fallback cap
