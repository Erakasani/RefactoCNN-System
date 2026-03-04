from __future__ import annotations
import argparse, os
from ..utils.io import discover_java_files, safe_read_text
from ..preprocessing.ast_parser import extract_methods

def collect_segments(path: str):
    files = discover_java_files(path)
    for fp in files:
        code = safe_read_text(fp)
        methods = extract_methods(code)
        for m in methods:
            m["file"] = fp
            yield m
