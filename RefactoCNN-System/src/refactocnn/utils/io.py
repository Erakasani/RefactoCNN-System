from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional
from ..constants import DEFAULT_IGNORE_DIRS

def discover_java_files(path: str, ignore_dirs: Optional[set[str]] = None) -> List[str]:
    ignore_dirs = ignore_dirs or DEFAULT_IGNORE_DIRS
    files: List[str] = []
    if os.path.isfile(path) and path.endswith(".java"):
        return [path]
    for root, dirs, filenames in os.walk(path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for fn in filenames:
            if fn.endswith(".java"):
                files.append(os.path.join(root, fn))
    return sorted(files)

def safe_read_text(fp: str, max_bytes: int = 2_000_000) -> str:
    with open(fp, "rb") as f:
        b = f.read(max_bytes)
    return b.decode("utf-8", errors="replace")
