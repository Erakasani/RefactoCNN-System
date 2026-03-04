from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import csv, json, os

@dataclass(frozen=True)
class SegmentKey:
    file: str
    start_line: int
    end_line: int

def _norm_path(p: str) -> str:
    # Normalize to forward slashes and remove redundant separators.
    return os.path.normpath(p).replace('\\', '/')

def load_labels(path: str) -> Dict[SegmentKey, int]:
    """
    Loads labels from CSV or JSONL.

    CSV required columns: file,start_line,end_line,label
    JSONL required keys: file,start_line,end_line,label

    Label must be 0/1 or boolean-like.
    """
    if path.lower().endswith('.csv'):
        return _load_csv(path)
    if path.lower().endswith('.jsonl'):
        return _load_jsonl(path)
    raise ValueError(f'Unsupported labels file format: {path}')

def _as_int_label(v) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().lower()
    if s in {'1','true','yes','y','refactor'}:
        return 1
    if s in {'0','false','no','n','no_refactor','noref'}:
        return 0
    # fallback: try int
    return int(float(s))

def _load_csv(path: str) -> Dict[SegmentKey, int]:
    out: Dict[SegmentKey, int] = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = _norm_path(row['file'])
            k = SegmentKey(fp, int(row['start_line']), int(row['end_line']))
            out[k] = _as_int_label(row['label'])
    return out

def _load_jsonl(path: str) -> Dict[SegmentKey, int]:
    out: Dict[SegmentKey, int] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            fp = _norm_path(obj['file'])
            k = SegmentKey(fp, int(obj['start_line']), int(obj['end_line']))
            out[k] = _as_int_label(obj['label'])
    return out
