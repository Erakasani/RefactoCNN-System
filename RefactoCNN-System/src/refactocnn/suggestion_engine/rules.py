from __future__ import annotations
import re
from typing import Dict, List, Tuple

JAVA_KEYWORDS = {
    "abstract","assert","boolean","break","byte","case","catch","char","class","const","continue",
    "default","do","double","else","enum","extends","final","finally","float","for","goto","if",
    "implements","import","instanceof","int","interface","long","native","new","package","private",
    "protected","public","return","short","static","strictfp","super","switch","synchronized","this",
    "throw","throws","transient","try","void","volatile","while"
}

def detect_long_method(code: str, loc_threshold: int = 40) -> bool:
    loc = len([ln for ln in code.splitlines() if ln.strip()])
    return loc >= loc_threshold

def detect_naming_issues(tokens: List[str]) -> bool:
    # crude: single-letter vars heavily used, or uppercase variable-like identifiers
    ids = [t for t in tokens if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", t) and t not in JAVA_KEYWORDS]
    if not ids:
        return False
    single = sum(1 for t in ids if len(t) == 1)
    upper = sum(1 for t in ids if t.isupper() and len(t) > 2)
    return (single / max(1,len(ids)) > 0.25) or (upper > 0)

def detect_duplication(tokens: List[str], ngram: int = 8, min_hits: int = 2) -> bool:
    if len(tokens) < ngram*min_hits:
        return False
    grams = {}
    for i in range(len(tokens)-ngram+1):
        g = tuple(tokens[i:i+ngram])
        grams[g] = grams.get(g, 0) + 1
        if grams[g] >= min_hits:
            return True
    return False
