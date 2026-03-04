from __future__ import annotations
import re
from typing import List

# Simple tokenizer tuned for Java-ish code. For stronger tokenization, replace with a parser tokenizer.
_TOKEN_RE = re.compile(r"""(
    //[^\n]*           |   # line comments
    /\*.*?\*/         |   # block comments
    "(?:\\.|[^"\\])*" |  # strings
    '(?:\\.|[^'\\])' |   # chars
    [A-Za-z_][A-Za-z0-9_]* |  # identifiers/keywords
    \d+\.\d+|\d+     |  # numbers
    ==|!=|<=|>=|&&|\|\||\+\+|--|<<|>>>|>>|
    [\{\}\(\)\[\];,\.]|
    [=+\-*/%<>!&|^~?:]     # operators
)""", re.VERBOSE | re.DOTALL)

def tokenize_java(code: str) -> List[str]:
    toks = []
    for m in _TOKEN_RE.finditer(code):
        t = m.group(0)
        # drop comments
        if t.startswith("//") or t.startswith("/*"):
            continue
        toks.append(t)
    return toks
