from __future__ import annotations
import argparse, json, os, random
from tqdm import tqdm

from src.refactocnn.utils.io import discover_java_files, safe_read_text
from src.refactocnn.preprocessing.ast_parser import extract_methods
from src.refactocnn.preprocessing.tokenizer import tokenize_java
from src.refactocnn.suggestion_engine.rules import detect_long_method, detect_duplication, detect_naming_issues
from src.refactocnn.data.labels import load_labels, SegmentKey

def weak_label(segment_code: str) -> int:
    toks = tokenize_java(segment_code)
    # weak rule: any of these signals => label 1
    return int(detect_long_method(segment_code) or detect_duplication(toks) or detect_naming_issues(toks))

def norm_path(p: str) -> str:
    return os.path.normpath(p).replace('\\', '/')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .java file or project directory")
    ap.add_argument("--out", required=True, help="Output JSONL for segments")
    ap.add_argument("--labels", default=None, help="Optional labels file (.csv or .jsonl) with: file,start_line,end_line,label")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    label_map = load_labels(args.labels) if args.labels else None

    files = discover_java_files(args.input)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for fp in tqdm(files, desc="Scanning Java files"):
            code = safe_read_text(fp)
            methods = extract_methods(code)
            for m in methods:
                m["file"] = fp
                if label_map is not None:
                    k = SegmentKey(norm_path(fp), int(m.get("start_line", -1)), int(m.get("end_line", -1)))
                    m["label"] = int(label_map.get(k, weak_label(m["code"])))
                else:
                    m["label"] = weak_label(m["code"])
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} method segments to {args.out}")

if __name__ == "__main__":
    main()
