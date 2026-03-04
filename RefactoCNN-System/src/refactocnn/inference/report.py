from __future__ import annotations
import json, os, csv, html
from typing import Iterable, Dict, List

DEFAULT_FIELDS: List[str] = [
    "file", "signature", "start_line", "end_line",
    "prob_no_refactor", "prob_refactor", "pred_label",
    "suggestion", "confidence", "rules_fired", "reason", "code_preview"
]

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: str, rows: Iterable[Dict], fields: List[str] | None = None) -> None:
    _ensure_dir(path)
    fields = fields or DEFAULT_FIELDS
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            if isinstance(rr.get("rules_fired"), list):
                rr["rules_fired"] = ";".join(rr["rules_fired"])
            w.writerow({k: rr.get(k, "") for k in fields})

def write_html(path: str, rows: Iterable[Dict], title: str = "RefactoCNN Inference Report") -> None:
    _ensure_dir(path)
    rows = list(rows)
    # simple standalone HTML, friendly for paper screenshots
    head = f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
h1 {{ font-size: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #f5f5f5; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius: 10px; font-size: 12px; }}
.ok {{ background:#e8f5e9; }}
.warn {{ background:#fff8e1; }}
.small {{ font-size: 12px; color: #555; }}
code {{ font-family: Consolas, monospace; font-size: 12px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class='small'>Rows: {len(rows)}</p>
<table>
<thead>
<tr>
  <th>Location</th>
  <th>Prediction</th>
  <th>Suggestion</th>
  <th>Reason</th>
  <th>Preview</th>
</tr>
</thead>
<tbody>
"""
    body_rows = []
    for r in rows:
        loc = f"{r.get('file','')}:{r.get('start_line','')}-{r.get('end_line','')}"
        pred = "Refactor" if int(r.get("pred_label",0))==1 else "No-Refactor"
        badge_cls = "warn" if pred=="Refactor" else "ok"
        pred_html = f"<span class='badge {badge_cls}'>{html.escape(pred)}</span><br><span class='small'>P(ref)={float(r.get('prob_refactor',0.0)):.3f}</span>"
        sug = html.escape(str(r.get("suggestion","")))
        reason = html.escape(str(r.get("reason","")))
        preview = html.escape(str(r.get("code_preview","")))
        sig = html.escape(str(r.get("signature","")))
        rules = r.get("rules_fired", [])
        if isinstance(rules, list):
            rules = ", ".join(rules)
        rules = html.escape(str(rules))
        body_rows.append(
            f"<tr><td><b>{html.escape(loc)}</b><br><span class='small'>{sig}</span></td>"
            f"<td>{pred_html}</td>"
            f"<td>{sug}<br><span class='small'>{rules}</span></td>"
            f"<td>{reason}</td>"
            f"<td><code>{preview}</code></td></tr>"
        )
    tail = """</tbody></table>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(body_rows) + tail)
