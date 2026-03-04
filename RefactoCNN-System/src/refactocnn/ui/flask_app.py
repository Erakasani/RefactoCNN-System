from __future__ import annotations
import os, tempfile, yaml, torch
from flask import Flask, request, render_template_string, redirect, url_for, send_file

from ..utils.io import safe_read_text, discover_java_files
from ..preprocessing.ast_parser import extract_methods
from ..embedding.embedder import TokenEmbedder
from ..models.refactocnn import RefactoCNN
from ..embedding.vocab import Vocab
from ..inference.pipeline import infer_segment
from ..inference.report import write_csv, write_html

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>RefactoCNN-System UI</title>
<style>
body{font-family:Arial,sans-serif;margin:20px;}
h1{font-size:20px;}
input[type=text]{width:70%;padding:8px;}
button{padding:8px 12px;}
table{border-collapse:collapse;width:100%;margin-top:16px;}
th,td{border:1px solid #ddd;padding:8px;vertical-align:top;}
th{background:#f5f5f5;}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;}
.ok{background:#e8f5e9;}
.warn{background:#fff8e1;}
.small{font-size:12px;color:#555;}
code{font-family:Consolas,monospace;font-size:12px;}
</style>
</head>
<body>
<h1>RefactoCNN-System: Inference UI</h1>
<form method="post" action="{{ url_for('run_infer') }}" enctype="multipart/form-data">
  <p class="small">Option A: Upload a single .java file</p>
  <input type="file" name="file"/>
  <p class="small">Option B: Enter a project/folder path (server-local)</p>
  <input type="text" name="path" placeholder="e.g., data/raw/my-java-project"/>
  <p class="small">Checkpoint path:</p>
  <input type="text" name="ckpt" value="{{ckpt_default}}"/>
  <p class="small">Config path:</p>
  <input type="text" name="config" value="{{config_default}}"/>
  <p style="margin-top:12px;">
    <button type="submit">Run Inference</button>
  </p>
</form>

{% if rows is not none %}
  <p class="small">Rows: {{ rows|length }}</p>
  <p>
    <a href="{{ url_for('download_csv') }}">Download CSV</a> |
    <a href="{{ url_for('download_html') }}">Download HTML</a>
  </p>
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
    {% for r in rows %}
      <tr>
        <td><b>{{r.file}}:{{r.start_line}}-{{r.end_line}}</b><br><span class="small">{{r.signature}}</span></td>
        <td>
          {% if r.pred_label == 1 %}
            <span class="badge warn">Refactor</span>
          {% else %}
            <span class="badge ok">No-Refactor</span>
          {% endif %}
          <br><span class="small">P(ref)={{ '%.3f'|format(r.prob_refactor) }}</span>
        </td>
        <td>{{r.suggestion}}<br><span class="small">{{r.rules_fired}}</span></td>
        <td>{{r.reason}}</td>
        <td><code>{{r.code_preview}}</code></td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
{% endif %}
</body>
</html>
"""

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _collect_segments(path: str):
    files = discover_java_files(path)
    for fp in files:
        code = safe_read_text(fp)
        for m in extract_methods(code):
            m["file"] = fp
            yield m

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["LAST_ROWS"] = None
    app.config["LAST_CSV"] = None
    app.config["LAST_HTML"] = None

    @app.get("/")
    def index():
        return render_template_string(PAGE, rows=app.config["LAST_ROWS"],
                                      ckpt_default="artifacts/models/best.pt",
                                      config_default="configs/default.yaml")

    @app.post("/run")
    def run_infer():
        up = request.files.get("file")
        path = (request.form.get("path") or "").strip()
        ckpt = (request.form.get("ckpt") or "").strip()
        config = (request.form.get("config") or "").strip()

        if up and up.filename:
            tmpdir = tempfile.mkdtemp(prefix="refactocnn_")
            fpath = os.path.join(tmpdir, up.filename)
            up.save(fpath)
            target_path = fpath
        else:
            target_path = path

        if not target_path:
            return redirect(url_for("index"))

        cfg = _load_yaml(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pack = torch.load(cfg["paths"]["features_pt"], map_location="cpu")
        vocab_tok: Vocab = pack["vocab_tok"]
        vocab_ast: Vocab = pack["vocab_ast"]

        embed_dim = int(cfg["data"]["embed_dim"])
        fuse_ast = bool(cfg["data"].get("fuse_ast", True))
        max_tokens = int(cfg["data"]["max_tokens"])
        max_ast_nodes = int(cfg["data"]["max_ast_nodes"])

        input_dim = embed_dim * (2 if fuse_ast else 1)
        model = RefactoCNN(
            input_dim=input_dim,
            conv1_filters=cfg["model"]["conv1_filters"],
            conv2_filters=cfg["model"]["conv2_filters"],
            kernel_size=cfg["model"]["kernel_size"],
            dropout=cfg["model"]["dropout"],
            num_classes=cfg["model"]["num_classes"],
        ).to(device)

        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
        model.eval()

        emb_tok = TokenEmbedder(len(vocab_tok.itos), embed_dim, vocab_tok.pad_id).to(device)
        emb_ast = TokenEmbedder(len(vocab_ast.itos), embed_dim, vocab_ast.pad_id).to(device)

        rows = []
        for seg in _collect_segments(target_path):
            out = infer_segment(model, emb_tok, emb_ast, vocab_tok, vocab_ast, seg,
                                max_tokens=max_tokens, max_ast_nodes=max_ast_nodes,
                                fuse_ast=fuse_ast, device=device)
            # Jinja friendly
            if isinstance(out.get("rules_fired"), list):
                out["rules_fired"] = ",".join(out["rules_fired"])
            rows.append(out)

        app.config["LAST_ROWS"] = rows

        # write downloadable artifacts to temp
        tmpdir = tempfile.mkdtemp(prefix="refactocnn_report_")
        csv_path = os.path.join(tmpdir, "infer_report.csv")
        html_path = os.path.join(tmpdir, "infer_report.html")
        write_csv(csv_path, rows)
        write_html(html_path, rows)
        app.config["LAST_CSV"] = csv_path
        app.config["LAST_HTML"] = html_path

        return redirect(url_for("index"))

    @app.get("/download.csv")
    def download_csv():
        p = app.config.get("LAST_CSV")
        if not p:
            return redirect(url_for("index"))
        return send_file(p, as_attachment=True, download_name="infer_report.csv")

    @app.get("/download.html")
    def download_html():
        p = app.config.get("LAST_HTML")
        if not p:
            return redirect(url_for("index"))
        return send_file(p, as_attachment=True, download_name="infer_report.html")

    return app
