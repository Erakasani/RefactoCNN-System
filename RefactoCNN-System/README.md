# RefactoCNN-System

RefactoCNN-System is an end-to-end **Java refactoring opportunity predictor** based on a lightweight **CNN over fixed-length code embeddings** (tokens + flattened AST).  
It ingests a **single `.java` file or an entire Java project directory**, extracts method-level code segments, builds token/AST representations, generates fixed-length embeddings, and classifies each segment as **Refactor / No-Refactor**. Predicted refactoring labels are mapped to actionable suggestions (e.g., **Extract Method**, **Unify Duplicate Code**, **Rename Variable**) via a rule engine.

> This repository follows the methodology in the associated manuscript (Algorithms 1–4; Figures 1–3).

---

## Features

- **Java input**: file or project folder (`.java`)
- **Preprocessing**: tokenization + AST parsing (best-effort) + AST flattening
- **Embedding**: learned embeddings + averaging + optional token/AST fusion
- **Model**: Conv1D(64,k=3) → Conv1D(128,k=3) → MaxPool → Dense(64)+Dropout(0.5) → 2-class output
- **Training**: Adam(1e-3), scheduler (patience=3, factor=0.5), early stopping
- **Evaluation**: Accuracy, Precision, Recall, F1, MCC, Confusion Matrix
- **Inference**: CLI pipeline with JSONL report output
- **Suggestion Engine**: heuristic mapping from prediction + code metadata to refactoring recommendations

---

## Quickstart

### 1) Create environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare data (example: build a small dataset from local repos)

The project assumes **method-level segments**. If you do not have labels yet, the default `prepare_data.py` can generate **weak labels** using heuristics (long method / duplication / naming issues) to let you run end-to-end experiments.

```bash
python scripts/prepare_data.py --input data/raw --out data/interim/segments.jsonl
python scripts/build_features.py --segments data/interim/segments.jsonl --out data/processed/features.pt
```

### 3) Train

```bash
python scripts/train.py --config configs/default.yaml
```

### 4) Evaluate

```bash
python scripts/evaluate.py --config configs/default.yaml --ckpt artifacts/models/best.pt
```

### 5) Infer (file or project)

```bash
python scripts/infer.py --path path/to/SomeFile.java --ckpt artifacts/models/best.pt
# or
python scripts/infer.py --path path/to/java-project/ --ckpt artifacts/models/best.pt
```

Outputs a JSONL report with: file, method signature, line range, prediction, probability, and suggested refactoring.

---

## Repository Layout

```
RefactoCNN-System/
  configs/                 YAML configs
  data/
    raw/                   (gitignored) Java projects/files
    interim/               extracted method segments (jsonl)
    processed/             tensors/features + splits
  artifacts/
    models/                checkpoints (.pt)
    logs/                  TensorBoard logs
    reports/               evaluation artifacts
  scripts/                 runnable entrypoints
  src/refactocnn/          library code
  tests/                   unit tests (lightweight)
```

---

## Notes on Labels & Datasets

- **Supervised training** expects labels per segment: `label=1 (Refactor), 0 (No-Refactor)`.
- If you have a labeled refactoring dataset, adapt `scripts/prepare_data.py` to read your labels.
- The included weak labeling mode is for **pipeline validation**, not for final claims.

---

## Reproducibility

- Fixed seeds via `src/refactocnn/utils/seed.py`
- Config snapshots saved alongside checkpoints
- TensorBoard logs stored under `artifacts/logs/`

---

## Citation

If you use this codebase in academic work, cite the accompanying manuscript describing RefactoCNN-System.


---

## Hyperparameter Tuning (Optuna)

This repo includes an Optuna tuner that searches over learning rate, weight decay, CNN filter sizes, kernel size, dropout, batch size, and optional class-weighting.

### Run a tuning study

```bash
python scripts/optuna_tune.py --config configs/optuna_search.yaml
```

### Persist the study (recommended)

```bash
python scripts/optuna_tune.py --config configs/optuna_search.yaml \
  --storage sqlite:///optuna.db --study refactocnn_study --trials 50
```

Outputs:
- `artifacts/reports/optuna_best_params.json`
- `artifacts/reports/optuna_test_metrics.json`
- `artifacts/models/best_optuna.pt`


## Labeled data mode (Milestone 5)

If you have your own ground-truth labels (recommended), provide a labels file:

### Labels CSV format
Create `labels.csv` with columns:

- `file` (path to the Java file)
- `start_line` (1-based, inclusive)
- `end_line` (1-based, inclusive)
- `label` (0/1)

Then run:

```bash
python scripts/prepare_data.py --input data/raw --out data/interim/segments.jsonl --labels labels.csv
python scripts/build_features.py --segments data/interim/segments.jsonl --out data/processed/features.pt
python scripts/make_splits.py --features data/processed/features.pt --out_dir data/splits
python scripts/train.py --config configs/default.yaml
```

If `data/splits/train.txt`, `val.txt`, and `test.txt` exist, training will use them; otherwise it falls back to stratified splitting.



## Milestone 6: Flask Inference UI

Run a lightweight local UI to upload a `.java` file (or point to a local project folder) and download inference reports (CSV/HTML).

```bash
python scripts/serve_ui.py --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in your browser.
