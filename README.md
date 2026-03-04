# RefactoCNN-System

### A Deep Learning Framework for Automated Code Refactoring Detection Using Token–AST Fusion and CNN-Based Classification

---

# 1. Overview

**RefactoCNN-System** is a deep learning framework designed to automatically detect code segments that require refactoring in **Java software projects**. The system analyzes source code using both **lexical token sequences** and **Abstract Syntax Tree (AST)** structural representations.

These representations are fused into a unified feature vector and processed by a **1D Convolutional Neural Network (CNN)** classifier to determine whether a code segment requires refactoring.

The framework can also generate **interpretable refactoring suggestions**, such as:

* Extract Method
* Rename Variable
* Unify Duplicate Code

An optional **Flask-based web interface** allows users to visualize predictions and suggestions interactively.

This repository provides the complete pipeline including:

* Java code parsing and segmentation
* Token and AST feature extraction
* Feature embedding and fusion
* CNN-based refactoring classification
* Automated suggestion generation
* Training, evaluation, and inference utilities
* Hyperparameter optimization using Optuna
* Optional web interface for interactive analysis

---

# 2. Key Features

* Token + AST **Feature Fusion** for rich code representation
* **Deep CNN Architecture** optimized for sequential code features
* Automatic **Refactoring Suggestion Engine**
* **CLI and Web Interface** for inference
* **Hyperparameter Optimization** using Optuna
* Stratified dataset splitting for reproducible experiments
* Comprehensive experiment logging and reporting

---

# 3. System Architecture

The framework follows a modular pipeline aligned with the research methodology.

### 1. Input Module

Accepts either:

* Individual Java files
* Complete Java project directories

### 2. Code Segmentation

Extracts **method-level code segments** from Java source files.

### 3. Preprocessing

* Tokenization of source code
* AST construction using `javalang`
* Preorder traversal to flatten AST sequences

### 4. Embedding Module

* Token embeddings generated using vocabulary mapping
* AST node embeddings generated similarly
* Token and AST embeddings fused into a unified representation

### 5. RefactoCNN Model

Architecture:

* Conv1D (64 filters)
* Conv1D (128 filters)
* MaxPooling
* Fully Connected Layer
* Softmax classifier

### 6. Suggestion Engine

Applies rule-based heuristics to convert predictions into **actionable refactoring suggestions**.

### 7. Output Layer

Generates structured reports in:

* JSONL
* CSV
* HTML

---

# 4. Repository Structure

```
RefactoCNN-System/
│
├── configs/
│   ├── default.yaml
│   ├── model_refactocnn.yaml
│   └── optuna_search.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── splits/
│
├── artifacts/
│   ├── models/
│   ├── logs/
│   └── reports/
│
├── notebooks/
│
├── scripts/
│   ├── prepare_data.py
│   ├── build_features.py
│   ├── make_splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── optuna_tune.py
│   └── serve_ui.py
│
├── src/refactocnn/
│   ├── preprocessing/
│   │   ├── java_loader.py
│   │   ├── tokenizer.py
│   │   ├── ast_parser.py
│   │   ├── ast_flatten.py
│   │   └── segmenter.py
│   │
│   ├── embedding/
│   │   ├── vocab.py
│   │   ├── embedder.py
│   │   └── fusion.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   ├── collate.py
│   │   └── splits.py
│   │
│   ├── models/
│   │   └── refactocnn.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── checkpoints.py
│   │
│   ├── suggestion_engine/
│   │   ├── rules.py
│   │   └── mapper.py
│   │
│   ├── inference/
│   │   └── pipeline.py
│   │
│   └── ui/
│       └── flask_app.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 5. Installation

Clone the repository

```
git clone https://github.com/yourusername/RefactoCNN-System.git
cd RefactoCNN-System
```

Install dependencies

```
pip install -r requirements.txt
```

Recommended Python version

```
Python 3.9+
```

---

# 6. Dataset Preparation

Place Java projects inside:

```
data/raw/
```

The system will automatically extract **method-level segments**.

---

# 7. Data Preprocessing

Generate code segments

```
python scripts/prepare_data.py --input data/raw --out data/interim/segments.jsonl
```

Generate token + AST features

```
python scripts/build_features.py \
--segments data/interim/segments.jsonl \
--out data/processed/features.pt
```

---

# 8. Dataset Splitting

Create stratified splits

```
python scripts/make_splits.py \
--features data/processed/features.pt \
--out_dir data/splits
```

Generated files:

```
train.txt
val.txt
test.txt
```

---

# 9. Model Training

```
python scripts/train.py --config configs/default.yaml
```

Training parameters include:

* Batch size
* Learning rate
* Dropout
* Epochs
* Early stopping

Saved model

```
artifacts/models/best.pt
```

---

# 10. Model Evaluation

```
python scripts/evaluate.py \
--model artifacts/models/best.pt \
--features data/processed/features.pt
```

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient
* Confusion Matrix

---

# 11. Hyperparameter Optimization

```
python scripts/optuna_tune.py \
--config configs/optuna_search.yaml
```

Outputs:

```
artifacts/models/best_optuna.pt
artifacts/reports/optuna_best_params.json
```

---

# 12. Inference (Command Line)

```
python scripts/infer.py \
--path path/to/java/project \
--ckpt artifacts/models/best.pt
```

Generated reports:

* JSONL
* CSV
* HTML

Each report contains:

* File name
* Method location
* Prediction
* Confidence score
* Refactoring suggestion
* Explanation

---

# 13. Web Interface (Optional)

Start the Flask UI

```
python scripts/serve_ui.py
```

Open in browser

```
http://127.0.0.1:5000
```

Features:

* Upload Java files
* Run model inference
* View suggestions
* Download reports

---

# 14. Example Refactoring Suggestions

| Detected Issue       | Suggested Refactoring        |
| -------------------- | ---------------------------- |
| Long method          | Extract Method               |
| Duplicate logic      | Unify Duplicate Code         |
| Poor variable naming | Rename Variable              |
| General complexity   | General Refactor Recommended |

---

# 15. Reproducibility

To reproduce experiments:

1. Fix random seeds
2. Use provided split files
3. Store configuration files with checkpoints

The framework automatically logs configurations and metrics.

---

# 16. Research Applications

This framework can be used for:

* Automated software quality assessment
* Intelligent code review assistance
* Technical debt detection
* AI-assisted software maintenance tools

---

# 17. Future Improvements

Possible future extensions include:

* Transformer-based code embeddings
* Graph Neural Networks for AST modeling
* IDE plugins (IntelliJ / VSCode)
* Multi-language support (Python, C++, JavaScript)
* Reinforcement learning–based refactoring recommendation

---

# 18. Citation

If you use this repository in academic research, please cite:

```
@article{RefactoCNN,
title={RefactoCNN: A Deep Learning Framework for Automated Code Refactoring Detection Using Token–AST Fusion},
author={...},
journal={...},
year={2026}
}
```

---

# 19. Contact

For questions, issues, or collaboration opportunities, please open an **issue in this repository**.

# 20. License

This project is released under the **MIT License**.

See the LICENSE file for details.

---
