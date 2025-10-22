RefactoCNN-System
An Optimized Deep Learning Framework for Predicting Software Refactoring Opportunities Using CNN-Based Code Analysis.

📌 Overview
RefactoCNN-System is a research-oriented tool designed to detect and prioritize refactoring opportunities in source code. It combines AST-based code representations with token embeddings and utilizes a Convolutional Neural Network (CNN) for effective classification. The framework supports explainability via Grad-CAM visualizations.

🧠 Key Features
- Dual embedding: AST + token-based representation
- Multi-channel CNN architecture
- Grad-CAM-based interpretability
- Compatible with Java-based codebases
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score

📂 Folder Structure

RefactoCNN-System/

├── data/

│   └── input_code_data.csv

├── preprocessing/

│   └── preprocess_ast.py

├── model/

│   └── refactocnn_model.py

├── explainability/

│   └── gradcam_utils.py

├── results/

│   └── output_plots/


├── main.py

├── requirements.txt


📝 Requirements
- Python 3.10+
- PyTorch 2.x
- NumPy, pandas, matplotlib
Install all dependencies using:
pip install -r requirements.txt
🚀 How to Run
Run the main script with:
python main.py
Customize paths in main.py for dataset and model output.
📊 Output
- Classification metrics (Precision, Recall, F1, Accuracy)
- Grad-CAM heatmaps
- CSV of predictions with true/false positives
📄 Citation
Please cite our paper if you use this codebase:

Lakshmi Prasanna et al., *RefactoCNN-System: An Optimized Deep Learning Framework for Predicting Software Refactoring Opportunities Using CNN-Based Code Analysis*, 2025.

