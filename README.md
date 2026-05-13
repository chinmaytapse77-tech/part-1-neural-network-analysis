# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Overview
This project builds and analyzes a feed-forward neural network to predict **customer churn** using structured tabular data. It demonstrates key neural network concepts including forward pass, loss calculation, backpropagation, and the effect of hyperparameter changes on training behavior.

## Dataset
- **File:** `customer_churn_nn.csv`
- **Source:** [Masai Project Dataset Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)
- **Rows:** 2000 | **Columns:** 17
- **Target:** `churn` (0 = retained, 1 = churned)
- **Problem Type:** Binary Classification

## Project Structure
```
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── eda_plots.png
    ├── evaluation_outputs.png
    ├── training_curves_all.png
    ├── model_comparison_table.csv
    └── model_comparison_table.png
```

## Tasks Completed

### Task 1: Dataset Understanding
- 2000 rows, 17 columns (16 features + 1 target)
- 4 categorical features: `region`, `plan_type`, `contract_type`, `payment_method`
- 11 numerical features: tenure, charges, login days, tickets, delays, etc.
- No missing values
- Class imbalance: 98.45% retained, 1.55% churned

### Task 2: Data Preprocessing
- Dropped `customer_id` (non-predictive)
- One-hot encoded categorical columns (`drop_first=True`)
- Applied `StandardScaler` to numerical features
- 80/20 stratified train-test split

### Task 3: Neural Network Model Building
- Framework: TensorFlow/Keras
- Architecture: Input → Dense(32, ReLU) → Dense(1, Sigmoid)
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr=0.001)
- Class weighting applied to handle imbalance

### Task 4: Training and Evaluation
- **Test Accuracy:** ~94%
- **ROC-AUC:** ~0.89
- Confusion matrix and classification report included
- Training/validation loss and accuracy curves plotted

### Task 5: Hyperparameter Experimentation

| Experiment | Hidden Layers | Neurons | Learning Rate | Batch Size | Epochs | Activation | Test Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Exp 1 — Baseline | 1 | 32 | 0.001 | 32 | 50 | ReLU | ~0.9425 | ~0.8909 |
| Exp 2 — Deeper | 2 | 64 | 0.001 | 32 | 50 | ReLU | ~0.9800 | ~0.7695 |
| Exp 3 — Low LR | 2 | 64 | 0.0001 | 32 | 100 | ReLU | ~0.9300 | ~0.8354 |
| Exp 4 — Tanh | 2 | 64 | 0.001 | 64 | 50 | Tanh | ~0.9550 | ~0.8697 |

**Best Model:** Experiment 1 (Baseline) — highest ROC-AUC of 0.89.

### Task 6: Final Reflection
- **Weights & Biases:** Weights scale the input's influence; biases shift activations — together they define what the network learns.
- **Activation Functions:** Introduce non-linearity so the network can model complex patterns beyond linear separation.
- **Learning Rate:** Too high → divergence/oscillation; too low → underfitting/slow convergence.
- **Overfitting/Underfitting:** Exp 3 showed underfitting (low LR); Exp 2 showed mild overfitting; Exps 1 & 4 were well-balanced.

## How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd part-1-neural-network-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset in the project root
#    Download customer_churn_nn.csv from the dataset folder link above

# 4. Run the notebook
jupyter notebook notebook.ipynb
```

## Requirements
See `requirements.txt` for full list of dependencies.
