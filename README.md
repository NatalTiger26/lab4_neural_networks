# Lab 4 — Neural Networks (PyTorch)

Two-day lab. Day 1 builds an MLP regressor on California Housing and compares it
against a hand-written NumPy gradient descent implementation. Day 2 builds a
Fashion-MNIST classifier with validation tracking, early stopping, and a
confusion matrix.

## Structure

```
lab4_neural_networks/
├── train.py                          # shared MLP builder + generic training loop (used by both days)
├── stage_a_regression.py             # Day 1: data loading, MLP, NumPy gradient descent
├── stage_b_classification.py         # Day 2: Fashion-MNIST loading, plotting, confusion matrix
├── day1_regression_starter.ipynb     # Day 1 notebook
├── day2_classification_starter.ipynb # Day 2 notebook
├── requirements.txt
├── data/                             # Fashion-MNIST, downloaded on first run (git-ignored)
└── models/                           # training checkpoints (git-ignored)
```

`stage_a.pt` and `stage_b.pt` (saved model weights from each day) are written to
the project root when the notebooks are run.

## Setup

```bash
python -m venv .lab4_neural_networks
source .lab4_neural_networks/bin/activate
pip install -r requirements.txt
```

`torch` is a large download — install it before the lab session starts.

## Running

Open the notebooks in order and run top to bottom:

1. `day1_regression_starter.ipynb`
   - Loads California Housing, standardizes features (scaler fit on train only),
     trains an MLP (`train.py`'s `training()` loop), and fits a NumPy gradient
     descent linear model on the same data for comparison.
   - Saves weights to `stage_a.pt`.
   - Also pre-downloads Fashion-MNIST into `data/` at the end, so Day 2 doesn't
     have to.

2. `day2_classification_starter.ipynb`
   - Loads Fashion-MNIST from `data/` (cached by Day 1), splits a validation
     set from the training data, and trains an MLP classifier with
     `nn.CrossEntropyLoss` (no softmax in the model — the loss applies it
     internally).
   - Uses patience-based early stopping (`train.py`'s `training()`, with
     `patience` set), restoring the best (cloned) validation state at the end.
   - Displays a confusion matrix and prints the most common misclassified
     pairs.
   - Saves weights to `stage_b.pt`.

Both notebooks import their model/training code from `stage_a_regression.py`,
`stage_b_classification.py`, and `train.py` rather than redefining it inline,
so the same `training()` function backs both days.

## Notes

- Device selection is automatic: MPS on Apple Silicon, else CPU.
- `data/` and `__pycache__/` are excluded via `.gitignore` and are not part of
  the submission.
- Expected Day 2 test accuracy: roughly 87–89%.