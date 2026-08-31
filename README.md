# Neural Networks in PyTorch: Regression and Classification

Two small, self-contained projects built on a shared training module:

1. **Regression** — an MLP on the California Housing dataset, benchmarked
   against a from-scratch NumPy gradient descent linear model on the same
   preprocessed data.
2. **Classification** — an MLP on Fashion-MNIST with a proper train/val/test
   split, a learning-rate range test, patience-based early stopping, and a
   confusion-matrix error analysis.

Both stages train through the same generic loop in `train.py`, so model
construction, training, and early stopping are written once and reused.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore                        # excludes data/ and __pycache__/
│
├── train.py                          # shared: build_mlp, training(), find_lr()
├── stage_a_regression.py             # regression: data loading, MLP builder, NumPy GD baseline
├── stage_b_classification.py         # classification: Fashion-MNIST loading, plots, confusion matrix
│
├── day1_regression_starter.ipynb     # regression notebook (imports from the .py files above)
├── day2_classification_starter.ipynb # classification notebook (imports from the .py files above)
│
├── stage_a.pt                        # saved regression model weights (state_dict)
├── stage_b.pt                        # saved classification model weights (state_dict)
│
├── data/                             # Fashion-MNIST, downloaded on first run — git-ignored
│   └── FashionMNIST/
│
└── models/                           # periodic training checkpoints written by training() — git-ignored
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` and `torchvision` are large downloads — install them ahead of time
rather than mid-session. Key pinned versions (see `requirements.txt` for the
full list):

| Package | Version |
|---|---|
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| numpy | 2.5.2 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |

Device selection is automatic in both stage scripts:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

This picks Apple Silicon's MPS backend when available, otherwise falls back
to CPU. Swap in `"cuda" if torch.cuda.is_available() else "cpu"` if running
on an NVIDIA GPU instead.

## Running

```bash
jupyter notebook day1_regression_starter.ipynb   # regression, run top to bottom
jupyter notebook day2_classification_starter.ipynb   # classification, run top to bottom
```

The classification notebook expects Fashion-MNIST to already be cached in
`./data`; if that directory is empty, `load_fashion_mnist` downloads it
automatically on first call.

---

## Part 1 — Regression (`stage_a_regression.py`)

### Data (`load_data`)

- Loads `sklearn.datasets.fetch_california_housing` (20,640 rows, 8 numeric
  features, target = median house value).
- `train_test_split(test_size=0.2, random_state=0)`.
- `StandardScaler` is **fit on the training split only**, then used to
  `.transform()` the test split, so no information about the test
  distribution leaks into preprocessing.
- Both feature sets are cast to `torch.float32`.
- Targets are **reshaped to `(n, 1)`**. Left as `(n,)`, `nn.MSELoss` would
  silently broadcast a `(n, 1)` prediction against an `(n,)` target into an
  `(n, n)` matrix — no error, a loss that still decreases, and a training run
  that's learning something meaningless.

### Model (`build_mlp`)

A small configurable MLP: `Linear(in, hidden[0]) → activation → [Linear →
activation]* → Linear(hidden[-1], out)`, with the activation selectable by
name (`relu`, `gelu`, `swish`, `tanh`, or a sign-function fallback). For
regression: `build_mlp([8, 32, 1])` — 8 input features → one hidden layer of
width 32 → 1 output, ReLU.

### Training

- Loss: `nn.MSELoss()`. Optimizer: `optim.Adam(lr=0.01)`.
- The training split is further divided 80/20 into train/val via
  `random_split`, then wrapped in `DataLoader`s (only the train loader
  shuffles).
- Full-batch gradient descent (`batch_size = len(x_train)`), matching the
  update rule used by the NumPy baseline so the two are directly comparable.
- Trained through the shared `training()` loop in `train.py`, which logs
  `train_loss`/`val_loss` per epoch and returns `(model, history)`.
- Inside the loop, the scalar loss is extracted with `.item()` before being
  accumulated — never the loss tensor itself, since holding onto the tensor
  keeps its autograd graph alive and leaks memory over many epochs.
- Evaluation sets `model.eval()` and wraps the forward pass in
  `torch.no_grad()`: eval mode changes behavior for layers like dropout or
  batchnorm (not present here, but the right default), and `no_grad()` skips
  building an autograd graph that inference doesn't need.

### NumPy baseline (`gradient_descent_numpy`)

A from-scratch linear model, `y_pred = X @ w + b`, trained with hand-derived
gradients on the **same** preprocessed train/test split as the MLP:

```
grad_w = (2/n) · Xᵀ(y_pred − y)
grad_b = (2/n) · Σ(y_pred − y)
```

### Results

| Model | Test MSE |
|---|---|
| MLP (1000 epochs, full-batch Adam) | **0.3115** |
| NumPy gradient descent (linear) | 0.5391 |

The MLP beats the linear baseline, but not dramatically — California
Housing is close enough to linear that a single hidden layer has limited
extra structure to exploit. This is the expected, honest outcome for this
dataset rather than a sign of an undertrained model: a decisive win would
require real non-linearity or feature interactions that a linear model is
structurally unable to represent.

Trained weights are saved with `torch.save(model.state_dict(), "stage_a.pt")`.

---

## Part 2 — Classification (`stage_b_classification.py`)

### Data (`load_fashion_mnist`)

- Loads Fashion-MNIST via `torchvision.datasets.FashionMNIST`.
- Images are flattened to 784-length vectors and scaled to `[0, 1]`; labels
  are the original integer class indices.
- The 60,000-example training set is split 80/20 (`seed=42`) into
  **48,000 train / 12,000 val**; the 10,000-example test set is held out
  entirely until final evaluation.
- Only the train `DataLoader` shuffles (`batch_size=64`).

### Exploratory plots

`plot_class_examples` and `plot_class_averages` render sample and mean
images per class before any model is trained, so later judgments about which
confusions are "reasonable" are grounded in what the classes actually look
like rather than reasoned backward from the model's mistakes.

### Model

`build_mlp([784, 256, 128, 10], act="relu")` — flatten → 784 → 256 → 128 →
10 raw class scores. **No softmax in the model**: `nn.CrossEntropyLoss`
applies `log_softmax` internally, and stacking an explicit softmax on top of
that produces a model that trains badly with no error to explain why.

### Learning-rate range test (`find_lr`, in `train.py`)

Before committing to a learning rate, `find_lr` runs a short pass while
exponentially ramping the LR from `1e-7` to `1`, records the smoothed loss
curve, restores the model's original weights, and suggests a starting LR one
decade below the minimum-loss point. This run suggested **`lr ≈ 3.35e-4`**,
which is what was used for `optim.Adam`.

### Training with early stopping (`training`, in `train.py`)

Each epoch: a full pass over `train_loader`
(`zero_grad → forward → loss → backward → step` per batch, with
`model.train()` set beforehand), then a no-grad pass over `val_loader`
(with `model.eval()` set beforehand). Both loss and accuracy are logged per
epoch for train and val.

Early stopping tracks a counter of **consecutive** non-improving epochs
against validation accuracy, resetting on any new best and stopping once the
counter reaches a set patience (5 epochs here) — this avoids stopping on
ordinary epoch-to-epoch noise rather than a genuine plateau.

The best-so-far weights are snapshotted with
`{k: v.clone() for k, v in model.state_dict().items()}`. Cloning is what
makes this a real snapshot: a bare reference into `state_dict()` keeps
pointing at the live, still-training tensors, so an unclonded "best"
checkpoint would silently drift to match the final epoch instead of actually
freezing the best one. The best cloned state is restored at the end of
training.

### Results

- Training stopped early at **epoch 19 of 50** (5 epochs without a new best
  validation accuracy). Best validation accuracy: **89.38%**.
- **Test accuracy: 88.66% (8,866 / 10,000)**.
- Most common confused pairs (true → predicted), all within the
  shirt/pullover/coat/t-shirt cluster:

  | True | Predicted | Count |
  |---|---|---|
  | Shirt | T-shirt/top | 171 |
  | Coat | Pullover | 115 |
  | Shirt | Pullover | 100 |
  | Pullover | Coat | 74 |
  | Shirt | Coat | 66 |
  | T-shirt/top | Shirt | 64 |
  | Pullover | Shirt | 61 |
  | Coat | Shirt | 53 |

  These are the visually most similar classes at 28×28 resolution — all
  upper-body garments differing mainly in sleeve length and silhouette — so
  the confusion is expected for a low-resolution grayscale MLP rather than a
  sign of a bug.

Trained weights are saved with `torch.save(model.state_dict(), "stage_b.pt")`.

---

## `train.py` — shared training module

- **`build_mlp(layers, act='relu')`** — takes `[in_dim, *hidden_dims,
  out_dim]` and returns an `nn.Module` of that shape with the chosen
  activation (`relu` / `gelu` / `swish` / `tanh`, else a sign-function
  fallback).
- **`training(model, train_loader, val_loader, epochs, optimizer, loss_fn, ...)`**
  — the generic train/validate loop used by both stages:
  - `metrics=["loss", "acc"]` — which running metrics to log per epoch.
  - `patience`, `early_stop_metric`, `early_stop_mode` — optional
    patience-based early stopping on any logged metric, with cloned
    best-state restoration.
  - `checkpoint_interval`, `checkpoint_path` — optional periodic checkpoints
    (model + optimizer state + history) written under `models/`.
  - `start_epoch`, `history` — for resuming training across calls.
  - Returns `(model, history)`; if early stopping fired, `model` is restored
    to its best-validation state before being returned.
- **`find_lr(model, train_loader, optimizer, loss_fn, init_lr=1e-7, final_lr=10, num_iter=100, ...)`**
  — a learning-rate range test (Smith, 2017): ramps the learning rate
  exponentially over `num_iter` batches, records the loss curve, restores the
  model's original weights, and returns a suggested starting LR (one decade
  below the minimum-smoothed-loss point) along with the raw `(lrs, losses)`
  arrays for plotting.

---

## Design notes

A few choices worth calling out, since each avoids a failure mode that runs
without error but produces a wrong result:

| Choice | What it prevents |
|---|---|
| Target reshaped to `(n, 1)` before training | `MSELoss` silently broadcasting `(n,1)` vs `(n,)` into a meaningless `(n, n)` loss |
| `StandardScaler` fit on train only, applied to test | Test-set statistics leaking into preprocessing |
| Loss pulled out with `.item()` before accumulating | Holding the loss tensor keeps its autograd graph alive — real memory growth over epochs |
| No softmax layer, raw scores into `CrossEntropyLoss` | Double-softmax: the model still runs and reports a loss, but learns badly with no error surfaced |
| Best `state_dict()` cloned before storing | An un-cloned reference keeps mutating with the model, so "best" silently becomes "final" |
| Early stopping on *consecutive* non-improving epochs, not one | Stopping on ordinary training noise instead of a genuine plateau |
| `model.eval()` + `torch.no_grad()` at evaluation time | Wrong layer behavior if dropout/batchnorm is ever added, plus unnecessary memory/compute |

---

## Reproducing these results

```bash
source .venv/bin/activate
jupyter notebook day1_regression_starter.ipynb    # restart kernel, run top to bottom
jupyter notebook day2_classification_starter.ipynb    # restart kernel, run top to bottom
```

Random seeds are fixed where it matters (`torch.manual_seed(0)`,
`np.random.seed(0)` in regression; `seed=42` for the Fashion-MNIST train/val
split), but exact numbers may still shift slightly across hardware and
library versions.