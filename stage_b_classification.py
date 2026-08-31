"""
Fashion-MNIST specific helpers for Day 2.

General MLP construction and the training loop live in train.py.
This file only has dataset loading and a couple of plots that only
make sense for this classification task.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def load_fashion_mnist(data_root="./data", train_ratio=0.8, batch_size=64, seed=42):
    """
    Load Fashion-MNIST, flatten to 784-d in [0, 1], split original train
    into train/val. Only the training loader is shuffled.

    Returns train_loader, val_loader, test_loader, class_names.
    """
    transform = transforms.ToTensor()
    train_full = datasets.FashionMNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_raw = datasets.FashionMNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    class_names = list(train_full.classes)

    X = train_full.data.float().flatten(start_dim=1) / 255.0
    y = train_full.targets
    X_test = test_raw.data.float().flatten(start_dim=1) / 255.0
    y_test = test_raw.targets

    g = torch.Generator().manual_seed(seed)
    n_train = int(train_ratio * len(train_full))
    idx = torch.randperm(len(train_full), generator=g).tolist()
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    train_loader = DataLoader(
        TensorDataset(X[train_idx], y[train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X[val_idx], y[val_idx]),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader, class_names


def plot_class_examples(loader, class_names, seed=42):
    """One random image per class — look before you model."""
    rng = random.Random(seed)
    ds = loader.dataset
    order = list(range(len(ds)))
    rng.shuffle(order)

    examples = {}
    for i in order:
        img, label = ds[i]
        label = int(label)
        if label not in examples:
            examples[label] = img
        if len(examples) == len(class_names):
            break

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Random example per class")
    for label, ax in enumerate(axes.flat):
        img = examples[label]
        if img.ndim == 1:
            img = img.reshape(28, 28)
        else:
            img = img.squeeze()
        ax.imshow(img, cmap="gray")
        ax.set_title(class_names[label])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_class_averages(loader, class_names):
    """Mean image per class — shirts vs coats at 28x28."""
    n = len(class_names)
    sums = torch.zeros(n, 784)
    counts = torch.zeros(n)

    for img, label in loader.dataset:
        label = int(label)
        sums[label] += img.view(-1).float()
        counts[label] += 1

    avgs = sums / counts[:, None]

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Average image per class")
    for label, ax in enumerate(axes.flat):
        ax.imshow(avgs[label].reshape(28, 28), cmap="gray")
        ax.set_title(class_names[label])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, loader, class_names, device="cpu", top_k=8):
    """Test-set confusion matrix + the pairs that get mixed up most."""
    ys, preds = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu().numpy()
            preds.append(pred)
            ys.append(y.numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=list(class_names)).plot(
        ax=ax, cmap="Blues", xticks_rotation=45
    )
    ax.set_title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.show()

    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)

    print("top confusions (true -> predicted):")
    for count, true_c, pred_c in pairs[:top_k]:
        print(f"  {true_c:12s} -> {pred_c:12s}  ({count})")

    return cm