from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm


def build_mlp(layers, act = 'relu'):
    in_dim, hidden, out_dim = layers[0], layers[1:-1], layers[-1]

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden, out_dim, activation = act):
            super().__init__()
            self.fc_first = nn.Linear(in_dim, hidden[0])
            self.hidden = nn.ModuleList(
                                        [nn.Linear(hidden[i], hidden[i+1]) for i in range(len(hidden) - 1)]
                                    )
            self.fc_last = nn.Linear(hidden[-1], out_dim)

            self.activation = self.activation_selector(activation.lower())


        def activation_selector(self, activation):
            if activation == 'relu':
                return nn.ReLU()
            elif activation == 'gelu':
                return nn.GELU()
            elif activation == 'swish':
                return nn.SiLU()
            elif activation == 'tanh':
                return nn.Tanh()
            else:
                return torch.sign
            

        def forward(self, x):
            x = self.fc_first(x)
            x = self.activation(x)
            for fc_hidden in self.hidden:
                x = fc_hidden(x)
                x = self.activation(x)
            x = self.fc_last(x)
            return x

    return MLP(in_dim, hidden, out_dim)




def training(
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    loss_fn,
    checkpoint_interval=None,
    checkpoint_path="models",
    device="cpu",
    start_epoch=0,
    history=None,
    metrics=["loss"],
    patience=None,
    early_stop_metric="val_acc",
    early_stop_mode="max",
):
    """
    Train a model with optional validation tracking, checkpoints, and
    patience-based early stopping.

    Parameters
    ----------
    patience : int or None
        Number of consecutive epochs without improvement before stopping.
        None disables early stopping.
    early_stop_metric : str
        History key to monitor (e.g. "val_acc" or "val_loss").
    early_stop_mode : {"max", "min"}
        Whether higher or lower values of the metric are better.

    Returns
    -------
    model, history
        Model restored to the best state (if early stopping used), and history dict.
    """
    from pathlib import Path

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if history is None:
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    best_metric = float("-inf") if early_stop_mode == "max" else float("inf")
    best_state = None
    epochs_without_improvement = 0

    pbar = tqdm(range(start_epoch, start_epoch + epochs), desc="Training")

    for epoch in pbar:
        # =========================
        # Training
        # =========================
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)
            batch_loss = loss_fn(predictions, y_batch)
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item() * y_batch.size(0)
            train_correct += (predictions.argmax(dim=1) == y_batch).sum().item()
            train_total += y_batch.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        if "loss" in metrics:
            history["train_loss"].append(train_loss)
        if "acc" in metrics:
            history["train_acc"].append(train_acc)

        # =========================
        # Validation
        # =========================
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(x_batch)
                batch_loss = loss_fn(predictions, y_batch)

                val_loss += batch_loss.item() * y_batch.size(0)
                val_correct += (predictions.argmax(dim=1) == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        if "loss" in metrics:
            history["val_loss"].append(val_loss)
        if "acc" in metrics:
            history["val_acc"].append(val_acc)

        # =========================
        # Progress bar
        # =========================
        postfix = {"epoch": epoch + 1}
        if "loss" in metrics:
            postfix["train_loss"] = f"{train_loss:.4f}"
            postfix["val_loss"] = f"{val_loss:.4f}"
        if "acc" in metrics:
            postfix["train_acc"] = f"{train_acc:.3f}"
            postfix["val_acc"] = f"{val_acc:.3f}"
        pbar.set_postfix(**postfix)

        # =========================
        # Early stopping
        # =========================
        if patience is not None:
            current = val_acc if early_stop_metric == "val_acc" else val_loss
            improved = (
                current > best_metric
                if early_stop_mode == "max"
                else current < best_metric
            )
            if improved:
                best_metric = current
                # Clone tensors so the snapshot does not keep changing
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    pbar.write(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(no improvement for {patience} epochs). "
                        f"Best {early_stop_metric}={best_metric:.4f}"
                    )
                    break

        # =========================
        # Checkpoint
        # =========================
        if checkpoint_interval is not None and (
            (epoch + 1) % checkpoint_interval == 0
            or (epoch + 1) == start_epoch + epochs
        ):
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }
            torch.save(checkpoint, checkpoint_path / "checkpoint.pt")

    # Restore best weights if early stopping was used
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

def find_lr(
    model,
    train_loader,
    optimizer,
    loss_fn,
    init_lr=1e-7,
    final_lr=10,
    num_iter=100,
    device="cpu",
):
    """
    LR range test (Smith 2017).

    Runs a short training pass while exponentially increasing the learning
    rate, then suggests a starting lr from the loss curve.

    Heuristic: take the lr where smoothed loss is lowest, then divide by 10
    (loss usually starts climbing a bit after the true good region).

    Restores model weights when finished.

    Returns
    -------
    suggested_lr : float
    lrs : list[float]
    losses : list[float]
    """
    state_before = {k: v.clone() for k, v in model.state_dict().items()}

    model.train()
    lrs, losses = [], []

    gamma = (final_lr / init_lr) ** (1 / max(num_iter - 1, 1))
    opt = type(optimizer)(model.parameters(), lr=init_lr)

    iter_count = 0
    for x, y in train_loader:
        if iter_count >= num_iter:
            break

        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)

        # skip non-finite losses (exploded) for the suggestion, still record them
        loss_val = loss.item()
        lrs.append(opt.param_groups[0]["lr"])
        losses.append(loss_val)

        if not (loss_val == loss_val) or loss_val > 1e8:  # nan or absurd
            break

        loss.backward()
        opt.step()

        for g in opt.param_groups:
            g["lr"] *= gamma
        iter_count += 1

    model.load_state_dict(state_before)

    # --- pick a suggestion from the curve ---
    # smooth a bit so one noisy batch doesn't dominate
    window = max(3, num_iter // 20)
    smoothed = []
    for i in range(len(losses)):
        lo = max(0, i - window // 2)
        hi = min(len(losses), i + window // 2 + 1)
        smoothed.append(sum(losses[lo:hi]) / (hi - lo))

    # ignore the very start (still adapting) and any exploding tail
    usable = [
        (i, s)
        for i, s in enumerate(smoothed)
        if s == s and s < 1e4  # finite and not exploded
    ]
    if not usable:
        suggested_lr = init_lr
    else:
        best_i, _ = min(usable, key=lambda t: t[1])
        # one decade below the min-loss lr is a common practical choice
        suggested_lr = lrs[best_i] / 10.0

    return suggested_lr, lrs, losses