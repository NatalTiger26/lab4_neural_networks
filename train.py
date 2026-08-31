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

            self.relu = nn.ReLU()
            self.gelu = nn.GELU()
            self.swish = nn.SiLU()
            self.tanh = nn.Tanh()
            self.sign = torch.sign

            self.activation = self.activation_selector(activation.lower())


        def activation_selector(self, activation):
            if activation == 'relu':
                return self.relu
            elif activation == 'gelu':
                return self.gelu
            elif activation == 'swish':
                return self.swish
            elif activation == 'tanh':
                return self.tanh
            else:
                return self.sign
            

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
    metrics=['loss']
):
    """
    Train a model and optionally save full training checkpoints.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.

    train_loader : DataLoader
        Training data loader.

    val_loader : DataLoader
        Validation data loader.

    epochs : int
        Number of ADDITIONAL epochs to train.

    optimizer : torch.optim.Optimizer
        Optimizer.

    loss_fn : loss function
        Loss function.

    checkpoint_interval : int or None
        Save checkpoint every N epochs.

    checkpoint_path : str
        Directory where checkpoints are saved.

    device : str or torch.device
        Device used for training.

    start_epoch : int
        Epoch number from which training resumes.

    history : dict or None
        Previous training history. Used when resuming.

    Returns
    -------
    history : dict
        Training and validation metrics.
    """

    from pathlib import Path

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Initialize history
    # -------------------------------------------------

    if history is None:
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------

    pbar = tqdm(
        range(start_epoch, start_epoch + epochs),
        desc="Training"
    )

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

            batch_loss = loss_fn(
                predictions,
                y_batch
            )

            batch_loss.backward()

            optimizer.step()

            train_loss += (
                batch_loss.item()
                * y_batch.size(0)
            )

            train_correct += (
                predictions.argmax(dim=1) == y_batch
            ).sum().item()

            train_total += y_batch.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        if 'loss' in metrics:
            history["train_loss"].append(train_loss)
        if 'acc' in metrics:
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

                batch_loss = loss_fn(
                    predictions,
                    y_batch
                )

                val_loss += (
                    batch_loss.item()
                    * y_batch.size(0)
                )

                val_correct += (
                    predictions.argmax(dim=1) == y_batch
                ).sum().item()

                val_total += y_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        if 'loss' in metrics:
            history["val_loss"].append(val_loss)
        if 'acc' in metrics:
            history["val_acc"].append(val_acc)

        # =========================
        # Progress bar
        # =========================

        if 'acc' in metrics and 'loss' in metrics:
            pbar.set_postfix(
                epoch=epoch + 1,
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.3f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.3f}"
            )
        elif 'acc' in metrics and 'loss' not in metrics:
            pbar.set_postfix(
                epoch=epoch + 1,
                train_acc=f"{train_acc:.3f}",
                val_acc=f"{val_acc:.3f}"
            )
        elif 'acc' not in metrics and 'loss' in metrics:
            pbar.set_postfix(
                epoch=epoch + 1,
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}"
            )


        # =========================
        # Checkpoint
        # =========================

        if (
            checkpoint_interval is not None
            and ((epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == start_epoch + epochs)
        ):

            checkpoint = {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "history":
                    history
            }

            torch.save(
                checkpoint,
                checkpoint_path / "checkpoint.pt"
            )

    return model, history