import time

import torch
from tqdm import tqdm

from utils.device import get_device


def run_validation(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = loss_fn(output, y)
            total_loss += loss.item()

            predictions = output.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

    return total_loss / len(val_loader), correct / total


def train(
    model, train_loader, val_loader, epochs=5, lr=0.001, device=None,
    optimizer_name="Adam", trial=None
):
    if device is None:
        device = get_device()

    # Model and batch must be on the same device or PyTorch complains.
    model.to(device)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer not supported: {optimizer_name}")
    loss_fn = torch.nn.CrossEntropyLoss()
    start = time.time()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_accuracy = run_validation(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Seeing both numbers makes overfitting easier to notice.
        print(
            f"Epoch {epoch + 1} "
            f"train loss: {train_loss:.4f} "
            f"val loss: {val_loss:.4f} "
            f"val accuracy: {val_accuracy:.4f}"
        )

        if trial is not None:
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                # Stopping weak trials early saves a lot of time.
                import optuna
                raise optuna.TrialPruned()

    training_time = time.time() - start
    return history, training_time
