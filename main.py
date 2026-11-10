import torch

from models.cnn import CNN
from utils.config import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEED
)
from utils.dataset import get_loaders
from utils.evaluate import evaluate
from utils.train import train
from utils.run import save_run, set_seed


# It falls back to CPU if CUDA is not installed on this machine.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

set_seed(DEFAULT_SEED)
train_loader, val_loader = get_loaders(batch_size=DEFAULT_BATCH_SIZE, augment=False)
model = CNN()

history, time = train(
    model,
    train_loader,
    val_loader,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    device=device
)

print("Training time:", time)
metrics = evaluate(model, val_loader, device)

run_dir = save_run(
    model,
    history,
    time,
    {
        "model": "cnn",
        "device": str(device),
        "seed": DEFAULT_SEED,
        "batch_size": DEFAULT_BATCH_SIZE,
        "epochs": DEFAULT_EPOCHS,
        "learning_rate": DEFAULT_LR,
        "augmentation": False,
        "classes": CLASS_NAMES
    },
    run_name="baseline",
    metrics=metrics
)
print(f"Saved run to: {run_dir}")
