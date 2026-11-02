import torch

from utils.dataset import get_loaders
from utils.config import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEED
)
from models.cnn import CNN
from utils.train import train
from utils.evaluate import evaluate
from utils.run import save_run, set_seed

print("=" * 50)
print("Experiment: Image Augmentation")
print("=" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# This lets me run the same experiment on a normal laptop too.
print(f"Using device: {device}")

set_seed(DEFAULT_SEED)
train_loader, val_loader = get_loaders(
    batch_size=DEFAULT_BATCH_SIZE,
)

model = CNN()

history, training_time = train(
    model,
    train_loader,
    val_loader,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    device=device
)

print(f"\nTraining Time: {training_time:.2f} seconds\n")

evaluate(model, val_loader, device)

run_dir = save_run(
    model,
    history,
    training_time,
    {
        "device": str(device),
        "seed": DEFAULT_SEED,
        "batch_size": DEFAULT_BATCH_SIZE,
        "epochs": DEFAULT_EPOCHS,
        "learning_rate": DEFAULT_LR,
        "augmentation": False,
        "classes": CLASS_NAMES
    },
    run_name="augmentation"
)
print(f"Saved run to: {run_dir}")
