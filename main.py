
from models.cnn import CNN
from utils.config import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEED,
    EARLY_STOPPING_PATIENCE,
)
from utils.dataset import get_loaders
from utils.device import get_device
from utils.evaluate import evaluate
from utils.run import save_run, set_seed
from utils.train import train

# It falls back to CPU if CUDA is not installed on this machine.
device = get_device()
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
    device=device,
    early_stopping_patience=EARLY_STOPPING_PATIENCE
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
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "augmentation": False,
        "classes": CLASS_NAMES
    },
    run_name="baseline",
    metrics=metrics
)
print(f"Saved run to: {run_dir}")
