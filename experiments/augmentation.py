
from models.cnn import CNN
from utils.config import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEED,
)
from utils.dataset import get_loaders
from utils.device import get_device
from utils.evaluate import evaluate
from utils.run import save_run, set_seed
from utils.train import train

print("=" * 50)
print("Experiment: Image Augmentation")
print("=" * 50)

device = get_device()
# This lets me run the same experiment on a normal laptop too.
print(f"Using device: {device}")

set_seed(DEFAULT_SEED)
train_loader, val_loader = get_loaders(
    batch_size=DEFAULT_BATCH_SIZE,
    augment=True
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

metrics = evaluate(model, val_loader, device)

run_dir = save_run(
    model,
    history,
    training_time,
    {
        "model": "cnn",
        "device": str(device),
        "seed": DEFAULT_SEED,
        "batch_size": DEFAULT_BATCH_SIZE,
        "epochs": DEFAULT_EPOCHS,
        "learning_rate": DEFAULT_LR,
        "augmentation": True,
        "classes": CLASS_NAMES
    },
    run_name="augmentation",
    metrics=metrics
)
print(f"Saved run to: {run_dir}")
