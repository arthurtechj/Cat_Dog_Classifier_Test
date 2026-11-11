
from models.resnet import ResNet18Classifier
from utils.config import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_SEED,
    TRANSFER_EPOCHS,
)
from utils.dataset import get_loaders
from utils.device import get_device
from utils.evaluate import evaluate
from utils.run import save_run, set_seed
from utils.train import train

print("=" * 50)
print("Experiment: ResNet18 Transfer Learning")
print("=" * 50)

device = get_device()
print(f"Using device: {device}")

set_seed(DEFAULT_SEED)
train_loader, val_loader = get_loaders(
    batch_size=DEFAULT_BATCH_SIZE,
    augment=True,
    imagenet_normalization=True
)

model = ResNet18Classifier(num_classes=len(CLASS_NAMES))

history, training_time = train(
    model,
    train_loader,
    val_loader,
    epochs=TRANSFER_EPOCHS,
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
        "model": "resnet18",
        "pretrained": True,
        "frozen_backbone": True,
        "device": str(device),
        "seed": DEFAULT_SEED,
        "batch_size": DEFAULT_BATCH_SIZE,
        "epochs": TRANSFER_EPOCHS,
        "learning_rate": DEFAULT_LR,
        "augmentation": True,
        "imagenet_normalization": True,
        "classes": CLASS_NAMES
    },
    run_name="resnet18_head",
    metrics=metrics
)
print(f"Saved run to: {run_dir}")
