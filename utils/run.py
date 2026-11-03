import csv
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import OUTPUT_DIR


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Same seed should give the same result when CUDA is being used.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_history_files(history, metrics, run_dir):
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

        for epoch, values in enumerate(
            zip(history["train_loss"], history["val_loss"], history["val_accuracy"]),
            start=1
        ):
            writer.writerow([epoch, *values])

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_training_plot(history, run_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_accuracy"], color="green")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)

    figure.tight_layout()
    figure.savefig(run_dir / "training_plot.png")
    plt.close(figure)


def save_run(
    model, history, training_time, settings, run_name, metrics=None, output_dir=OUTPUT_DIR
):
    output_dir = Path(output_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_dir / f"{run_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "settings": settings,
        "metrics": metrics
    }
    torch.save(checkpoint, run_dir / "model.pt")

    metadata = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "training_time_seconds": training_time,
        "history": history,
        "settings": settings,
        "metrics": metrics
    }

    with open(run_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    # Keeping these next to the checkpoint makes comparisons less annoying.
    save_history_files(history, metrics, run_dir)
    save_training_plot(history, run_dir)

    return run_dir
