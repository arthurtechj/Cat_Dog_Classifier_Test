import json
import random
from datetime import datetime
from pathlib import Path

import torch

from utils.config import OUTPUT_DIR


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Same seed should give the same result when CUDA is being used.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    return run_dir
