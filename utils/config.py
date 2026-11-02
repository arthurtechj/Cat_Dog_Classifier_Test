from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
OUTPUT_DIR = PROJECT_DIR / "outputs"

CLASS_NAMES = ("cats", "dogs")
IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.001
DEFAULT_SEED = 42


def validate_dataset(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    split_dirs = {
        "train": data_dir / "train",
        "val": data_dir / "val"
    }

    for split_name, split_dir in split_dirs.items():
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Could not find {split_name} data folder: {split_dir}"
            )

        found_classes = tuple(sorted(item.name for item in split_dir.iterdir() if item.is_dir()))
        if found_classes != CLASS_NAMES:
            expected = ", ".join(CLASS_NAMES)
            found = ", ".join(found_classes) or "no class folders"
            raise ValueError(
                f"{split_name} needs class folders [{expected}], found [{found}]"
            )

    return split_dirs["train"], split_dirs["val"]
