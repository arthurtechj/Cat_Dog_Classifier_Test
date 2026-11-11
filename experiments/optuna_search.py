import json
from datetime import datetime

import optuna

from models.cnn import CNN
from utils.config import DEFAULT_SEED, OUTPUT_DIR
from utils.dataset import get_loaders
from utils.device import get_device
from utils.evaluate import evaluate
from utils.run import set_seed
from utils.train import train

device = get_device()
set_seed(DEFAULT_SEED)


def objective(trial):
    set_seed(DEFAULT_SEED + trial.number)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    train_loader, val_loader = get_loaders(
        batch_size=batch_size,
        augment=True
    )
    model = CNN(dropout=dropout)

    train(
        model,
        train_loader,
        val_loader,
        epochs=3,
        lr=lr,
        device=device,
        optimizer_name=optimizer_name,
        trial=trial
    )

    metrics = evaluate(model, val_loader, device, show=False)
    return metrics["accuracy"]


OUTPUT_DIR.mkdir(exist_ok=True)
study_path = OUTPUT_DIR / "optuna_study.db"

# The database means stopped searches can be continued another day.
study = optuna.create_study(
    study_name="cat_dog_search",
    storage=f"sqlite:///{study_path.as_posix()}",
    load_if_exists=True,
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
)

study.optimize(objective, n_trials=10)

best_result = {
    "saved_at": datetime.now().isoformat(timespec="seconds"),
    "best_accuracy": study.best_value,
    "best_parameters": study.best_params,
    "completed_trials": len(study.trials)
}

with open(OUTPUT_DIR / "optuna_best.json", "w", encoding="utf-8") as file:
    json.dump(best_result, file, indent=2)

print("\nBest Accuracy")
print(study.best_value)
print("\nBest Parameters")
print(study.best_params)
print(f"Saved search results to: {OUTPUT_DIR}")
