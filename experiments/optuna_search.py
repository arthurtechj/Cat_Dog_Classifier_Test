import optuna
import torch

from models.cnn import CNN
from utils.dataset import get_loaders
from utils.train import train
from utils.evaluate import evaluate


# Search can take a while, but at least it still runs without CUDA.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial):

    lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-2,
        log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64]
    )

    dropout = trial.suggest_float(
        "dropout",
        0.2,
        0.5
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["Adam", "SGD"]
    )

    train_loader, val_loader = get_loaders(
        batch_size=batch_size,
        augment=True
    )

    model = CNN(dropout=dropout).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr
        )

    model.train()

    for epoch in range(3):

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predicted = outputs.argmax(1)

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    return accuracy


study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=10
)

print("\nBest Accuracy")

print(study.best_value)

print("\nBest Parameters")

print(study.best_params)
