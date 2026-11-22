import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.train import train


def test_train_stops_when_validation_loss_does_not_improve():
    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 8 * 8, 2)
    )

    history, _ = train(
        model,
        loader,
        loader,
        epochs=5,
        lr=0,
        device=torch.device("cpu"),
        early_stopping_patience=1
    )

    assert len(history["val_loss"]) == 2
    assert history["best_epoch"] == 1
