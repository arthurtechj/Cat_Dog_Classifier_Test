import torch

from models.cnn import CNN
from utils.dataset import get_loaders
from utils.evaluate import evaluate
from utils.train import train


# It falls back to CPU if CUDA is not installed on this machine.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, val_loader = get_loaders(batch_size=32, augment=False)
model = CNN()

history, time = train(
    model,
    train_loader,
    val_loader,
    epochs=10,
    device=device
)

print("Training time:", time)
evaluate(model, val_loader, device)
