import torch

from utils.dataset import get_loaders
from models.cnn import CNN
from utils.train import train
from utils.evaluate import evaluate

print("=" * 50)
print("Experiment: Image Augmentation")
print("=" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# This lets me run the same experiment on a normal laptop too.
print(f"Using device: {device}")

train_loader, val_loader = get_loaders(
    batch_size=32,
)

model = CNN()

history, training_time = train(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=0.001,
    device=device
)

print(f"\nTraining Time: {training_time:.2f} seconds\n")

evaluate(model, val_loader, device)
