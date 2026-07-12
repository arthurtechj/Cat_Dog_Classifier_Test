from utils.dataset import get_loaders
from models.cnn import CNN
from utils.train import train
from utils.evaluate import evaluate


train_loader,val_loader=get_loaders(
    batch_size=32,
    augment=False
)


model=CNN()


history,time=train(
    model,
    train_loader,
    val_loader,
    epochs=10
)


print(
    "Training time:",
    time
)


evaluate(
    model,
    val_loader
)