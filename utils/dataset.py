from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.config import (
    DEFAULT_BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    validate_dataset
)


def get_inference_transform(imagenet_normalization=False):
    normalization = []
    if imagenet_normalization:
        normalization = [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        *normalization
    ])


def get_loaders(
    batch_size=DEFAULT_BATCH_SIZE, augment=False, data_dir=None,
    imagenet_normalization=False
):
    if data_dir is None:
        train_dir, val_dir = validate_dataset()
    else:
        train_dir, val_dir = validate_dataset(data_dir)

    normalization = []
    if imagenet_normalization:
        normalization = [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # Changes are only for train images, validation must stay stable.
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            *normalization
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            *normalization
        ])

    val_transform = get_inference_transform(imagenet_normalization)

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader
