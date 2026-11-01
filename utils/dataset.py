from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.config import DEFAULT_BATCH_SIZE, IMAGE_SIZE, validate_dataset


def get_loaders(batch_size=DEFAULT_BATCH_SIZE, augment=False, data_dir=None):
    if data_dir is None:
        train_dir, val_dir = validate_dataset()
    else:
        train_dir, val_dir = validate_dataset(data_dir)

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
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader
