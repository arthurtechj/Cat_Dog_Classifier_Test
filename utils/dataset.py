from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(batch_size=32, augment=False):
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # Changes are only for train images, validation must stay stable.
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder("data/train", transform=train_transform)
    val_data = datasets.ImageFolder("data/val", transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader
