import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        self.network = nn.Sequential(
            # Small model for the first baseline, nothing fancy here.
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.network(x)
