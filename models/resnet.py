import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.network = resnet18(weights=weights)

        # The first run only learns the small cat/dog head.
        for parameter in self.network.parameters():
            parameter.requires_grad = False

        features = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features, num_classes)
        )

    def forward(self, x):
        return self.network(x)
