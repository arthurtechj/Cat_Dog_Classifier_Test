import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate(model, loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            output = model(x)
            prediction = torch.argmax(output, dim=1)

            # sklearn metrics work with CPU values.
            preds.extend(prediction.cpu().numpy())
            labels.extend(y.numpy())

    print("Accuracy:", accuracy_score(labels, preds))
    print("Precision:", precision_score(labels, preds))
    print("Recall:", recall_score(labels, preds))
    print("F1:", f1_score(labels, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))
