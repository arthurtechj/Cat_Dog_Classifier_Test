import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from utils.config import CLASS_NAMES


def evaluate(model, loader, device=None, class_names=CLASS_NAMES):
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

    if not labels:
        raise ValueError("Validation loader has no images")

    class_ids = list(range(len(class_names)))
    precision = precision_score(
        labels, preds, labels=class_ids, average=None, zero_division=0
    )
    recall = recall_score(
        labels, preds, labels=class_ids, average=None, zero_division=0
    )
    f1 = f1_score(labels, preds, labels=class_ids, average=None, zero_division=0)
    matrix = confusion_matrix(labels, preds, labels=class_ids)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "classes": {}
    }

    for index, class_name in enumerate(class_names):
        metrics["classes"][class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index])
        }

    metrics["confusion_matrix"] = matrix.tolist()

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    for class_name, values in metrics["classes"].items():
        print(
            f"{class_name} - "
            f"precision: {values['precision']:.4f} "
            f"recall: {values['recall']:.4f} "
            f"f1: {values['f1']:.4f}"
        )

    print("Confusion Matrix (actual x predicted):")
    print(" " * 12 + " ".join(f"{name:>8}" for name in class_names))
    for class_name, row in zip(class_names, matrix):
        print(f"{class_name:>10}  " + " ".join(f"{value:>8}" for value in row))

    return metrics
