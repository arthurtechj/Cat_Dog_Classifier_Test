import argparse
from pathlib import Path

import torch
from PIL import Image

from models.cnn import CNN
from models.resnet import ResNet18Classifier
from utils.config import CLASS_NAMES
from utils.dataset import get_inference_transform


SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_paths(input_path):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Image or folder does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_IMAGES:
            raise ValueError(f"Unsupported image type: {input_path.suffix}")
        return [input_path]

    image_paths = [
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGES
    ]

    if not image_paths:
        raise ValueError(f"No supported images found in: {input_path}")

    return sorted(image_paths)


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model weights")

    settings = checkpoint.get("settings", {})
    class_names = tuple(settings.get("classes", CLASS_NAMES))
    model_name = settings.get("model", "cnn")

    if model_name == "resnet18":
        model = ResNet18Classifier(
            num_classes=len(class_names),
            pretrained=False
        )
    elif model_name == "cnn":
        model = CNN()
    else:
        raise ValueError(f"Model type is not supported: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_inference_transform(
        settings.get("imagenet_normalization", False)
    )
    return model, class_names, transform


def predict_image(image_path, model, class_names, transform, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except OSError as error:
        raise ValueError(f"Could not read image: {image_path}") from error

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]

    class_index = probabilities.argmax().item()
    return {
        "image": str(image_path),
        "prediction": class_names[class_index],
        "confidence": probabilities[class_index].item()
    }


def main():
    parser = argparse.ArgumentParser(description="Predict cats or dogs from images")
    parser.add_argument("input_path", help="An image file or a folder of images")
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, class_names, transform = load_model(args.checkpoint, device)
    for image_path in get_image_paths(args.input_path):
        result = predict_image(image_path, model, class_names, transform, device)
        print(
            f"{result['image']} -> {result['prediction']} "
            f"({result['confidence']:.2%})"
        )


if __name__ == "__main__":
    main()
