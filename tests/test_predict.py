import torch
from PIL import Image

from predict import get_image_paths, predict_pil_image
from utils.dataset import get_inference_transform


def test_predict_image_returns_class_and_confidence(tmp_path):
    image_path = tmp_path / "dog.png"
    Image.new("RGB", (16, 16)).save(image_path)

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 2)
    )
    model[1].weight.data.zero_()
    model[1].bias.data = torch.tensor([0.0, 2.0])

    with Image.open(image_path) as image:
        result = predict_pil_image(
            image.convert("RGB"),
            model,
            ("cats", "dogs"),
            get_inference_transform(),
            torch.device("cpu")
        )

    assert result["prediction"] == "dogs"
    assert result["confidence"] > 0.8


def test_get_image_paths_skips_non_images(tmp_path):
    image_path = tmp_path / "cat.png"
    Image.new("RGB", (16, 16)).save(image_path)
    (tmp_path / "notes.txt").write_text("not an image")

    assert get_image_paths(tmp_path) == [image_path]
