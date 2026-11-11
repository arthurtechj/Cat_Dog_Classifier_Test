from PIL import Image

from utils.dataset import get_loaders


def test_get_loaders_reads_cat_and_dog_images(tmp_path):
    for split in ("train", "val"):
        for class_name in ("cats", "dogs"):
            folder = tmp_path / split / class_name
            folder.mkdir(parents=True)
            Image.new("RGB", (12, 12)).save(folder / "image.png")

    train_loader, val_loader = get_loaders(batch_size=2, data_dir=tmp_path)
    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))

    assert train_images.shape == (2, 3, 224, 224)
    assert set(train_labels.tolist()) == {0, 1}
    assert val_images.shape == (2, 3, 224, 224)
    assert val_labels.tolist() == [0, 1]
