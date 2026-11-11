# Cat and Dog Classifier

A PyTorch image-classification project with a small custom CNN, image augmentation, Optuna search, and a pretrained ResNet-18 option.

It uses CUDA when available and falls back to CPU when it is not.

## Dataset layout

Put images in these folders. Both splits must contain `cats` and `dogs` directories.

```text
data/
  train/
    cats/
    dogs/
  val/
    cats/
    dogs/
```

## Setup

```powershell
uv sync --group dev
```

## Train models

```powershell
# Basic CNN
uv run main.py

# CNN with augmentation
uv run experiments/augmentation.py

# Pretrained ResNet-18, training only its new classification head first
uv run experiments/transfer_learning.py

# Hyperparameter search; can resume using outputs/optuna_study.db
uv run experiments/optuna_search.py
```

Every training run writes a timestamped folder in `outputs/` with the checkpoint, metrics, training history, and a loss/accuracy plot.

## Predict images

```powershell
# One image
uv run predict.py path\to\image.jpg --checkpoint outputs\baseline_...\model.pt

# A folder of images
uv run predict.py path\to\images --checkpoint outputs\resnet18_head_...\model.pt
```

The command prints the predicted class and its confidence.

## Web interface

```powershell
uv run app.py
```

Open the local URL shown in the terminal, upload an image, paste the path to `model.pt`, then click **Classify**.

## Tests and checks

```powershell
uv run ruff check .
uv run pytest
```

GitHub Actions runs the same lint and test commands on pushes and pull requests.

## Current baseline

The original CNN baseline recorded **78.94% validation accuracy**. Run the augmentation and ResNet experiments to compare new results; their exact metrics and plots are saved under `outputs/`.
