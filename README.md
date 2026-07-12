# Deep Learning Model Optimization Playground

A deep learning project demonstrating the complete workflow of building, training, optimizing, and evaluating a Convolutional Neural Network (CNN) for image classification using **PyTorch** with **CUDA GPU acceleration**.

The project uses the **Cats vs Dogs** dataset and explores several stages of a real-world deep learning pipeline:

- Building a baseline CNN classifier
- Training on GPU using CUDA
- Feature engineering through image augmentation
- Hyperparameter optimization with Optuna
- Model evaluation using multiple performance metrics

---

# Technologies Used

- Python 3.13
- PyTorch
- Torchvision
- CUDA (GPU Training)
- Optuna
- Scikit-Learn
- Matplotlib
- tqdm

---

# Project Structure

```
DeepLearningProjectCatsDogsClassifier/

│
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   │
│   └── val/
│       ├── cats/
│       └── dogs/
│
├── models/
│   └── cnn.py
│
├── utils/
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── experiments/
│   ├── augmentation.py
│   └── optuna_search.py
│
├── outputs/
│
├── main.py
│
└── README.md
```

---

# General Workflow

The project follows a typical deep learning workflow.

```
Dataset

      │

      ▼

Image Preprocessing

      │

      ▼

CNN Architecture

      │

      ▼

GPU Training

      │

      ▼

Evaluation

      │

      ▼

Feature Engineering

      │

      ▼

Hyperparameter Optimization

      │

      ▼

Final Evaluation
```

---

# Dataset

The dataset consists of two classes:

- Cats
- Dogs

Training images are stored inside:

```
data/train/
```

Validation images are stored inside:

```
data/val/
```

PyTorch's `ImageFolder` automatically assigns labels based on folder names.

---

# CNN Architecture

The model is implemented inside:

```
models/cnn.py
```

Architecture:

```
Input Image (224×224×3)

        │

Conv2D

        │

ReLU

        │

MaxPooling

        │

Conv2D

        │

ReLU

        │

MaxPooling

        │

Conv2D

        │

ReLU

        │

MaxPooling

        │

Flatten

        │

Fully Connected Layer

        │

Dropout

        │

Output Layer

        │

Cat / Dog
```

The convolutional layers automatically learn image features such as:

- edges
- textures
- shapes
- facial features
- fur patterns

without requiring manual feature extraction.

---

# GPU Training

Training is performed entirely on the NVIDIA GPU using CUDA.

The model and tensors are moved to the GPU:

```python
model.to("cuda")

images = images.to("cuda")

labels = labels.to("cuda")
```

GPU acceleration significantly reduces training time compared to CPU execution.

---

# Baseline Experiment

File:

```
main.py
```

The baseline experiment trains the CNN without any image augmentation.

Original images are simply:

- resized to 224×224
- converted into tensors

The following values are recorded:

- Training Loss
- Training Time
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

This experiment establishes a performance baseline for future improvements.

---

# Dataset Loader

File:

```
utils/dataset.py
```

Responsible for:

- loading training dataset
- loading validation dataset
- applying transformations
- creating DataLoaders
- batching images

When augmentation is disabled:

```
Resize

↓

Tensor
```

When augmentation is enabled:

```
Resize

↓

Random Rotation

↓

Random Crop

↓

Horizontal Flip

↓

Color Jitter

↓

Tensor
```

---

# Training Loop

File:

```
utils/train.py
```

Responsible for training the CNN.

Each epoch consists of:

```
Load Batch

↓

Forward Pass

↓

Loss Calculation

↓

Backpropagation

↓

Weight Update

↓

Next Batch
```

This process repeats until every image has been processed once.

One complete pass over the dataset is called an **epoch**.

During training the project records:

- training loss
- total training time

---

# Evaluation

File:

```
utils/evaluate.py
```

The trained model is evaluated using:

### Accuracy

Percentage of correctly classified images.

---

### Precision

Of all images predicted as a given class,
how many predictions were correct.

---

### Recall

Of all actual images belonging to a class,
how many were correctly detected.

---

### F1 Score

Harmonic mean of precision and recall.

Useful when balancing false positives and false negatives.

---

### Confusion Matrix

Displays classification performance:

```
                Predicted

             Cat      Dog

Actual Cat   TP       FN

Actual Dog   FP       TN
```

This helps identify which class is more difficult to classify.

---

# Feature Engineering

File:

```
experiments/augmentation.py
```

This experiment evaluates whether image augmentation improves model performance.

Random augmentations include:

- Random Rotation
- Horizontal Flip
- Random Crop
- Color Jitter

Example:

Original image

↓

Rotated

↓

Flipped

↓

Brighter

↓

Cropped

Although the original dataset remains unchanged,
the model sees a different variation of the same image during each training epoch.

This improves generalization and reduces overfitting.

Results are compared against the baseline model.

---

# Hyperparameter Optimization

File:

```
experiments/optuna_search.py
```

Instead of manually testing hyperparameters,
Optuna automatically performs multiple training runs.

Parameters searched include:

- Learning Rate
- Batch Size
- Dropout
- Optimizer

Each trial trains a new CNN using a different parameter combination.

Example:

Trial 1

```
Learning Rate = 0.001

Batch Size = 32

Optimizer = Adam

Accuracy = 79%
```

Trial 2

```
Learning Rate = 0.0005

Batch Size = 64

Optimizer = SGD

Accuracy = 82%
```

...

After all trials, Optuna reports the best-performing configuration.

---

# Experiments

The project consists of three experiments.

## 1. Baseline CNN

```
uv run main.py
```

Purpose:

Train a CNN without augmentation.

Measures:

- Loss
- Accuracy
- Precision
- Recall
- F1
- Training Time

---

## 2. Image Augmentation

```
uv run experiments/augmentation.py
```

Purpose:

Evaluate the impact of feature engineering.

Comparison:

Baseline

↓

Augmented Images

↓

Performance Difference

---

## 3. Hyperparameter Optimization

```
uv run experiments/optuna_search.py
```

Purpose:

Automatically search for better hyperparameters.

The best configuration can later be used to retrain the final model.

---

# Typical Output

Training:

```
Epoch 1 Loss: 0.61

Epoch 2 Loss: 0.47

Epoch 3 Loss: 0.34

...

Epoch 10 Loss: 0.03
```

Evaluation:

```
Accuracy: 0.81

Precision: 0.82

Recall: 0.80

F1 Score: 0.81

Confusion Matrix:

[[2050 450]
 [510 1990]]
```

---

# Learning Outcomes

This project demonstrates understanding of:

- Convolutional Neural Networks (CNNs)
- GPU-accelerated Deep Learning
- PyTorch training workflow
- Image preprocessing
- Feature engineering through augmentation
- Hyperparameter optimization
- Deep learning evaluation metrics
- Experiment comparison
- Model performance analysis

---

# Conclusion

This project follows a complete deep learning workflow starting from a baseline CNN, improving the model through image augmentation, optimizing hyperparameters using Optuna, and evaluating performance using multiple classification metrics.

The implementation demonstrates practical experience with modern deep learning development using PyTorch and CUDA, while following an experimental approach commonly used in real-world machine learning projects.