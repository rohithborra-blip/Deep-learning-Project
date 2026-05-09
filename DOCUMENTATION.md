# Deep Learning Project - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Module Documentation](#module-documentation)
5. [Configuration Guide](#configuration-guide)
6. [Training Details](#training-details)
7. [Visualization Guide](#visualization-guide)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

This is a **production-ready deep learning image classification project** built with PyTorch. It provides:

- ✅ Multiple CNN architectures (ImageClassifier, SimpleCNN)
- ✅ Support for CIFAR-10 and MNIST datasets
- ✅ Advanced training pipeline with learning rate scheduling
- ✅ Comprehensive evaluation metrics
- ✅ Five professional visualizations
- ✅ Model checkpointing and inference
- ✅ Clean, well-documented code

### Key Features

| Feature | Details |
|---------|---------|
| **Models** | ImageClassifier (1.2M params), SimpleCNN (0.3M params) |
| **Datasets** | CIFAR-10, MNIST |
| **Optimizer** | Adam with weight decay |
| **Scheduler** | ReduceLROnPlateau (adaptive learning rate) |
| **Regularization** | Batch norm, dropout, data augmentation |
| **Metrics** | Accuracy, Precision, Recall, F1-Score |
| **Visualizations** | Loss curves, accuracy, confusion matrix, predictions, metrics |

---

## Quick Start

### Installation

```bash
# Clone the repository
cd /workspaces/Deep-learning-Project

# Install dependencies
pip install -r requirements.txt
```

### Train Model

```bash
python main.py
```

Expected output:
- Training logs with loss and accuracy
- Best model saved to `checkpoints/best_model.pth`
- 5 visualization images in `visualizations/` folder
- Results summary in `results_summary.txt`

### Run Inference

```bash
python inference.py
```

### Quick Start Script

```bash
python quickstart.py
```

---

## Project Structure

```
Deep-learning-Project/
├── main.py                 # Main training script
├── model.py               # Model architectures
├── train.py              # Training and visualization
├── utils.py              # Utility functions
├── inference.py          # Inference examples
├── quickstart.py         # Quick start guide
├── config.py             # Configuration management
├── requirements.txt      # Dependencies
├── README.md             # Project readme
│
├── checkpoints/          # Model weights
│   └── best_model.pth   # Trained model
│
├── visualizations/       # Output plots
│   ├── 01_loss_curves.png
│   ├── 02_accuracy_curves.png
│   ├── 03_confusion_matrix.png
│   ├── 04_sample_predictions.png
│   └── 05_metrics_summary.png
│
├── data/                 # Downloaded datasets
│   ├── cifar-10-batches-py/
│   └── MNIST/
│
└── results_summary.txt   # Text results
```

---

## Module Documentation

### `model.py` - Model Architectures

#### ConvBlock Class
```python
class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and dropout."""
```

**Parameters:**
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `kernel_size`: Kernel size (default: 3)
- `padding`: Padding (default: 1)
- `dropout_rate`: Dropout probability (default: 0.5)

**Contains:**
- Conv2d layer
- BatchNorm2d
- ReLU activation
- Dropout2d

#### ImageClassifier Class
```python
class ImageClassifier(nn.Module):
    """CNN for image classification (CIFAR-10)."""
```

**Architecture:**
- Input: 3×32×32 images
- Block 1: Conv64×2 → MaxPool
- Block 2: Conv128×2 → MaxPool
- Block 3: Conv256×2 → MaxPool
- Global Average Pooling
- FC: 256→128→num_classes
- **Total Parameters: ~1.2M**

#### SimpleCNN Class
```python
class SimpleCNN(nn.Module):
    """Simpler CNN for MNIST or quick training."""
```

**Architecture:**
- Input: 1×28×28 images (MNIST)
- Conv32 → Conv64 + MaxPool
- Conv128 + MaxPool
- FC: 128×7×7 → 256 → num_classes
- **Total Parameters: ~0.3M**

---

### `train.py` - Training Pipeline

#### Trainer Class

**Constructor:**
```python
Trainer(model, device, learning_rate=0.001, weight_decay=1e-4)
```

**Key Methods:**

- `train_epoch(train_loader)`: Train for one epoch
- `validate(val_loader)`: Validate the model
- `fit(train_loader, val_loader, epochs=30)`: Main training loop
- `evaluate(test_loader, class_names=None)`: Test set evaluation

**Attributes:**
- `train_losses`: List of training losses per epoch
- `val_losses`: List of validation losses per epoch
- `val_accuracies`: List of validation accuracies per epoch
- `best_val_acc`: Best validation accuracy achieved
- `best_model_state`: State dict of best model

#### Visualization Function

```python
create_visualizations(trainer, results, class_names, output_dir='visualizations')
```

**Creates 5 visualizations:**
1. Training and validation loss curves
2. Validation accuracy progression
3. Confusion matrix heatmap
4. Sample predictions with confidence
5. Performance metrics bar chart

---

### `utils.py` - Utility Functions

#### load_model()
```python
model = load_model(model_class, checkpoint_path, device='cpu', num_classes=10)
```
Load a trained model from checkpoint file.

#### predict_image()
```python
pred_class, confidence, probs = predict_image(
    model, image_path, class_names, device='cpu', dataset='CIFAR10'
)
```
Make prediction on a single image.

#### visualize_prediction()
```python
pred_class, confidence = visualize_prediction(
    image_path, class_names, model, device='cpu', dataset='CIFAR10'
)
```
Create a visualization of prediction with confidence scores.

#### get_model_info()
```python
get_model_info(model)
```
Print detailed model information and parameter counts.

---

### `config.py` - Configuration Management

#### Base Config Class
Centralized configuration for the entire project.

**Common Settings:**
```python
DATASET = 'CIFAR10'           # Dataset choice
MODEL_TYPE = 'ImageClassifier' # Model architecture
BATCH_SIZE = 128              # Training batch size
EPOCHS = 30                   # Number of epochs
LEARNING_RATE = 0.001        # Initial learning rate
WEIGHT_DECAY = 1e-4           # L2 regularization
DROPOUT_RATE = 0.5           # Dropout probability
NUM_WORKERS = 2               # Data loading workers
```

#### Predefined Configs

- **CIFARConfig**: Optimized for CIFAR-10
- **MNISTConfig**: Optimized for MNIST
- **DevelopmentConfig**: Fast training for testing
- **Config**: Default configuration

**Usage:**
```python
from config import get_config
cfg = get_config('CIFAR10')
```

---

## Configuration Guide

### Using Configuration

**Option 1: Directly in main.py**
```python
DATASET = 'CIFAR10'
MODEL_TYPE = 'ImageClassifier'
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
```

**Option 2: Using config.py**
```python
from config import get_config
cfg = get_config('CIFAR10')  # Use predefined config
print(cfg.BATCH_SIZE)
```

### Creating Custom Config

```python
from config import Config

class MyConfig(Config):
    DATASET = 'CIFAR10'
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.0005
    DROPOUT_RATE = 0.3
```

### Dataset-Specific Settings

**CIFAR-10:**
- Image size: 32×32 (RGB)
- Classes: 10
- Train samples: 50,000
- Test samples: 10,000
- Recommended batch size: 128

**MNIST:**
- Image size: 28×28 (Grayscale)
- Classes: 10
- Train samples: 60,000
- Test samples: 10,000
- Recommended batch size: 256

---

## Training Details

### Optimizer: Adam

**Parameters:**
- Learning rate: 0.001
- Beta1 (momentum): 0.9
- Beta2: 0.999
- Weight decay: 1e-4

**Why Adam:** Better convergence, adaptive learning rates per parameter

### Scheduler: ReduceLROnPlateau

**Strategy:** Reduce learning rate when validation loss plateaus

**Parameters:**
- Mode: min (minimize loss)
- Factor: 0.5 (new_lr = lr × 0.5)
- Patience: 5 epochs
- Threshold: detected by PyTorch

**Example:** If validation loss doesn't improve for 5 epochs, learning rate is halved.

### Regularization Techniques

1. **Batch Normalization**
   - Normalizes layer inputs
   - Speeds up training
   - Reduces internal covariate shift

2. **Dropout**
   - Rate: 0.5 (50% neurons dropped)
   - Reduces co-adaptation
   - Improves generalization

3. **Data Augmentation** (Training only)
   - Random horizontal flip
   - Random crop with padding
   - Increases training data diversity

4. **Weight Decay**
   - L2 regularization: 1e-4
   - Penalizes large weights
   - Prevents overfitting

### Training Loop

```
for epoch in range(EPOCHS):
    for batch in train_loader:
        forward pass
        compute loss
        backward pass
        optimize
    
    validate on val_loader
    update learning rate
    save best model
```

---

## Visualization Guide

### 1. Loss Curves (`01_loss_curves.png`)

**What it shows:**
- Training loss (blue line)
- Validation loss (orange line)

**How to interpret:**
- Both curves decreasing: Good training
- Training loss decreases, validation loss increases: Overfitting
- Both curves high: Underfitting

**Expected pattern:** Both curves should decrease and converge

### 2. Accuracy Curves (`02_accuracy_curves.png`)

**What it shows:**
- Validation accuracy over epochs (green line)

**How to interpret:**
- Monotonic increase: Good training
- Plateauing: Model has learned most features
- Oscillating: Learning rate might be too high

**Expected final value:** 88-92% for CIFAR-10

### 3. Confusion Matrix (`03_confusion_matrix.png`)

**What it shows:**
- Heatmap of predicted vs true labels
- Diagonal = correct predictions

**How to interpret:**
- Dark diagonal: Good predictions
- Off-diagonal: Misclassified samples
- Symmetric errors indicate confusion between class pairs

**Example:** If airplane/bird confusion is high, visually similar classes are confused

### 4. Sample Predictions (`04_sample_predictions.png`)

**What it shows:**
- 16 random test samples
- 8 correct predictions (✓, green)
- 8 incorrect predictions (✗, red)
- Confidence scores

**How to interpret:**
- Green samples: Model is correct with high confidence
- Red samples: Model fails on hard examples
- Confidence values: How certain the model is

### 5. Metrics Summary (`05_metrics_summary.png`)

**Metrics displayed:**

- **Accuracy**: (TP+TN)/(Total) - Overall correctness
- **Precision**: TP/(TP+FP) - Positives that are correct
- **Recall**: TP/(TP+FN) - True positives found
- **F1-Score**: Harmonic mean of precision and recall

**Good values for CIFAR-10:**
- Accuracy: 0.88-0.92
- Precision: 0.88-0.93
- Recall: 0.88-0.92
- F1-Score: 0.89-0.92

---

## Advanced Usage

### Custom Dataset

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Use with model
train_loader = DataLoader(CustomDataset(...), batch_size=128)
```

### Transfer Learning

```python
import torchvision.models as models

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for CIFAR-10
model.fc = torch.nn.Linear(512, 10)

# Fine-tune only the last layer
model = model.to(device)
trainer = Trainer(model, device)
```

### Ensemble Predictions

```python
models = [load_model(...) for _ in range(5)]

def ensemble_predict(images):
    predictions = []
    for model in models:
        with torch.no_grad():
            output = model(images)
            predictions.append(output)
    return torch.mean(torch.stack(predictions), dim=0)
```

### Model Quantization

```python
import torch.quantization as quantization

# Quantize model for inference
quantized_model = quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## Troubleshooting

### Issue: CUDA Out of Memory (OOM)

**Solutions:**
1. Reduce batch size:
   ```python
   BATCH_SIZE = 32  # Instead of 128
   ```

2. Use SimpleCNN instead of ImageClassifier

3. Reduce number of epochs

4. Clear GPU cache:
   ```python
   torch.cuda.empty_cache()
   ```

### Issue: Very Slow Training

**Verify GPU usage:**
```python
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

**If CPU-bound:**
- Ensure PyTorch has CUDA support
- Use `nvidia-smi` to check GPU availability
- Reduce `NUM_WORKERS` if data loading is slow

### Issue: Accuracy Not Improving

**Possible causes:**
1. Learning rate too low: Increase to 0.01
2. Learning rate too high: Decrease to 0.0001
3. Model too small: Use ImageClassifier instead of SimpleCNN
4. Data issues: Check normalization values

**Solutions:**
```python
# Try learning rate search
for lr in [0.0001, 0.001, 0.01, 0.1]:
    trainer = Trainer(model, device, learning_rate=lr)
    trainer.fit(train_loader, val_loader, epochs=5)
    print(f"LR {lr}: {trainer.val_accuracies[-1]}")
```

### Issue: Model Not Downloading CIFAR-10

**Solutions:**
```python
# Download manually
import torchvision
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
```

Or check network connection and firewall settings.

### Issue: Results Not Saved

**Check permissions:**
```bash
ls -la checkpoints/
ls -la visualizations/
```

**Create directories manually:**
```bash
mkdir -p checkpoints visualizations
```

---

## Performance Optimization Tips

1. **Use DataLoader with num_workers > 0** for faster data loading
2. **Enable cudnn benchmarking** for faster GPU operations:
   ```python
   torch.backends.cudnn.benchmark = True
   ```
3. **Use mixed precision** for faster training:
   ```python
   from torch.cuda.amp import autocast
   with autocast():
       output = model(images)
   ```
4. **Gradient accumulation** for larger effective batch size
5. **Profile your code** to find bottlenecks

---

## Testing the Project

### Unit Test Example

```python
def test_model_output_shape():
    model = ImageClassifier(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    assert y.shape == (4, 10)

def test_trainer_fit():
    model = ImageClassifier()
    trainer = Trainer(model, device='cpu')
    # Create dummy data
    assert len(trainer.train_losses) > 0
```

---

**Last Updated**: May 2026
**Tested On**: Ubuntu 24.04 LTS with Python 3.10+
