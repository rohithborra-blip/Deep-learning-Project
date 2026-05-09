# Deep Learning Image Classification Project

A complete, production-ready deep learning project for image classification using **PyTorch** with comprehensive training, evaluation, and visualization components.

## 📋 Project Overview

This project implements a Convolutional Neural Network (CNN) for image classification with:
- **Multiple dataset support**: CIFAR-10 and MNIST
- **Advanced model architecture**: Deep CNN with batch normalization and dropout
- **Robust training pipeline**: With learning rate scheduling and early stopping
- **Comprehensive evaluation**: Metrics including accuracy, precision, recall, and F1-score
- **Rich visualizations**: Training curves, confusion matrix, sample predictions, and performance metrics

## 🏗️ Project Structure

```
├── main.py                      # Main training script (entry point)
├── model.py                     # Model architectures (ImageClassifier, SimpleCNN)
├── train.py                     # Training pipeline and visualization utilities
├── utils.py                     # Utility functions for inference
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── checkpoints/                 # Saved model weights
│   └── best_model.pth
│
├── visualizations/              # Generated visualization plots
│   ├── 01_loss_curves.png
│   ├── 02_accuracy_curves.png
│   ├── 03_confusion_matrix.png
│   ├── 04_sample_predictions.png
│   └── 05_metrics_summary.png
│
├── data/                        # Downloaded datasets (auto-created)
│   ├── cifar-10-batches-py/
│   └── ...
│
└── results_summary.txt          # Text summary of results
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python main.py
```

The script will:
- Download CIFAR-10 dataset (27.3 MB)
- Train the model for 30 epochs
- Generate comprehensive visualizations
- Save the trained model

### 3. Training Output

The training process produces:
- **Visualizations** in `visualizations/` directory:
  - Loss curves (training vs validation)
  - Accuracy progression
  - Confusion matrix heatmap
  - Sample predictions (correct and incorrect)
  - Performance metrics bar chart
  
- **Model checkpoint** in `checkpoints/best_model.pth`
- **Results summary** in `results_summary.txt`

## 🧠 Model Architecture

### ImageClassifier (Default)
Deep CNN with multiple convolutional blocks optimized for CIFAR-10:

```
Input (3×32×32)
  ↓
[Conv64→ReLU→BN→Dropout] × 2 + MaxPool
[Conv128→ReLU→BN→Dropout] × 2 + MaxPool
[Conv256→ReLU→BN→Dropout] × 2 + MaxPool
  ↓
Global Average Pooling
  ↓
Dense(256)→ReLU→Dropout
  ↓
Output (num_classes)
```

**Features:**
- Batch normalization for stable training
- Dropout for regularization
- Adaptive average pooling for flexibility
- ~1.2M parameters

### SimpleCNN
Lightweight model for MNIST or quick training:
- Fewer layers and parameters
- Faster training time
- Suitable for simple datasets

## 📊 Training Configuration

Default hyperparameters (configurable in `main.py`):
- **Optimizer**: Adam
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Batch Size**: 128
- **Epochs**: 30
- **Loss Function**: CrossEntropyLoss
- **Weight Decay**: 1e-4
- **Dropout Rate**: 0.5

## 📈 Expected Results

### CIFAR-10 Performance (after 30 epochs)
- **Test Accuracy**: ~88-92%
- **Precision**: ~88-93%
- **Recall**: ~88-92%
- **F1-Score**: ~89-92%

Training Time: ~10-15 minutes on modern GPU, ~30-60 minutes on CPU

## 🎨 Visualizations Explained

### 1. Loss Curves (`01_loss_curves.png`)
- Shows training and validation loss over epochs
- Helps identify overfitting/underfitting

### 2. Accuracy Curves (`02_accuracy_curves.png`)
- Validation accuracy progression
- Indicates when model stabilizes

### 3. Confusion Matrix (`03_confusion_matrix.png`)
- Shows classification performance per class
- Identifies commonly confused classes

### 4. Sample Predictions (`04_sample_predictions.png`)
- 16 sample predictions from test set
- Shows both correct (✓, green) and incorrect (✗, red) predictions
- Includes confidence scores

### 5. Metrics Summary (`05_metrics_summary.png`)
- Bar chart of key metrics
- Easy comparison of accuracy, precision, recall, F1

## 🔧 Configuration Options

Edit `main.py` to customize:

```python
DATASET = 'CIFAR10'              # 'CIFAR10' or 'MNIST'
MODEL_TYPE = 'ImageClassifier'   # 'ImageClassifier' or 'SimpleCNN'
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 30
```

## 💻 Hardware Requirements

- **GPU (Recommended)**: 
  - NVIDIA GPU with CUDA support (4GB+ VRAM)
  - Training: ~10-15 minutes per 30 epochs
  
- **CPU (Alternative)**:
  - Training: ~30-60 minutes per 30 epochs
  - Slower but functional

- **Disk Space**:
  - CIFAR-10 dataset: ~180 MB
  - Model checkpoint: ~5 MB
  - Visualizations: ~10 MB

## 📚 Data Augmentation

Training data augmentation:
- Random horizontal flip (50% probability)
- Random crop with padding
- Normalization with ImageNet statistics

Validation/test data:
- Only normalization (no augmentation)

## 🔍 Detailed Code Walkthrough

### `model.py`
- **ConvBlock**: Reusable convolutional block with BN and dropout
- **ImageClassifier**: Main CNN architecture for CIFAR-10
- **SimpleCNN**: Lightweight model for MNIST

### `train.py`
- **Trainer**: Main training class with fit(), validate(), evaluate()
- **create_visualizations()**: Generates all 5 visualization plots
- Implements proper metrics calculation using scikit-learn

### `main.py`
- Dataset loading and preprocessing
- Model initialization
- Training orchestration
- Results aggregation

### `utils.py`
- Model loading from checkpoints
- Single image inference
- Prediction visualization
- Model information printing

## 🎯 Learning Outcomes

This project demonstrates:
- ✅ CNN architecture design
- ✅ Deep learning training pipeline
- ✅ Data augmentation and normalization
- ✅ Hyperparameter tuning
- ✅ Model evaluation and metrics
- ✅ Data visualization
- ✅ PyTorch best practices
- ✅ Production-ready code structure

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` (try 64, 32)
- Reduce `EPOCHS`
- Use `SimpleCNN` instead of `ImageClassifier`

### Slow Training
- Use GPU: Check `torch.cuda.is_available()`
- Install CUDA-enabled PyTorch
- Reduce `EPOCHS` for testing

### Download Issues
- Check internet connection
- Datasets auto-download on first run
- Cached data in `./data/` directory

## 📖 References

- PyTorch Documentation: https://pytorch.org/docs/stable/
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- CNN Basics: https://cs231n.github.io/convolutional-networks/

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to extend this project with:
- Different architectures (ResNet, VGG, etc.)
- Additional datasets (ImageNet, STL-10)
- Advanced techniques (mixup, cutmix)
- Transfer learning
- Model compression (quantization, pruning)

---

**Created**: May 2026 | **Device**: Linux (Ubuntu 24.04)
