"""
Quick start guide and utility functions for the deep learning project.
This script demonstrates all major components of the project.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from model import ImageClassifier, SimpleCNN
from train import Trainer, create_visualizations
from config import get_config


def setup_environment():
    """Setup the environment and print information."""
    print("=" * 70)
    print("Deep Learning Image Classification - Quick Start Guide")
    print("=" * 70)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"✓ PyTorch Version: {torch.__version__}")
    print(f"✓ Random Seed: Set for reproducibility")
    
    return device


def load_dataset(config):
    """Load and prepare dataset."""
    print(f"\n{'='*70}")
    print(f"Loading Dataset: {config.DATASET}")
    print(f"{'='*70}")
    
    if config.DATASET == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225))
        ])
        
        dataset = datasets.CIFAR10(root=config.DATA_DIR, train=True, 
                                  transform=transform_train, download=True)
        test_dataset = datasets.CIFAR10(root=config.DATA_DIR, train=False, 
                                       transform=transform_test, download=True)
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    else:  # MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = datasets.MNIST(root=config.DATA_DIR, train=True, 
                                transform=transform, download=True)
        test_dataset = datasets.MNIST(root=config.DATA_DIR, train=False, 
                                     transform=transform, download=True)
        
        class_names = [str(i) for i in range(10)]
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, 
                          shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.TEST_BATCH_SIZE, 
                           shuffle=False, num_workers=config.NUM_WORKERS)
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print(f"  Classes: {config.NUM_CLASSES}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    
    return train_loader, val_loader, test_loader, class_names


def create_model(config, device):
    """Create and initialize model."""
    print(f"\n{'='*70}")
    print(f"Creating Model: {config.MODEL_TYPE}")
    print(f"{'='*70}")
    
    if config.MODEL_TYPE == 'ImageClassifier':
        model = ImageClassifier(num_classes=config.NUM_CLASSES, 
                              dropout_rate=config.DROPOUT_RATE)
    else:
        model = SimpleCNN(num_classes=config.NUM_CLASSES)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size: {total_params * 4 / 1e6:.2f} MB")
    
    return model


def train_model(model, device, train_loader, val_loader, config):
    """Train the model."""
    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Weight Decay: {config.WEIGHT_DECAY}")
    print(f"  Optimizer: {config.OPTIMIZER}")
    
    trainer = Trainer(model, device, learning_rate=config.LEARNING_RATE, 
                     weight_decay=config.WEIGHT_DECAY)
    trainer.fit(train_loader, val_loader, epochs=config.EPOCHS)
    
    return trainer


def evaluate_and_visualize(model, trainer, test_loader, class_names):
    """Evaluate model and create visualizations."""
    print(f"\n{'='*70}")
    print(f"Evaluation and Visualization")
    print(f"{'='*70}")
    
    results = trainer.evaluate(test_loader, class_names=class_names)
    create_visualizations(trainer, results, class_names)
    
    return results


def print_summary(results):
    """Print final summary."""
    print(f"\n{'='*70}")
    print(f"Final Results Summary")
    print(f"{'='*70}")
    print(f"✓ Accuracy:  {results['accuracy']:.4f}")
    print(f"✓ Precision: {results['precision']:.4f}")
    print(f"✓ Recall:    {results['recall']:.4f}")
    print(f"✓ F1 Score:  {results['f1']:.4f}")
    print(f"\n✓ Visualizations saved to: visualizations/")
    print(f"✓ Model saved to: checkpoints/best_model.pth")


def main():
    """Main execution flow."""
    # Setup
    device = setup_environment()
    
    # Load configuration
    config = get_config('CIFAR10')
    
    # Load dataset
    train_loader, val_loader, test_loader, class_names = load_dataset(config)
    
    # Create model
    model = create_model(config, device)
    
    # Train model
    trainer = train_model(model, device, train_loader, val_loader, config)
    
    # Evaluate and visualize
    results = evaluate_and_visualize(model, trainer, test_loader, class_names)
    
    # Print summary
    print_summary(results)
    
    print(f"\n{'='*70}")
    print("Quick Start Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
