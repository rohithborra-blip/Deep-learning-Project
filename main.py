import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from model import ImageClassifier, SimpleCNN
from train import Trainer, create_visualizations


def main():
    """Main training script."""
    
    # Configuration
    DATASET = 'CIFAR10'  # Options: 'CIFAR10', 'MNIST'
    MODEL_TYPE = 'ImageClassifier'  # Options: 'ImageClassifier', 'SimpleCNN'
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 30
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print(f"Deep Learning Image Classification Project")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET}")
    print(f"Model: {MODEL_TYPE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 60)
    
    # Prepare dataset
    if DATASET == 'CIFAR10':
        num_classes = 10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
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
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        val_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_test)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
        
        # Split train/val
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        val_indices = torch.arange(train_size, len(train_dataset) + train_size)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        
    else:  # MNIST
        num_classes = 10
        class_names = [str(i) for i in range(10)]
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"\nDataset Information:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: {num_classes}")
    
    # Create model
    if MODEL_TYPE == 'ImageClassifier':
        model = ImageClassifier(num_classes=num_classes, dropout_rate=0.5)
    else:
        model = SimpleCNN(num_classes=num_classes)
    
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Train model
    trainer = Trainer(model, DEVICE, learning_rate=LEARNING_RATE)
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    results = trainer.evaluate(test_loader, class_names=class_names)
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    create_visualizations(trainer, results, class_names, output_dir='visualizations')
    
    # Save model and results
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/best_model.pth')
    print(f"\n✓ Model saved to: checkpoints/best_model.pth")
    
    # Save summary
    with open('results_summary.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Deep Learning Image Classification Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {DATASET}\n")
        f.write(f"Model: {MODEL_TYPE}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
        f.write("Test Set Performance:\n")
        f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"  Precision: {results['precision']:.4f}\n")
        f.write(f"  Recall:    {results['recall']:.4f}\n")
        f.write(f"  F1 Score:  {results['f1']:.4f}\n\n")
        f.write("Training Details:\n")
        f.write(f"  Epochs: {EPOCHS}\n")
        f.write(f"  Batch Size: {BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {LEARNING_RATE}\n")
        f.write(f"  Device: {DEVICE}\n")
    
    print(f"✓ Results summary saved to: results_summary.txt")
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
