import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and dropout."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.5):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ImageClassifier(nn.Module):
    """CNN for image classification (CIFAR-10)."""
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(ImageClassifier, self).__init__()
        
        # First block: 3 -> 64 channels
        self.block1 = nn.Sequential(
            ConvBlock(3, 64, dropout_rate=dropout_rate),
            ConvBlock(64, 64, dropout_rate=dropout_rate),
            nn.MaxPool2d(2, 2)
        )
        
        # Second block: 64 -> 128 channels
        self.block2 = nn.Sequential(
            ConvBlock(64, 128, dropout_rate=dropout_rate),
            ConvBlock(128, 128, dropout_rate=dropout_rate),
            nn.MaxPool2d(2, 2)
        )
        
        # Third block: 128 -> 256 channels
        self.block3 = nn.Sequential(
            ConvBlock(128, 256, dropout_rate=dropout_rate),
            ConvBlock(256, 256, dropout_rate=dropout_rate),
            nn.MaxPool2d(2, 2)
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleCNN(nn.Module):
    """Simpler CNN for MNIST or quick training."""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
