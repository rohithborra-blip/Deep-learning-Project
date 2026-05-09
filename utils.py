"""Utility functions for inference and model operations."""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_model(model_class, checkpoint_path, device='cpu', num_classes=10):
    """Load a trained model from checkpoint."""
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_image(model, image_path, class_names, device='cpu', dataset='CIFAR10'):
    """Make a prediction on a single image."""
    
    # Load and preprocess image
    if dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225))
        ])
    else:  # MNIST
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    img = Image.open(image_path).convert('RGB' if dataset == 'CIFAR10' else 'L')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    return class_names[pred_idx], confidence, probs[0].cpu().numpy()


def visualize_prediction(image_path, class_names, model, device='cpu', dataset='CIFAR10'):
    """Visualize a prediction with confidence scores."""
    
    pred_class, confidence, probs = predict_image(model, image_path, class_names, device, dataset)
    
    # Load image for display
    img = Image.open(image_path).convert('RGB' if dataset == 'CIFAR10' else 'L')
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Display image
    axes[0].imshow(img if dataset == 'CIFAR10' else img, cmap='gray')
    axes[0].set_title(f'Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Display probabilities
    sorted_indices = np.argsort(probs)[::-1]
    top_k = min(10, len(class_names))
    
    colors = ['#2ecc71' if i == sorted_indices[0] else '#95a5a6' for i in range(len(class_names))]
    axes[1].barh(range(top_k), probs[sorted_indices[:top_k]], color=[colors[i] for i in sorted_indices[:top_k]])
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels([class_names[i] for i in sorted_indices[:top_k]])
    axes[1].set_xlabel('Probability', fontsize=11)
    axes[1].set_title(f'Prediction: {pred_class} ({confidence:.2%})', fontsize=12, fontweight='bold')
    axes[1].set_xlim([0, 1])
    
    for i, (idx, prob) in enumerate(zip(sorted_indices[:top_k], probs[sorted_indices[:top_k]])):
        axes[1].text(prob, i, f' {prob:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    print(f"✓ Prediction visualization saved to: prediction_result.png")
    plt.show()
    
    return pred_class, confidence


def get_model_info(model):
    """Print detailed model information."""
    print("\n" + "=" * 60)
    print("Model Architecture")
    print("=" * 60)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 60)
    print("Parameter Summary")
    print("=" * 60)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")


if __name__ == '__main__':
    print("Utility functions for model inference and analysis")
