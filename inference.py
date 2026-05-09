"""
Example inference script for making predictions on images.
"""

import torch
from model import ImageClassifier
from utils import predict_image, visualize_prediction, get_model_info


def main():
    """Example inference."""
    
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'checkpoints/best_model.pth'
    DATASET = 'CIFAR10'
    
    # Class names
    if DATASET == 'CIFAR10':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = 10
    else:  # MNIST
        class_names = [str(i) for i in range(10)]
        num_classes = 10
    
    print("=" * 60)
    print("Deep Learning Image Classification - Inference Example")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET}")
    
    # Load model
    try:
        model = ImageClassifier(num_classes=num_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        print(f"\n✓ Model loaded from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"\n✗ Error: Model checkpoint not found at {MODEL_PATH}")
        print("Please train the model first by running: python main.py")
        return
    
    # Display model info
    get_model_info(model)
    
    # Example: Predict on a sample from test set
    print("\n" + "=" * 60)
    print("Making Predictions on Test Samples")
    print("=" * 60)
    
    from torchvision import datasets, transforms
    
    if DATASET == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225))
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                       transform=transform, download=False)
    else:  # MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(root='./data', train=False, 
                                     transform=transform, download=False)
    
    # Make predictions on random samples
    model.eval()
    with torch.no_grad():
        for idx in [0, 42, 100, 256, 512]:
            if idx < len(test_dataset):
                image, true_label = test_dataset[idx]
                image = image.unsqueeze(0).to(DEVICE)
                
                outputs = model(image)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_idx].item()
                
                match = "✓" if pred_idx == true_label else "✗"
                print(f"\nSample {idx}:")
                print(f"  {match} True: {class_names[true_label]}")
                print(f"  {match} Pred: {class_names[pred_idx]} ({confidence:.2%})")
    
    print("\n" + "=" * 60)
    print("Inference Example Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
