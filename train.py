import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-4):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        # Save best model
        if accuracy > self.best_val_acc:
            self.best_val_acc = accuracy
            self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        self.scheduler.step(avg_loss)
        return avg_loss, accuracy
    
    def fit(self, train_loader, val_loader, epochs=30):
        """Train the model."""
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\nTraining complete! Best validation accuracy: {self.best_val_acc:.4f}")
    
    def evaluate(self, test_loader, class_names=None):
        """Evaluate on test set and compute metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_images = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_images.append(images.cpu())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_images = torch.cat(all_images, dim=0)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'images': all_images,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        
        print(f"\nTest Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        return results


def create_visualizations(trainer, results, class_names, output_dir='visualizations'):
    """Create comprehensive visualizations."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Training and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.train_losses, label='Train Loss', linewidth=2)
    plt.plot(trainer.val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_loss_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/01_loss_curves.png")
    plt.close()
    
    # 2. Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.val_accuracies, label='Validation Accuracy', linewidth=2, color='green')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_accuracy_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/02_accuracy_curves.png")
    plt.close()
    
    # 3. Confusion Matrix
    cm = results['confusion_matrix']
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/03_confusion_matrix.png")
    plt.close()
    
    # 4. Sample Predictions
    preds = results['predictions']
    labels = results['labels']
    images = results['images']
    probs = results['probabilities']
    
    # Get diverse samples: correct and incorrect
    correct_mask = preds == labels
    incorrect_indices = np.where(~correct_mask)[0][:8]
    correct_indices = np.where(correct_mask)[0][:8]
    
    sample_indices = np.concatenate([correct_indices, incorrect_indices])
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, ax_idx in enumerate(sample_indices[:16]):
        img = images[ax_idx]
        
        # Denormalize the image
        if img.shape[0] == 3:  # RGB
            img = img.permute(1, 2, 0)
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            img = img * std + mean
        else:  # Grayscale
            img = img.squeeze()
        
        img = torch.clamp(img, 0, 1)
        
        ax = axes[idx]
        ax.imshow(img if img.shape[-1] == 3 else img, cmap='gray')
        
        true_label = labels[ax_idx]
        pred_label = preds[ax_idx]
        confidence = probs[ax_idx, pred_label]
        
        is_correct = true_label == pred_label
        color = 'green' if is_correct else 'red'
        status = '✓' if is_correct else '✗'
        
        ax.set_title(f'{status} True: {class_names[true_label]}\n'
                    f'Pred: {class_names[pred_label]} ({confidence:.2%})',
                    fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_sample_predictions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/04_sample_predictions.png")
    plt.close()
    
    # 5. Metrics summary
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        results['accuracy'],
        results['precision'],
        results['recall'],
        results['f1']
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_metrics_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/05_metrics_summary.png")
    plt.close()
    
    print(f"\n✓ All visualizations saved to '{output_dir}/' directory!")
