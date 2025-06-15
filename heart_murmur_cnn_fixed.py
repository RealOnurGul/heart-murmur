import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import os
from tqdm import tqdm
import json
from datetime import datetime

# Set device (M3 Mac optimization)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class HeartMurmurDataset(Dataset):
    """Custom dataset for heart murmur spectrograms"""
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get image path and label
        img_path = self.dataframe.iloc[idx]['spectrogram_path']
        label = self.dataframe.iloc[idx]['label_encoded']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class SimplerHeartMurmurCNN(nn.Module):
    """Simplified CNN to avoid overfitting with small dataset"""
    
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super(SimplerHeartMurmurCNN, self).__init__()
        
        # Simpler architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(4, 4)  # Larger pooling
        self.dropout = nn.Dropout(dropout_rate)
        
        # Smaller fully connected layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional layers with batch norm and pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ImprovedHeartMurmurTrainer:
    """Improved training class with class imbalance handling"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 criterion, optimizer, device, class_names, class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.class_names = class_names
        self.class_weights = class_weights
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch with detailed monitoring"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(len(self.class_names))}
        class_total = {i: 0 for i in range(len(self.class_names))}
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class statistics
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        # Print per-class training accuracy
        print("Training per-class accuracy:")
        for i, class_name in enumerate(self.class_names):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"  {class_name}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch with detailed monitoring"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(len(self.class_names))}
        class_total = {i: 0 for i in range(len(self.class_names))}
        all_predictions = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Track predictions
                all_predictions.extend(predicted.cpu().numpy())
                
                # Per-class statistics
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        # Print per-class validation accuracy
        print("Validation per-class accuracy:")
        for i, class_name in enumerate(self.class_names):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"  {class_name}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")
        
        # Check if model is stuck predicting one class
        unique_preds = np.unique(all_predictions)
        print(f"Unique predictions: {[self.class_names[i] for i in unique_preds]}")
        if len(unique_preds) == 1:
            print(f"⚠️  WARNING: Model only predicting class {self.class_names[unique_preds[0]]}!")
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs, save_path="best_model_fixed.pth"):
        """Full training loop with early stopping"""
        best_val_acc = 0.0
        patience = 15  # Increased patience
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training on {len(self.train_loader.dataset)} samples")
        print(f"Validating on {len(self.val_loader.dataset)} samples")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"\nEpoch Summary:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"🎉 New best model saved! Val Acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def evaluate(self, model_path="best_model_fixed.pth"):
        """Evaluate model on test set"""
        # Load best model
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        test_acc = accuracy_score(all_labels, all_predictions)
        
        print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        self.plot_confusion_matrix(cm)
        
        return test_acc, all_predictions, all_labels
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Fixed Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()

def prepare_data_with_weights():
    """Prepare dataset with class weights for imbalanced data"""
    print("Preparing dataset with class balancing...")
    
    # Load metadata
    df = pd.read_csv('processed_data/complete_metadata_augmented.csv')
    print(f"Total samples: {len(df)}")
    
    # Check class distribution
    print("\nClass distribution:")
    class_counts = df['disease'].value_counts()
    for disease, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {disease}: {count} samples ({percentage:.1f}%)")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['disease'])
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(df['label_encoded']),
        y=df['label_encoded']
    )
    
    print(f"\nClass weights (to balance dataset):")
    for i, (class_name, weight) in enumerate(zip(label_encoder.classes_, class_weights)):
        print(f"  {class_name}: {weight:.3f}")
    
    # Save label encoder for later use
    class_names = label_encoder.classes_.tolist()
    with open('class_names_fixed.json', 'w') as f:
        json.dump(class_names, f)
    
    print(f"Classes: {class_names}")
    
    # Split data: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, 
                                        stratify=df['label_encoded'], 
                                        random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, 
                                      stratify=temp_df['label_encoded'], 
                                      random_state=42)
    
    print(f"\nData splits:")
    print(f"Training: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df, class_names, class_weights

def create_data_loaders_minimal_augmentation(train_df, val_df, test_df, batch_size=8):
    """Create data loaders with minimal augmentation to avoid overfitting"""
    
    # Minimal data transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Smaller images
        transforms.RandomHorizontalFlip(p=0.3),  # Less aggressive augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = HeartMurmurDataset(train_df, transform=train_transform)
    val_dataset = HeartMurmurDataset(val_df, transform=val_test_transform)
    test_dataset = HeartMurmurDataset(test_df, transform=val_test_transform)
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def main():
    """Main training function with fixes"""
    print("Heart Murmur CNN Training - FIXED VERSION")
    print("=" * 60)
    
    # Prepare data with class weights
    train_df, val_df, test_df, class_names, class_weights = prepare_data_with_weights()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders_minimal_augmentation(
        train_df, val_df, test_df, batch_size=8  # Smaller batch size
    )
    
    # Create simpler model
    model = SimplerHeartMurmurCNN(num_classes=len(class_names), dropout_rate=0.3)
    model = model.to(device)
    
    print(f"\nSimplified Model architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Weighted loss function for class imbalance
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Much smaller learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    print(f"\nTraining Configuration:")
    print(f"  Learning rate: 0.0001 (reduced from 0.001)")
    print(f"  Batch size: 8 (reduced from 16)")
    print(f"  Image size: 128x128 (reduced from 224x224)")
    print(f"  Weighted loss: Yes (addresses class imbalance)")
    print(f"  Model complexity: Reduced")
    
    # Create trainer
    trainer = ImprovedHeartMurmurTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        class_names=class_names,
        class_weights=class_weights
    )
    
    # Train model for fewer epochs initially
    print(f"\n🚀 Starting training with fixes...")
    best_val_acc = trainer.train(num_epochs=20, save_path="best_heart_murmur_model_fixed.pth")
    
    # Evaluate on test set
    test_acc, predictions, labels = trainer.evaluate("best_heart_murmur_model_fixed.pth")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'best_validation_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc * 100),
        'class_names': class_names,
        'class_weights': class_weights.tolist(),
        'model_parameters': trainable_params,
        'fixes_applied': [
            'Reduced learning rate to 0.0001',
            'Added class weights for imbalanced data',
            'Simplified model architecture',
            'Reduced image size to 128x128',
            'Smaller batch size (8)',
            'Per-class accuracy monitoring'
        ]
    }
    
    with open('training_results_fixed.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 FIXED TRAINING COMPLETED!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final test accuracy: {test_acc*100:.2f}%")
    print(f"Results saved to training_results_fixed.json")
    print(f"Model saved to best_heart_murmur_model_fixed.pth")

if __name__ == "__main__":
    main() 