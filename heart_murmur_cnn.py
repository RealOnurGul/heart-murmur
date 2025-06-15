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

class HeartMurmurCNN(nn.Module):
    """Custom CNN for heart murmur classification"""
    
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super(HeartMurmurCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        # Note: Input size will be calculated dynamically
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional layers with batch norm and pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Adaptive pooling to handle different input sizes
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class HeartMurmurTrainer:
    """Training and evaluation class"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 criterion, optimizer, device, class_names):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.class_names = class_names
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs, save_path="best_model.pth"):
        """Full training loop"""
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training on {len(self.train_loader.dataset)} samples")
        print(f"Validating on {len(self.val_loader.dataset)} samples")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def evaluate(self, model_path="best_model.pth"):
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
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def prepare_data():
    """Prepare dataset with train/val/test splits"""
    print("Preparing dataset...")
    
    # Load metadata
    df = pd.read_csv('processed_data/complete_metadata.csv')
    print(f"Total samples: {len(df)}")
    
    # Check class distribution
    print("\nClass distribution:")
    print(df['disease'].value_counts())
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['disease'])
    
    # Save label encoder for later use
    class_names = label_encoder.classes_.tolist()
    with open('class_names.json', 'w') as f:
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
    
    return train_df, val_df, test_df, class_names

def create_data_loaders(train_df, val_df, test_df, batch_size=16):
    """Create data loaders with appropriate transforms"""
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
        transforms.RandomRotation(degrees=5),    # Slight rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color variation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = HeartMurmurDataset(train_df, transform=train_transform)
    val_dataset = HeartMurmurDataset(val_df, transform=val_test_transform)
    test_dataset = HeartMurmurDataset(test_df, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def main():
    """Main training function"""
    print("Heart Murmur CNN Training")
    print("=" * 50)
    
    # Prepare data
    train_df, val_df, test_df, class_names = prepare_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, batch_size=16
    )
    
    # Create model
    model = HeartMurmurCNN(num_classes=len(class_names), dropout_rate=0.5)
    model = model.to(device)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Create trainer
    trainer = HeartMurmurTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        class_names=class_names
    )
    
    # Train model
    best_val_acc = trainer.train(num_epochs=50, save_path="best_heart_murmur_model.pth")
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate on test set
    test_acc, predictions, labels = trainer.evaluate("best_heart_murmur_model.pth")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'best_validation_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc * 100),
        'class_names': class_names,
        'model_parameters': trainable_params,
        'dataset_sizes': {
            'train': len(train_df),
            'validation': len(val_df),
            'test': len(test_df)
        }
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final test accuracy: {test_acc*100:.2f}%")
    print(f"Results saved to training_results.json")
    print(f"Model saved to best_heart_murmur_model.pth")

if __name__ == "__main__":
    main() 