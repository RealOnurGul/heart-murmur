import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class SpectrogramDataset(Dataset):
    """Dataset class for loading heart murmur spectrograms."""
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
        # Create label mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(dataframe['disease'].unique()))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Label mapping: {self.label_to_idx}")
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Load spectrogram image
        image_path = row['spectrogram_path']
        if pd.isna(image_path) or not os.path.exists(image_path):
            print(f"Missing spectrogram: {image_path}")
            # Create a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.label_to_idx[row['disease']]
        
        return image, label

class HeartMurmurCNN(nn.Module):
    """
    Convolutional Neural Network for heart murmur classification from spectrograms.
    
    Architecture optimized for 224x224 spectrogram images with 5 classes:
    - AR (Aortic Regurgitation)
    - AS (Aortic Stenosis) 
    - MR (Mitral Regurgitation)
    - MS (Mitral Stenosis)
    - N (Normal)
    """
    
    def __init__(self, num_classes=5):
        super(HeartMurmurCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: Initial feature extraction
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            nn.Dropout2d(0.1),
            
            # Block 2: Deeper feature extraction
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            nn.Dropout2d(0.1),
            
            # Block 3: Complex pattern recognition
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28
            nn.Dropout2d(0.2),
            
            # Block 4: High-level feature extraction
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14
            nn.Dropout2d(0.2),
        )
        
        # Global Average Pooling to reduce overfitting and avoid MPS compatibility issues
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_and_validate_data():
    """Load and validate the spectrogram dataset."""
    print("Loading and validating spectrogram data...")
    
    # Load metadata
    df = pd.read_csv('processed_data/complete_metadata_augmented.csv')
    
    # Remove rows with missing spectrograms
    df = df.dropna(subset=['spectrogram_path'])
    df = df[df['spectrogram_path'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)]
    
    print(f"Dataset size after filtering: {len(df)} samples")
    print(f"Class distribution:")
    class_counts = df['disease'].value_counts()
    print(class_counts)
    
    # Check a few sample spectrograms
    print("\nValidating sample spectrograms...")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        image_path = row['spectrogram_path']
        disease = row['disease']
        
        if os.path.exists(image_path):
            img = Image.open(image_path)
            print(f"  {disease}: {image_path} - Size: {img.size}, Mode: {img.mode}")
        else:
            print(f"  {disease}: {image_path} - MISSING!")
    
    # Check image diversity to ensure spectrograms contain meaningful patterns
    print("\nAnalyzing spectrogram diversity...")
    sample_images = []
    for i in range(min(10, len(df))):
        img_path = df.iloc[i]['spectrogram_path']
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img_array = np.array(img)
            sample_images.append(img_array.flatten())
    
    if len(sample_images) > 1:
        # Calculate variance across images
        img_stack = np.stack(sample_images)
        variance = np.var(img_stack, axis=0).mean()
        print(f"Average pixel variance across spectrograms: {variance:.4f}")
        if variance < 100:
            print("Warning: Spectrograms have very low variance - they might be too similar!")
        else:
            print("Good: Spectrograms have sufficient variance for learning!")
    
    return df

def train_model():
    """Train the heart murmur classification model."""
    
    # Load and validate data
    df = load_and_validate_data()
    
    # Set device (MPS for Apple Silicon, CPU otherwise)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Split data into train/validation/test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['disease'])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['disease'])
    
    print(f"Data splits - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Data preprocessing and augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize spectrograms to optimal CNN input size
        transforms.RandomHorizontalFlip(p=0.3),  # Light augmentation for better generalization
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and data loaders
    train_dataset = SpectrogramDataset(train_df, transform=train_transform)
    val_dataset = SpectrogramDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize model
    model = HeartMurmurCNN(num_classes=len(train_dataset.label_to_idx))
    model = model.to(device)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Handle class imbalance with weighted loss
    class_counts = train_df['disease'].value_counts()
    total_samples = len(train_df)
    class_weights = []
    for disease in sorted(train_df['disease'].unique()):
        weight = total_samples / (len(class_counts) * class_counts[disease])
        class_weights.append(weight)
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights for imbalanced data: {dict(zip(sorted(train_df['disease'].unique()), class_weights.cpu().numpy()))}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training configuration
    num_epochs = 30
    best_val_acc = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Debug first few batches of first epoch
            if epoch == 0 and batch_idx < 2:
                print(f"\n  Batch {batch_idx}: Labels = {labels.cpu().numpy()}")
                print(f"  Batch {batch_idx}: Predictions = {predicted.cpu().numpy()}")
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct_train/total_train:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_predictions = []
        all_val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                all_val_predictions.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct_val / total_val
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Show prediction distribution for monitoring
        unique_preds, pred_counts = np.unique(all_val_predictions, return_counts=True)
        pred_dist = dict(zip(unique_preds, pred_counts))
        print(f"  Validation Predictions Distribution: {pred_dist}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'label_to_idx': train_dataset.label_to_idx,
                'epoch': epoch
            }, 'best_heart_murmur_model.pth')
            print(f'  New best model saved! Validation Accuracy: {val_acc:.2f}%')
        
        # Early stopping conditions
        if val_acc >= 75.0:
            print(f"Reached excellent accuracy! Validation Accuracy: {val_acc:.2f}%")
            break
        
        # Stop if model is stuck predicting single class
        if len(pred_dist) == 1 and epoch > 5:
            print(f"Model stuck predicting single class. Stopping training.")
            break
    
    print(f"\nTraining completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    if best_val_acc > 60.0:
        print("SUCCESS: Model achieved good performance on heart murmur classification!")
    else:
        print("Training completed but accuracy could be improved.")
    
    # Save final class mapping
    with open('class_names.json', 'w') as f:
        json.dump(train_dataset.idx_to_label, f)
    
    return best_val_acc

if __name__ == "__main__":
    print("Heart Murmur Classification CNN")
    print("=" * 50)
    print("Training deep learning model for heart murmur detection from spectrograms")
    print()
    
    final_accuracy = train_model()
    
    print(f"\nFinal Model Performance: {final_accuracy:.2f}% validation accuracy")
    print("Model saved as: best_heart_murmur_model.pth")
    print("Class mapping saved as: class_names.json") 