import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
import os
from tqdm import tqdm

# Set device
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
        img_path = self.dataframe.iloc[idx]['spectrogram_path']
        label = self.dataframe.iloc[idx]['label_encoded']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class SimplerHeartMurmurCNN(nn.Module):
    """Simplified CNN architecture (same as training)"""
    
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super(SimplerHeartMurmurCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(4, 4)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def prepare_test_data():
    """Prepare the same data splits as training"""
    print("🔄 Preparing test data...")
    
    # Load metadata
    df = pd.read_csv('processed_data/complete_metadata.csv')
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['disease'])
    
    # Same splits as training (random_state=42)
    train_df, temp_df = train_test_split(df, test_size=0.3, 
                                        stratify=df['label_encoded'], 
                                        random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, 
                                      stratify=temp_df['label_encoded'], 
                                      random_state=42)
    
    class_names = label_encoder.classes_.tolist()
    
    print(f"✅ Data prepared:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples") 
    print(f"  Test: {len(test_df)} samples")
    print(f"  Classes: {class_names}")
    
    return train_df, val_df, test_df, class_names

def create_test_loaders(train_df, val_df, test_df, batch_size=8):
    """Create data loaders for testing"""
    
    # Same transforms as training
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = HeartMurmurDataset(train_df, transform=transform)
    val_dataset = HeartMurmurDataset(val_df, transform=transform)
    test_dataset = HeartMurmurDataset(test_df, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def evaluate_model(model, data_loader, class_names, dataset_name):
    """Evaluate model on a dataset"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"\n📊 {dataset_name} Results:")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(all_predictions)[class_mask] == i)
            class_count = np.sum(class_mask)
            correct_count = np.sum((np.array(all_predictions)[class_mask] == i))
            print(f"  {class_name}: {class_acc:.3f} ({class_acc*100:.1f}%) - {correct_count}/{class_count}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=class_names, digits=3))
    
    return accuracy, all_predictions, all_labels, all_probabilities

def plot_confusion_matrix(y_true, y_pred, class_names, dataset_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.title(f'Confusion Matrix - {dataset_name} Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    filename = f'confusion_matrix_{dataset_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📈 Confusion matrix saved as {filename}")
    plt.show()

def analyze_predictions(y_true, y_pred, class_names):
    """Analyze prediction patterns"""
    print(f"\n🔍 Prediction Analysis:")
    
    # Check which classes are being predicted
    unique_preds = np.unique(y_pred)
    predicted_classes = [class_names[i] for i in unique_preds]
    print(f"Classes being predicted: {predicted_classes}")
    
    if len(unique_preds) < len(class_names):
        missing_classes = [class_names[i] for i in range(len(class_names)) if i not in unique_preds]
        print(f"⚠️  Classes NEVER predicted: {missing_classes}")
    
    # Prediction distribution
    print(f"\nPrediction distribution:")
    for i, class_name in enumerate(class_names):
        pred_count = np.sum(np.array(y_pred) == i)
        total_count = len(y_pred)
        percentage = (pred_count / total_count) * 100
        print(f"  {class_name}: {pred_count}/{total_count} ({percentage:.1f}%)")

def main():
    """Main testing function"""
    print("🧪 COMPREHENSIVE MODEL TESTING")
    print("=" * 60)
    
    # Check if model file exists
    model_path = "best_heart_murmur_model_fixed.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found!")
        print("Available model files:")
        for file in os.listdir('.'):
            if file.endswith('.pth'):
                print(f"  - {file}")
        return
    
    # Prepare data
    train_df, val_df, test_df, class_names = prepare_test_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_test_loaders(
        train_df, val_df, test_df, batch_size=8
    )
    
    # Load model
    print(f"\n🤖 Loading model from {model_path}...")
    model = SimplerHeartMurmurCNN(num_classes=len(class_names), dropout_rate=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Evaluate on all datasets
    print(f"\n🎯 EVALUATION RESULTS")
    print("=" * 60)
    
    # Test set (most important)
    test_acc, test_preds, test_labels, test_probs = evaluate_model(
        model, test_loader, class_names, "Test"
    )
    
    # Validation set
    val_acc, val_preds, val_labels, val_probs = evaluate_model(
        model, val_loader, class_names, "Validation"
    )
    
    # Training set (to check overfitting)
    train_acc, train_preds, train_labels, train_probs = evaluate_model(
        model, train_loader, class_names, "Training"
    )
    
    # Summary
    print(f"\n📋 SUMMARY")
    print("=" * 60)
    print(f"Training Accuracy:   {train_acc*100:.2f}%")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Test Accuracy:       {test_acc*100:.2f}%")
    
    # Check for overfitting
    if train_acc - test_acc > 0.15:
        print(f"⚠️  Possible overfitting detected (train-test gap: {(train_acc-test_acc)*100:.1f}%)")
    else:
        print(f"✅ Good generalization (train-test gap: {(train_acc-test_acc)*100:.1f}%)")
    
    # Analyze predictions
    analyze_predictions(test_labels, test_preds, class_names)
    
    # Plot confusion matrices
    plot_confusion_matrix(test_labels, test_preds, class_names, "Test")
    plot_confusion_matrix(val_labels, val_preds, class_names, "Validation")
    
    # Performance vs random baseline
    random_acc = 1.0 / len(class_names)
    improvement = (test_acc - random_acc) / random_acc * 100
    print(f"\n🎲 Baseline Comparison:")
    print(f"Random baseline: {random_acc*100:.1f}%")
    print(f"Our model: {test_acc*100:.1f}%")
    print(f"Improvement: {improvement:.1f}% better than random")
    
    # Save results
    results = {
        'model_file': model_path,
        'model_parameters': total_params,
        'training_accuracy': float(train_acc),
        'validation_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'class_names': class_names,
        'improvement_over_random': float(improvement),
        'overfitting_gap': float(train_acc - test_acc)
    }
    
    with open('comprehensive_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to comprehensive_test_results.json")
    print(f"\n🎉 Testing completed successfully!")

if __name__ == "__main__":
    main() 