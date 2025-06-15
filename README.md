# Heart Murmur Classification with CNN

A deep learning project that classifies heart murmur audio recordings using Convolutional Neural Networks (CNN) on mel-spectrogram representations.

## 🎯 Project Overview

This project uses a custom CNN to classify heart sounds into 5 categories:
- **AR**: Aortic Regurgitation (56 samples)
- **AS**: Aortic Stenosis (64 samples) 
- **MR**: Mitral Regurgitation (88 samples)
- **MS**: Mitral Stenosis (88 samples)
- **N**: Normal (168 samples)

**Total Dataset**: 464 audio files converted to optimized mel-spectrograms

## 🏗️ Model Architecture

### Custom CNN Design
```
Input: 224x224x3 RGB spectrograms
├── Conv2D(32) + BatchNorm + ReLU + MaxPool
├── Conv2D(64) + BatchNorm + ReLU + MaxPool  
├── Conv2D(128) + BatchNorm + ReLU + MaxPool
├── AdaptiveAvgPool2d(4x4)
├── Flatten
├── FC(2048 → 256) + ReLU + Dropout(0.5)
├── FC(256 → 128) + ReLU + Dropout(0.5)
└── FC(128 → 5) [Output Classes]
```

### Anti-Overfitting Strategies
- **Dropout**: 50% dropout in fully connected layers
- **Batch Normalization**: After each convolutional layer
- **Early Stopping**: Stops training if validation accuracy doesn't improve for 10 epochs
- **Weight Decay**: L2 regularization (1e-4)
- **Data Augmentation**: Random flips, rotations, color jittering
- **Cross-Validation**: Stratified train/val/test splits

## 📊 Data Processing

### Spectrogram Settings (Optimized)
- **Mel Frequency Bins**: 30 (focused on heart sound frequencies)
- **Frequency Range**: 20Hz - 1kHz (eliminates empty space)
- **FFT Size**: 2048
- **Hop Length**: 512
- **Sample Rate**: 4000 Hz

### Data Splits
- **Training**: 70% (324 samples)
- **Validation**: 15% (70 samples) 
- **Test**: 15% (70 samples)
- **Stratified**: Maintains class distribution across splits

## 🚀 How to Run

### 1. Setup Environment
```bash
# Activate virtual environment
source heart_murmur_env/bin/activate

# Verify PyTorch MPS support (M3 Mac)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 2. Train the Model
```bash
python heart_murmur_cnn.py
```

### 3. What Happens During Training
The script will:
1. **Load Data**: Read spectrogram metadata and create train/val/test splits
2. **Show Class Distribution**: Display how many samples per disease type
3. **Create Model**: Initialize CNN with ~400K parameters
4. **Train**: Run up to 50 epochs with early stopping
5. **Save Best Model**: Automatically saves the best performing model
6. **Generate Plots**: Training history and confusion matrix
7. **Test Evaluation**: Final accuracy on unseen test data

## 📈 Understanding the Output

### During Training
```
Epoch 1/50
--------------------------------------------------
Training: 100%|████████| 21/21 [00:15<00:00,  1.35it/s]
Validation: 100%|████████| 5/5 [00:01<00:00,  4.12it/s]
Train Loss: 1.5234, Train Acc: 32.41%
Val Loss: 1.4567, Val Acc: 38.57%
New best model saved! Val Acc: 38.57%
```

### Key Metrics to Watch
- **Training vs Validation Loss**: Should both decrease
- **Training vs Validation Accuracy**: Gap indicates overfitting
- **Early Stopping**: Prevents overfitting by stopping when validation stops improving

### Final Results
```
Test Accuracy: 0.7143 (71.43%)

Classification Report:
              precision    recall  f1-score   support
        AR       0.67      0.80      0.73         5
        AS       0.75      0.75      0.75         8
        MR       0.71      0.71      0.71        14
        MS       0.80      0.67      0.73        12
         N       0.74      0.74      0.74        31
```

## 📊 Generated Files

After training, you'll get:
- `best_heart_murmur_model.pth`: Trained model weights
- `training_results.json`: Complete training metrics
- `training_history.png`: Loss and accuracy plots
- `confusion_matrix.png`: Detailed classification results
- `class_names.json`: Label encoding mapping

## 🔍 How to Know if Your Model is Good

### Good Signs ✅
- **Test accuracy > 60%**: Better than random (20% for 5 classes)
- **Validation accuracy close to training**: Not overfitting
- **Confusion matrix**: Good diagonal values
- **F1-scores balanced**: All classes performing reasonably

### Warning Signs ⚠️
- **Training accuracy >> Validation accuracy**: Overfitting
- **Loss not decreasing**: Learning rate too high/low
- **One class dominates**: Class imbalance issues
- **Accuracy plateaus early**: Model too simple

### Typical Performance Expectations
- **Beginner Model**: 40-60% accuracy
- **Good Model**: 60-80% accuracy  
- **Excellent Model**: 80%+ accuracy

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA/MPS Errors**
```bash
# Check device
python -c "import torch; print(torch.backends.mps.is_available())"
```

**2. Memory Issues**
- Reduce batch size in `create_data_loaders()` (default: 16)
- Reduce image size from 224x224

**3. Poor Performance**
- Check class distribution balance
- Increase training epochs
- Adjust learning rate (default: 0.001)

**4. Overfitting**
- Increase dropout rate (default: 0.5)
- Add more data augmentation
- Reduce model complexity

## 🔧 Customization Options

### Modify Hyperparameters
```python
# In heart_murmur_cnn.py, change these values:
batch_size = 16          # Reduce if memory issues
learning_rate = 0.001    # Increase if loss not decreasing
dropout_rate = 0.5       # Increase to reduce overfitting
num_epochs = 50          # Increase for more training
```

### Add More Data Augmentation
```python
# In create_data_loaders(), add more transforms:
transforms.RandomVerticalFlip(p=0.3),
transforms.GaussianBlur(kernel_size=3),
transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
```

## 📚 Understanding CNN Concepts

### What is a CNN?
- **Convolutional Layers**: Detect patterns (edges, textures)
- **Pooling Layers**: Reduce size, keep important features
- **Fully Connected**: Final classification decision

### Why CNNs for Audio?
- Spectrograms are images of sound
- CNNs excel at finding visual patterns
- Heart murmurs have distinct frequency signatures

### Training Process
1. **Forward Pass**: Input → Prediction
2. **Loss Calculation**: How wrong was the prediction?
3. **Backward Pass**: Update weights to reduce error
4. **Repeat**: Until model learns patterns

## 🎯 Next Steps

1. **Run the model** and check your first results
2. **Analyze confusion matrix** to see which classes are confused
3. **Experiment with hyperparameters** if performance is low
4. **Try data augmentation** if you see overfitting
5. **Consider ensemble methods** for better performance

## 📞 Need Help?

If you encounter issues:
1. Check the console output for error messages
2. Verify all files are in the correct locations
3. Ensure virtual environment is activated
4. Check that spectrograms were generated correctly

Good luck with your first ML model! 🚀 