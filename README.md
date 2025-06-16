# Heart Murmur Classification using Deep Learning

A convolutional neural network (CNN) for classifying heart murmurs from audio spectrograms. This project achieves **71.14% validation accuracy** on a 5-class heart murmur classification task.

## Overview

This project implements a deep learning solution for automated heart murmur detection and classification. The model analyzes spectrogram representations of heart sounds to distinguish between different types of heart conditions.

### Classification Categories
- **N** - Normal (no murmur)
- **AS** - Aortic Stenosis  
- **AR** - Aortic Regurgitation
- **MS** - Mitral Stenosis
- **MR** - Mitral Regurgitation

## Model Performance

- **Validation Accuracy**: 71.14%
- **Training Dataset**: 928 samples (464 original + 464 augmented)
- **Model Architecture**: Custom CNN optimized for 224x224 spectrogram images
- **Parameters**: 430,853 trainable parameters

## Dataset

The dataset consists of heart sound recordings processed into spectrogram images:

### Class Distribution
- Normal (N): 336 samples (36.2%)
- Mitral Stenosis (MS): 176 samples (19.0%) 
- Mitral Regurgitation (MR): 176 samples (19.0%)
- Aortic Stenosis (AS): 128 samples (13.8%)
- Aortic Regurgitation (AR): 112 samples (12.1%)

### Data Augmentation
- Time-reversal augmentation doubles the dataset size
- Random horizontal flipping during training
- Normalization using ImageNet statistics

## Architecture

The CNN architecture is specifically designed for heart murmur classification:

```
Input: 224x224x3 spectrogram images
├── Feature Extraction Blocks (4 blocks)
│   ├── Conv2D + BatchNorm + ReLU + MaxPool + Dropout
│   └── Progressively increasing channels: 32→64→128→256
├── Global Average Pooling
└── Classification Head
    ├── Dropout + Linear(256→128) + ReLU
    ├── Dropout + Linear(128→64) + ReLU  
    └── Linear(64→5) [output classes]
```

### Key Features
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout Regularization**: Prevents overfitting (rates: 0.1-0.5)
- **Global Average Pooling**: Reduces parameters and overfitting risk
- **Class-weighted Loss**: Handles dataset imbalance effectively
- **Gradient Clipping**: Prevents exploding gradients

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd heart-murmur
```

2. Create and activate virtual environment:
```bash
python -m venv heart_murmur_env
source heart_murmur_env/bin/activate  # On Windows: heart_murmur_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main training script:
```bash
python heart_murmur_cnn.py
```

The script will:
1. Load and validate the spectrogram dataset
2. Split data into train/validation/test sets (60%/20%/20%)
3. Train the CNN for up to 30 epochs
4. Save the best model based on validation accuracy
5. Output training progress and final performance metrics

### Model Files

After training, the following files are generated:
- `best_heart_murmur_model.pth` - Best model weights
- `class_names.json` - Class label mappings

### Training Configuration

- **Optimizer**: AdamW with learning rate 0.001
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Batch Size**: 16
- **Weight Decay**: 0.01
- **Early Stopping**: Stops at 75% accuracy or if stuck predicting single class

## Data Processing Pipeline

1. **Audio Collection**: Heart sound recordings in WAV format
2. **Spectrogram Generation**: Convert audio to visual spectrograms (1500x600 pixels)
3. **Preprocessing**: Resize to 224x224, normalize, and apply augmentations
4. **Training**: Feed processed spectrograms to CNN

## Results Analysis

The model demonstrates strong performance across all heart murmur types:

### Key Achievements
- **71.14% validation accuracy** - significantly above random chance (20%)
- **Balanced predictions** across all 5 classes during validation
- **Stable training** with consistent improvement over epochs
- **Effective handling** of class imbalance through weighted loss

### Training Insights
- Model learns meaningful patterns from spectrogram frequency representations
- Class-weighted loss function crucial for handling imbalanced dataset
- Global average pooling prevents overfitting better than fully connected layers
- Proper data augmentation improves generalization without overfitting

## Technical Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- scikit-learn
- pandas
- numpy
- PIL (Pillow)
- matplotlib
- tqdm

### Hardware Recommendations
- **GPU**: CUDA-compatible GPU or Apple Silicon (MPS) for faster training
- **RAM**: 8GB+ recommended for dataset loading
- **Storage**: 2GB+ for dataset and model files

## Project Structure

```
heart-murmur/
├── heart_murmur_cnn.py          # Main training script
├── processed_data/              # Dataset and spectrograms
│   ├── spectrograms/           # Generated spectrogram images
│   └── complete_metadata_augmented.csv
├── raw/                        # Original audio files
├── failed_attempts/            # Previous model iterations
├── best_heart_murmur_model.pth # Trained model weights
├── class_names.json           # Class mappings
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Future Improvements

Potential areas for enhancement:
- **Ensemble Methods**: Combine multiple models for better accuracy
- **Advanced Augmentation**: Spectral augmentation techniques
- **Transfer Learning**: Pre-trained models on medical audio data
- **Real-time Inference**: Optimize model for live audio classification
- **Larger Dataset**: More diverse heart sound recordings
- **Cross-validation**: More robust performance evaluation

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is for educational and research purposes. Please ensure compliance with medical data regulations when using with real patient data.

---

**Note**: This model is for research purposes only and should not be used for medical diagnosis without proper validation and regulatory approval. 