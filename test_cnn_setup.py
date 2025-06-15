#!/usr/bin/env python3
"""
Test script to verify CNN setup and dependencies
Run this before training to catch any issues early
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}")
        
        import librosa
        print(f"✅ Librosa {librosa.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
        
        import seaborn as sns
        print(f"✅ Seaborn {sns.__version__}")
        
        from PIL import Image
        print(f"✅ Pillow (PIL)")
        
        import tqdm
        print(f"✅ tqdm {tqdm.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_device():
    """Test PyTorch device availability"""
    print("\nTesting PyTorch device...")
    
    import torch
    
    # Check MPS (M3 Mac)
    if torch.backends.mps.is_available():
        print("✅ MPS (M3 Mac GPU) available")
        device = torch.device("mps")
    else:
        print("⚠️  MPS not available, using CPU")
        device = torch.device("cpu")
    
    # Test tensor creation
    try:
        x = torch.randn(2, 3, 224, 224).to(device)
        print(f"✅ Created test tensor on {device}: {x.shape}")
        return True
    except Exception as e:
        print(f"❌ Device test failed: {e}")
        return False

def test_model_creation():
    """Test CNN model creation"""
    print("\nTesting model creation...")
    
    try:
        # Import our CNN
        from heart_murmur_cnn import HeartMurmurCNN
        
        # Create model
        model = HeartMurmurCNN(num_classes=5, dropout_rate=0.5)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {total_params:,} parameters")
        
        # Test forward pass
        import torch
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
        
        x = torch.randn(1, 3, 224, 224).to(device)
        output = model(x)
        print(f"✅ Forward pass successful: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_data_files():
    """Test required data files exist"""
    print("\nTesting data files...")
    
    required_files = [
        'processed_data/complete_metadata.csv',
        'processed_data/spectrograms'  # directory
    ]
    
    all_good = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            if file_path == 'processed_data/spectrograms':
                # Count spectrograms in subdirectories
                spec_count = 0
                for subdir in os.listdir(file_path):
                    subdir_path = os.path.join(file_path, subdir)
                    if os.path.isdir(subdir_path):
                        spec_count += len([f for f in os.listdir(subdir_path) 
                                         if f.endswith('.png')])
                print(f"✅ {file_path}/ directory exists with {spec_count} PNG files")
            else:
                print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} not found")
            all_good = False
    
    return all_good

def test_metadata():
    """Test metadata file structure"""
    print("\nTesting metadata structure...")
    
    try:
        import pandas as pd
        
        df = pd.read_csv('processed_data/complete_metadata.csv')
        print(f"✅ Metadata loaded: {len(df)} rows")
        
        required_columns = ['disease', 'spectrogram_path']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        
        print(f"✅ Required columns present: {required_columns}")
        
        # Check class distribution
        print("\nClass distribution:")
        for disease, count in df['disease'].value_counts().items():
            print(f"  {disease}: {count} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Heart Murmur CNN Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Device Test", test_device),
        ("Data Files Test", test_data_files),
        ("Metadata Test", test_metadata),
        ("Model Creation Test", test_model_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready to train your CNN!")
        print("\nNext step: Run 'python heart_murmur_cnn.py' to start training")
    else:
        print("⚠️  SOME TESTS FAILED. Please fix issues before training.")
        print("\nCheck the error messages above and:")
        print("1. Ensure virtual environment is activated")
        print("2. Install missing packages: pip install -r requirements.txt")
        print("3. Verify spectrogram files were generated")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 