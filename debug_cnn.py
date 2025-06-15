import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

def analyze_data_distribution():
    """Analyze the data distribution and potential issues"""
    print("🔍 DEBUGGING CNN TRAINING ISSUES")
    print("=" * 50)
    
    # Load metadata
    df = pd.read_csv('processed_data/complete_metadata.csv')
    print(f"Total samples: {len(df)}")
    
    # Check class distribution
    print("\n📊 Class Distribution:")
    class_counts = df['disease'].value_counts()
    for disease, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {disease}: {count} samples ({percentage:.1f}%)")
    
    # Calculate what 37.14% accuracy means
    print(f"\n🎯 37.14% accuracy analysis:")
    total_samples = len(df)
    val_samples = int(total_samples * 0.15)  # 15% for validation
    correct_predictions = int(val_samples * 0.3714)
    print(f"  Validation samples: {val_samples}")
    print(f"  Correct predictions at 37.14%: {correct_predictions}")
    
    # Check if model is predicting majority class
    majority_class = class_counts.index[0]
    majority_count = class_counts.iloc[0]
    majority_percentage = (majority_count / len(df)) * 100
    
    print(f"\n🤔 Majority class analysis:")
    print(f"  Majority class: {majority_class} ({majority_percentage:.1f}%)")
    print(f"  If predicting only majority class: ~{majority_percentage:.1f}% accuracy")
    
    # Check for potential issues
    print(f"\n⚠️  Potential Issues:")
    
    # Issue 1: Class imbalance
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class
    print(f"  1. Class imbalance ratio: {imbalance_ratio:.1f}:1 ({max_class} vs {min_class})")
    
    if imbalance_ratio > 3:
        print(f"     ❌ SEVERE IMBALANCE! Model likely predicting majority class")
    
    # Issue 2: Check stratified split
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['disease'])
    
    try:
        train_df, temp_df = train_test_split(df, test_size=0.3, 
                                            stratify=df['label_encoded'], 
                                            random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, 
                                          stratify=temp_df['label_encoded'], 
                                          random_state=42)
        
        print(f"\n📈 Validation set distribution:")
        val_class_counts = val_df['disease'].value_counts()
        for disease, count in val_class_counts.items():
            percentage = (count / len(val_df)) * 100
            print(f"  {disease}: {count} samples ({percentage:.1f}%)")
            
        # Check if 37.14% matches any specific pattern
        val_majority = val_class_counts.iloc[0]
        val_majority_pct = (val_majority / len(val_df)) * 100
        print(f"\n🎯 Validation majority class accuracy: {val_majority_pct:.2f}%")
        
        if abs(val_majority_pct - 37.14) < 1:
            print(f"     ❌ FOUND THE ISSUE! Model is predicting majority class only")
        
    except Exception as e:
        print(f"  2. Stratification error: {e}")
    
    # Issue 3: Learning rate problems
    print(f"\n🧠 Likely Root Causes:")
    print(f"  1. Learning rate too high → Model not learning")
    print(f"  2. Class imbalance → Model defaults to majority class")
    print(f"  3. Loss function issue → CrossEntropyLoss with imbalanced data")
    print(f"  4. Model too complex → Overfitting immediately")
    
    return df, val_df

def suggest_fixes():
    """Suggest specific fixes for the issues"""
    print(f"\n🔧 RECOMMENDED FIXES:")
    print("=" * 50)
    
    print("1. 📉 REDUCE LEARNING RATE:")
    print("   - Change from 0.001 to 0.0001")
    print("   - Add learning rate scheduler")
    
    print("\n2. ⚖️  HANDLE CLASS IMBALANCE:")
    print("   - Use weighted CrossEntropyLoss")
    print("   - Add class weights inversely proportional to frequency")
    
    print("\n3. 🏗️  SIMPLIFY MODEL:")
    print("   - Reduce model complexity")
    print("   - Start with smaller learning rate")
    
    print("\n4. 📊 BETTER MONITORING:")
    print("   - Print predictions per class")
    print("   - Monitor if model is stuck predicting one class")
    
    print("\n5. 🎯 IMMEDIATE TEST:")
    print("   - Train for just 3 epochs with verbose output")
    print("   - Check if validation accuracy changes at all")

if __name__ == "__main__":
    df, val_df = analyze_data_distribution()
    suggest_fixes()
    
    print(f"\n🚀 NEXT STEPS:")
    print("1. Run this diagnostic")
    print("2. Apply the recommended fixes")
    print("3. Test with a simplified version first") 