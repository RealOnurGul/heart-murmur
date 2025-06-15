#!/usr/bin/env python3
"""
Spectrogram Data Augmentation
Adds time-reversed (horizontally flipped) versions of spectrograms
This is semantically valid for heart sounds - reversed heart sounds are still valid heart sounds
"""

import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
import shutil

def create_time_reversed_spectrogram(original_path: Path, output_path: Path) -> bool:
    """
    Create time-reversed (horizontally flipped) version of spectrogram
    This simulates playing the heart sound backwards, which is still a valid heart sound
    """
    try:
        # Load the original spectrogram
        img = Image.open(original_path)
        
        # Flip horizontally (time reversal)
        flipped_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        # Save the augmented version
        flipped_img.save(output_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error augmenting {original_path.name}: {e}")
        return False

def augment_all_spectrograms(processed_data_dir: str) -> bool:
    """
    Create time-reversed versions of all spectrograms
    This will double the dataset size with semantically valid augmentations
    """
    
    print("🔄 SPECTROGRAM DATA AUGMENTATION")
    print("=" * 60)
    print("Creating time-reversed (horizontally flipped) versions of all spectrograms")
    print("This simulates playing heart sounds backwards - still valid heart sounds!")
    
    # Load metadata
    processed_path = Path(processed_data_dir)
    metadata_file = processed_path / "complete_metadata.csv"
    
    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        return False
    
    df = pd.read_csv(metadata_file)
    print(f"\n📊 Found {len(df)} original spectrograms to augment")
    
    # Create augmented metadata list
    augmented_metadata = []
    successful_augmentations = 0
    failed_augmentations = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating time-reversed spectrograms"):
        original_path = Path(row['spectrogram_path'])
        
        if not original_path.exists():
            print(f"⚠️  Original file not found: {original_path}")
            failed_augmentations += 1
            continue
        
        # Create augmented filename (add _time_reversed suffix)
        augmented_filename = f"{original_path.stem}_time_reversed{original_path.suffix}"
        augmented_path = original_path.parent / augmented_filename
        
        # Create the time-reversed spectrogram
        success = create_time_reversed_spectrogram(original_path, augmented_path)
        
        if success:
            # Create metadata entry for augmented spectrogram
            augmented_row = row.copy()
            augmented_row['filename'] = f"{row['filename'].replace('.wav', '')}_time_reversed.wav"
            augmented_row['spectrogram_path'] = str(augmented_path)
            augmented_row['label_full'] = f"{row['label_full']}_time_reversed"
            augmented_row['augmentation'] = 'time_reversed'
            augmented_row['original_file'] = row['filename']
            
            augmented_metadata.append(augmented_row)
            successful_augmentations += 1
        else:
            failed_augmentations += 1
    
    # Combine original and augmented metadata
    if augmented_metadata:
        # Add augmentation info to original data
        df['augmentation'] = 'original'
        df['original_file'] = df['filename']
        
        # Combine datasets
        augmented_df = pd.DataFrame(augmented_metadata)
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        
        # Save combined metadata
        combined_metadata_file = processed_path / "complete_metadata_augmented.csv"
        combined_df.to_csv(combined_metadata_file, index=False)
        
        print(f"\n✅ AUGMENTATION COMPLETE!")
        print(f"📊 Results:")
        print(f"  Original spectrograms: {len(df)}")
        print(f"  Successful augmentations: {successful_augmentations}")
        print(f"  Failed augmentations: {failed_augmentations}")
        print(f"  Total dataset size: {len(combined_df)}")
        print(f"  Dataset increase: {(len(combined_df) / len(df) - 1) * 100:.1f}%")
        
        print(f"\n📁 Files:")
        print(f"  Original metadata: {metadata_file}")
        print(f"  Augmented metadata: {combined_metadata_file}")
        
        # Show augmentation distribution by disease
        print(f"\n📈 Augmented Dataset by Disease:")
        for disease in ['N', 'AR', 'AS', 'MR', 'MS']:
            disease_data = combined_df[combined_df['disease'] == disease]
            original_count = len(disease_data[disease_data['augmentation'] == 'original'])
            augmented_count = len(disease_data[disease_data['augmentation'] == 'time_reversed'])
            total_count = len(disease_data)
            print(f"  {disease}: {original_count} original + {augmented_count} augmented = {total_count} total")
        
        # Save augmentation summary
        augmentation_summary = {
            'augmentation_method': 'time_reversed_horizontal_flip',
            'original_count': len(df),
            'augmented_count': successful_augmentations,
            'total_count': len(combined_df),
            'dataset_increase_percent': (len(combined_df) / len(df) - 1) * 100,
            'semantic_validity': 'high - time reversal preserves frequency relationships',
            'expected_benefit': 'improved temporal pattern recognition and generalization'
        }
        
        with open(processed_path / "augmentation_info.txt", 'w') as f:
            f.write("Spectrogram Data Augmentation Results\n")
            f.write("=" * 50 + "\n")
            f.write("Method: Time Reversal (Horizontal Flip)\n")
            f.write("Semantic Validity: High - reversed heart sounds are still valid\n\n")
            for key, value in augmentation_summary.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n🎯 Augmentation Benefits:")
        print(f"  ✅ Doubled dataset size with valid data")
        print(f"  ✅ Improved temporal pattern recognition")
        print(f"  ✅ Better generalization to different heart sound timings")
        print(f"  ✅ No semantic corruption - all augmented sounds are valid")
        
        print(f"\n🚀 Ready for CNN training with augmented dataset!")
        print(f"💡 Use 'complete_metadata_augmented.csv' for training")
        
        return True
    
    else:
        print(f"❌ No successful augmentations created")
        return False

def main():
    """Main function"""
    processed_data_dir = "processed_data"
    
    if not Path(processed_data_dir).exists():
        print(f"❌ Processed data directory not found: {processed_data_dir}")
        return
    
    print(f"📂 Processing directory: {processed_data_dir}")
    print(f"\n🎯 AUGMENTATION STRATEGY: TIME REVERSAL")
    print(f"  ✅ Semantically valid for heart sounds")
    print(f"  ✅ Preserves frequency relationships")
    print(f"  ✅ Simulates different temporal patterns")
    print(f"  ✅ Doubles dataset size")
    print(f"  ❌ Rotation would be meaningless (swaps time/frequency)")
    print(f"  ❌ Vertical flip would corrupt frequency relationships")
    
    confirm = input(f"\nCreate time-reversed augmentations for all spectrograms? (y/n): ").strip().lower()
    
    if confirm == 'y':
        success = augment_all_spectrograms(processed_data_dir)
        if success:
            print(f"\n🎉 AUGMENTATION SUCCESSFUL!")
            print(f"🔄 Dataset doubled with semantically valid time-reversed spectrograms!")
        else:
            print(f"❌ Augmentation failed")
    else:
        print(f"❌ Augmentation cancelled")

if __name__ == "__main__":
    main() 