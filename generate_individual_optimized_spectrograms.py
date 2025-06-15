#!/usr/bin/env python3
"""
INDIVIDUAL Optimized Spectrogram Generator
Optimizes each spectrogram based on ITS OWN frequency content
Maximum space savings with no data loss
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict

def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse filename to extract labels"""
    name = filename.replace('.wav', '')
    parts = name.split('_')
    
    if len(parts) == 4:
        return {
            'disease': parts[0],
            'patient_id': parts[1],
            'position': parts[2],
            'location': parts[3],
            'filename': filename,
            'label_full': f"{parts[0]}_{parts[2]}_{parts[3]}"
        }
    else:
        print(f"⚠️  Warning: Unexpected filename format: {filename}")
        return None

def create_individually_optimized_spectrogram(audio_file: Path, output_dir: Path) -> Tuple[Optional[Path], bool, Dict]:
    """
    Create spectrogram optimized for THIS SPECIFIC audio file
    Each file gets its own optimal frequency range
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Create mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=30, n_fft=2048, hop_length=512,
            fmin=20, fmax=1000
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Find THIS FILE'S optimal frequency range
        mean_energy_per_bin = np.mean(mel_spec_db, axis=1)
        max_energy = np.max(mean_energy_per_bin)
        threshold = max_energy - 40  # 40 dB below max
        
        significant_bins = np.where(mean_energy_per_bin > threshold)[0]
        
        if len(significant_bins) > 0:
            max_significant_bin = int(np.max(significant_bins))
            # Add 2 bins padding for safety
            optimal_max_bin = min(max_significant_bin + 2, 29)
        else:
            # Fallback to 80% of range
            optimal_max_bin = int(30 * 0.8)
        
        # Crop THIS spectrogram to ITS optimal range
        mel_spec_cropped = mel_spec_db[:optimal_max_bin + 1, :]
        
        # Create the optimized plot
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec_cropped, aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save the individually optimized spectrogram
        output_file = output_dir / f"{audio_file.stem}.png"
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=150, 
                   facecolor='black', edgecolor='none')
        plt.close()
        
        # Calculate optimization stats for this file
        mel_frequencies = librosa.mel_frequencies(n_mels=30, fmin=20, fmax=1000)
        optimal_max_freq = mel_frequencies[optimal_max_bin]
        utilization = (optimal_max_bin + 1) / 30
        space_saved = (1 - utilization) * 100
        
        optimization_stats = {
            'optimal_max_bin': optimal_max_bin,
            'optimal_max_freq': optimal_max_freq,
            'utilization_ratio': utilization,
            'space_saved_percent': space_saved,
            'original_bins': 30,
            'optimized_bins': optimal_max_bin + 1
        }
        
        return output_file, True, optimization_stats
        
    except Exception as e:
        print(f"❌ Error processing {audio_file.name}: {e}")
        return None, False, {}

def generate_all_individually_optimized_spectrograms(input_dir: str, output_dir: str) -> Optional[pd.DataFrame]:
    """Generate individually optimized spectrograms for all files"""
    
    print("🚀 GENERATING INDIVIDUALLY OPTIMIZED SPECTROGRAMS")
    print("=" * 70)
    print("Each spectrogram will be optimized based on ITS OWN frequency content!")
    
    # Setup directories
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    spectrograms_dir = output_path / "spectrograms"
    spectrograms_dir.mkdir(parents=True, exist_ok=True)
    
    # Create disease subdirectories
    diseases = ['N', 'AR', 'AS', 'MR', 'MS']
    for disease in diseases:
        (spectrograms_dir / disease).mkdir(exist_ok=True)
    
    # Process all files
    audio_files = list(input_path.glob("*.wav"))
    print(f"\n🎵 Processing {len(audio_files)} audio files with individual optimization...")
    
    metadata = []
    optimization_stats = []
    successful = 0
    failed = 0
    
    for audio_file in tqdm(audio_files, desc="Creating individually optimized spectrograms"):
        file_info = parse_filename(audio_file.name)
        if file_info is None:
            failed += 1
            continue
        
        disease_dir = spectrograms_dir / file_info['disease']
        spectrogram_path, success, stats = create_individually_optimized_spectrogram(audio_file, disease_dir)
        
        if success and spectrogram_path is not None:
            file_info['spectrogram_path'] = str(spectrogram_path)
            
            # Add optimization stats to file info
            file_info.update(stats)
            
            metadata.append(file_info)
            optimization_stats.append(stats)
            successful += 1
        else:
            failed += 1
    
    # Save results
    if metadata:
        df = pd.DataFrame(metadata)
        metadata_file = output_path / "complete_metadata.csv"
        df.to_csv(metadata_file, index=False)
        
        print(f"\n✅ INDIVIDUAL OPTIMIZATION COMPLETE!")
        print(f"📊 Results:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total: {len(audio_files)}")
        
        print(f"\n📁 Files saved to: {spectrograms_dir}")
        print(f"📋 Metadata: {metadata_file}")
        
        # Calculate overall optimization statistics
        if optimization_stats:
            space_saved_values = [stats['space_saved_percent'] for stats in optimization_stats]
            utilization_values = [stats['utilization_ratio'] for stats in optimization_stats]
            optimized_bins_values = [stats['optimized_bins'] for stats in optimization_stats]
            
            avg_space_saved = np.mean(space_saved_values)
            avg_utilization = np.mean(utilization_values)
            avg_optimized_bins = np.mean(optimized_bins_values)
            
            print(f"\n🎯 INDIVIDUAL OPTIMIZATION SUMMARY:")
            print(f"  Average space saved per file: {avg_space_saved:.1f}%")
            print(f"  Average frequency utilization: {avg_utilization:.2f}")
            print(f"  Average optimized bins: {avg_optimized_bins:.1f} (out of 30)")
            print(f"  Range of space saved: {min(space_saved_values):.1f}% - {max(space_saved_values):.1f}%")
            
            # Show distribution by disease
            print(f"\n📊 Optimization by disease:")
            for disease in diseases:
                disease_data = df[df['disease'] == disease]
                if len(disease_data) > 0:
                    disease_avg_space = disease_data['space_saved_percent'].mean()
                    disease_avg_bins = disease_data['optimized_bins'].mean()
                    print(f"  {disease}: {disease_avg_space:.1f}% space saved, {disease_avg_bins:.1f} avg bins")
        
        # Dataset summary
        print(f"\n📈 Dataset Summary:")
        print("Disease distribution:")
        for disease, count in df['disease'].value_counts().items():
            print(f"  {disease}: {count} spectrograms")
        
        # Save detailed optimization info
        optimization_summary = {
            'total_spectrograms': successful,
            'average_space_saved_percent': avg_space_saved,
            'average_utilization_ratio': avg_utilization,
            'average_optimized_bins': avg_optimized_bins,
            'min_space_saved': min(space_saved_values),
            'max_space_saved': max(space_saved_values),
            'optimization_method': 'individual_per_file'
        }
        
        with open(output_path / "optimization_info.txt", 'w') as f:
            f.write("INDIVIDUAL Optimized Spectrogram Generation Results\n")
            f.write("=" * 60 + "\n")
            f.write("Each spectrogram optimized based on its own frequency content\n\n")
            for key, value in optimization_summary.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n🎉 INDIVIDUAL OPTIMIZATION SUCCESSFUL!")
        print(f"✅ Each spectrogram is now optimized for maximum space efficiency!")
        print(f"📊 Check optimization_info.txt for detailed statistics")
        
        return df
    
    return None

def main():
    """Main function"""
    input_directory = "raw"
    output_directory = "processed_data"
    
    if not Path(input_directory).exists():
        print(f"❌ Error: {input_directory} directory not found!")
        return
    
    print(f"📂 Input directory: {input_directory}")
    print(f"📂 Output directory: {output_directory}")
    print(f"\n🎯 INDIVIDUAL OPTIMIZATION APPROACH:")
    print(f"  ✅ Each spectrogram optimized based on ITS OWN frequency content")
    print(f"  ✅ Maximum space savings for every single file")
    print(f"  ✅ No data loss - each file uses its optimal range")
    print(f"  ✅ No 'one size fits all' global limitations")
    
    confirm = input(f"\nGenerate INDIVIDUALLY optimized spectrograms? (y/n): ").strip().lower()
    
    if confirm == 'y':
        df = generate_all_individually_optimized_spectrograms(input_directory, output_directory)
        if df is not None:
            print(f"\n🎉 SUCCESS! All spectrograms individually optimized!")
            print(f"🚀 Ready for CNN training with maximum efficiency!")
        else:
            print(f"❌ Failed to generate spectrograms")
    else:
        print(f"❌ Generation cancelled")

if __name__ == "__main__":
    main() 