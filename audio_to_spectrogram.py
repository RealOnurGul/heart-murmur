#!/usr/bin/env python3
"""
Heart Murmur Audio to Spectrogram Converter
Converts audio files to mel-spectrograms for CNN training
Optimized for M3 Mac
"""

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm

def parse_filename(filename):
    """
    Parse the filename to extract labels
    Format: DISEASE_PATIENT_POSITION_LOCATION.wav
    Example: AR_016_sit_Aor.wav
    """
    # Remove .wav extension
    name = filename.replace('.wav', '')
    
    # Split by underscore
    parts = name.split('_')
    
    if len(parts) == 4:
        disease = parts[0]  # N, AR, AS, MR, MS
        patient_id = parts[1]  # 016, 034, etc.
        position = parts[2]  # sit, sup
        location = parts[3]  # Aor, Mit, Pul, Tri
        
        return {
            'disease': disease,
            'patient_id': patient_id,
            'position': position,
            'location': location,
            'filename': filename,
            'label_full': f"{disease}_{position}_{location}"
        }
    else:
        print(f"⚠️  Warning: Unexpected filename format: {filename}")
        return None

def create_spectrogram(audio_file, output_dir, 
                      target_sr=22050, n_mels=30, n_fft=2048, 
                      hop_length=512, figsize=(10, 4)):
    """
    Convert audio file to mel-spectrogram and save as image
    
    Parameters:
    - audio_file: path to audio file
    - output_dir: directory to save spectrogram
    - target_sr: target sample rate
    - n_mels: number of mel bands
    - n_fft: FFT window size
    - hop_length: hop length for FFT
    - figsize: figure size for matplotlib
    """
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=target_sr)
        
        # Create mel-spectrogram focused on heart sound frequencies
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length,
            fmin=20,            # Start from 20 Hz (heart sounds start here)
            fmax=1000           # Limit max frequency to 1kHz (heart sounds end around 200-400Hz)
        )
        
        # Convert to dB scale (logarithmic)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Create the plot
        plt.figure(figsize=figsize)
        plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')  # Remove axes for clean image
        plt.tight_layout(pad=0)
        
        # Save the spectrogram
        output_file = output_dir / f"{Path(audio_file).stem}.png"
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=150, 
                   facecolor='black', edgecolor='none')
        plt.close()
        
        return output_file, True
        
    except Exception as e:
        print(f"❌ Error processing {Path(audio_file).name}: {e}")
        return None, False

def create_sample_spectrograms(input_dir, output_dir, num_samples=5):
    """Create a few sample spectrograms to test the setup"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    samples_dir = output_path / "sample_spectrograms"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a few sample files from each disease type
    audio_files = list(input_path.glob("*.wav"))
    
    # Group by disease type
    disease_files = {}
    for file in audio_files:
        disease = file.name.split('_')[0]
        if disease not in disease_files:
            disease_files[disease] = []
        disease_files[disease].append(file)
    
    print(f"🎵 Found diseases: {list(disease_files.keys())}")
    
    samples_created = []
    
    # Create samples from each disease type
    for disease, files in disease_files.items():
        sample_files = files[:min(num_samples, len(files))]
        
        print(f"\n🔬 Creating {len(sample_files)} samples for {disease}:")
        
        for audio_file in sample_files:
            print(f"  Processing: {audio_file.name}")
            
            # Parse filename
            file_info = parse_filename(audio_file.name)
            if file_info is None:
                continue
            
            # Create spectrogram
            spectrogram_path, success = create_spectrogram(audio_file, samples_dir)
            
            if success and spectrogram_path is not None:
                file_info['spectrogram_path'] = str(spectrogram_path)
                samples_created.append(file_info)
                print(f"  ✅ Saved: {spectrogram_path.name}")
    
    # Save sample metadata
    if samples_created:
        df = pd.DataFrame(samples_created)
        metadata_file = samples_dir / "sample_metadata.csv"
        df.to_csv(metadata_file, index=False)
        print(f"\n📋 Sample metadata saved to: {metadata_file}")
        
        # Print summary
        print(f"\n📊 Sample Summary:")
        print(f"Total samples: {len(samples_created)}")
        print("Disease distribution:")
        for disease, count in df['disease'].value_counts().items():
            print(f"  {disease}: {count} samples")
    
    return samples_created

def process_full_dataset(input_dir, output_dir):
    """Process the complete dataset"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    spectrograms_dir = output_path / "spectrograms"
    spectrograms_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each disease
    diseases = ['N', 'AR', 'AS', 'MR', 'MS']
    for disease in diseases:
        (spectrograms_dir / disease).mkdir(exist_ok=True)
    
    # Get all audio files
    audio_files = list(input_path.glob("*.wav"))
    print(f"🎵 Found {len(audio_files)} audio files")
    
    # Process all files
    metadata = []
    successful = 0
    failed = 0
    
    for audio_file in tqdm(audio_files, desc="Converting audio to spectrograms"):
        # Parse filename
        file_info = parse_filename(audio_file.name)
        if file_info is None:
            failed += 1
            continue
        
        # Create disease-specific subdirectory path
        disease_dir = spectrograms_dir / file_info['disease']
        
        # Create spectrogram
        spectrogram_path, success = create_spectrogram(audio_file, disease_dir)
        
        if success and spectrogram_path is not None:
            file_info['spectrogram_path'] = str(spectrogram_path)
            metadata.append(file_info)
            successful += 1
        else:
            failed += 1
    
    # Save complete metadata
    if metadata:
        df = pd.DataFrame(metadata)
        metadata_file = output_path / "complete_metadata.csv"
        df.to_csv(metadata_file, index=False)
        
        print(f"\n✅ Processing Complete!")
        print(f"📊 Results:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total: {len(audio_files)}")
        
        print(f"\n📁 Files saved to: {output_path}")
        print(f"📋 Metadata: {metadata_file}")
        
        # Print dataset summary
        print(f"\n📈 Dataset Summary:")
        print("Disease distribution:")
        for disease, count in df['disease'].value_counts().items():
            print(f"  {disease}: {count} spectrograms")
        
        print("\nPosition distribution:")
        for pos, count in df['position'].value_counts().items():
            print(f"  {pos}: {count} spectrograms")
        
        print("\nLocation distribution:")
        for loc, count in df['location'].value_counts().items():
            print(f"  {loc}: {count} spectrograms")
    
    return df if metadata else None

def main():
    """Main function"""
    print("🎵 Heart Murmur Audio to Spectrogram Converter")
    print("=" * 50)
    
    # Define paths
    input_directory = "raw"
    output_directory = "processed_data"
    
    # Check if raw directory exists
    if not Path(input_directory).exists():
        print(f"❌ Error: {input_directory} directory not found!")
        return
    
    print(f"📂 Input directory: {input_directory}")
    print(f"📂 Output directory: {output_directory}")
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Create sample spectrograms (5 per disease type)")
    print("2. Process complete dataset")
    print("3. Both")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        print("\n🔬 Creating sample spectrograms...")
        samples = create_sample_spectrograms(input_directory, output_directory)
        if samples:
            print(f"✅ Created {len(samples)} sample spectrograms!")
    
    if choice in ['2', '3']:
        print("\n🚀 Processing complete dataset...")
        confirm = input("This will process all 464 files. Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            df = process_full_dataset(input_directory, output_directory)
            if df is not None:
                print("🎉 Complete dataset processed successfully!")
        else:
            print("❌ Complete dataset processing cancelled.")
    
    print("\n🎉 Done! Your spectrograms are ready for machine learning!")

if __name__ == "__main__":
    main() 