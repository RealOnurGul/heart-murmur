#!/usr/bin/env python3
"""
Test script to convert a single audio file to spectrogram
Perfect for testing your setup!
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def test_single_conversion(audio_file_path):
    """Test converting a single audio file to spectrogram"""
    
    audio_path = Path(audio_file_path)
    
    if not audio_path.exists():
        print(f"❌ Error: File {audio_file_path} not found!")
        return False
    
    print(f"🎵 Testing conversion of: {audio_path.name}")
    
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=22050)
        print(f"✅ Audio loaded: {len(y)} samples at {sr} Hz")
        print(f"   Duration: {len(y)/sr:.2f} seconds")
        
        # Create mel-spectrogram with EXACTLY 30 frequency bins - FOCUSED on heart sound range
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=30,          # Exactly 30 mel frequency bins
            n_fft=2048, 
            hop_length=512,
            fmin=20,            # Start from 20 Hz (heart sounds start here)
            fmax=1000           # Limit max frequency to 1kHz (heart sounds end around 200-400Hz)
        )
        
        print(f"🔍 Mel-spectrogram shape: {mel_spec.shape}")
        print(f"   Frequency bins: {mel_spec.shape[0]} (should be 30)")
        print(f"   Time frames: {mel_spec.shape[1]}")
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Create and save spectrogram
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Waveform
        plt.subplot(3, 1, 1)
        plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
        plt.title(f"Waveform: {audio_path.name}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        
        # Plot 2: Spectrogram with proper scaling
        plt.subplot(3, 1, 2)
        librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, 
                                fmin=20, fmax=1000, cmap='viridis')
        plt.title(f"Mel-Spectrogram (30 bins, 20Hz-1kHz)")
        plt.colorbar(format='%+2.0f dB')
        
        # Plot 3: Raw imshow for comparison
        plt.subplot(3, 1, 3)
        plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Raw Spectrogram View (30 bins × {mel_spec_db.shape[1]} frames)")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bins (0-29)")
        plt.colorbar(label='Power (dB)')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = output_dir / f"{audio_path.stem}_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Analysis saved to: {output_file}")
        
        # Also save just the clean spectrogram (50 bins only)
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        clean_output = output_dir / f"{audio_path.stem}_spectrogram.png"
        plt.savefig(clean_output, bbox_inches='tight', pad_inches=0, dpi=150,
                   facecolor='black', edgecolor='none')
        print(f"✅ Clean spectrogram saved to: {clean_output}")
        
        # Verify the dimensions
        print(f"\n📐 Verification:")
        print(f"   Mel-spec shape: {mel_spec_db.shape}")
        print(f"   Expected: (30, X) where X is time frames")
        if mel_spec_db.shape[0] == 30:
            print("   ✅ Correct: 30 frequency bins!")
        else:
            print(f"   ❌ Wrong: {mel_spec_db.shape[0]} frequency bins!")
        
        plt.close('all')  # Close all figures
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Single Audio File Test - 30 Mel Bins")
    print("=" * 40)
    
    # Test with the first file we can find
    raw_dir = Path("raw")
    if not raw_dir.exists():
        print("❌ Raw directory not found!")
        return
    
    # Get first audio file
    audio_files = list(raw_dir.glob("*.wav"))
    if not audio_files:
        print("❌ No audio files found in raw directory!")
        return
    
    # Use the first AR file for testing
    test_file = None
    for f in audio_files:
        if f.name.startswith("AR_"):
            test_file = f
            break
    
    if test_file is None:
        test_file = audio_files[0]  # Use any file if no AR found
    
    print(f"🎯 Using test file: {test_file.name}")
    
    # Parse the filename
    parts = test_file.stem.split('_')
    if len(parts) == 4:
        disease, patient, position, location = parts
        print(f"📋 File info:")
        print(f"   Disease: {disease}")
        print(f"   Patient: {patient}")
        print(f"   Position: {position}")
        print(f"   Location: {location}")
    
    # Run the test
    success = test_single_conversion(test_file)
    
    if success:
        print("\n🎉 Test successful!")
        print("📁 Check the 'test_output' folder to see your spectrograms!")
        print("🖼️  You should see two files:")
        print("   1. *_analysis.png - Shows waveform and spectrogram analysis")
        print("   2. *_spectrogram.png - Clean 30-bin spectrogram for ML")
        print("\n⚠️  PLEASE CHECK: The spectrogram should have NO empty space at the top!")
        print("   The image should be completely filled with frequency data.")
    else:
        print("\n❌ Test failed!")

if __name__ == "__main__":
    main() 