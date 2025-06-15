#!/bin/bash

# Heart Murmur ML Project Setup Script for M3 Mac
# Run this script to set up your complete environment

echo "🎵 Heart Murmur ML Project Setup"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -d "raw" ]; then
    echo "❌ Error: 'raw' directory not found!"
    echo "Please run this script from the heart-murmur project root directory."
    exit 1
fi

echo "📂 Found raw directory with audio files ✅"

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
if [ -d "heart_murmur_env" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf heart_murmur_env
fi

python3 -m venv heart_murmur_env
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment!"
    exit 1
fi

echo "✅ Virtual environment created!"

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source heart_murmur_env/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install packages
echo ""
echo "📦 Installing required packages (this may take a few minutes)..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install packages!"
    echo "Please check the error messages above and try again."
    exit 1
fi

echo ""
echo "✅ All packages installed successfully!"

# Create project directories
echo ""
echo "📁 Creating project directories..."
mkdir -p processed_data
mkdir -p test_output
mkdir -p models
mkdir -p notebooks

echo "✅ Project structure created!"

# Make Python scripts executable
echo ""
echo "🔧 Making scripts executable..."
chmod +x audio_to_spectrogram.py
chmod +x test_single_audio.py

echo "✅ Scripts are ready to run!"

# Final instructions
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 What you can do now:"
echo ""
echo "1. 🧪 Test single audio conversion:"
echo "   source heart_murmur_env/bin/activate"
echo "   python test_single_audio.py"
echo ""
echo "2. 🚀 Convert all audio to spectrograms:"
echo "   source heart_murmur_env/bin/activate"
echo "   python audio_to_spectrogram.py"
echo ""
echo "3. 📊 Start Jupyter notebook for exploration:"
echo "   source heart_murmur_env/bin/activate"
echo "   jupyter notebook"
echo ""
echo "💡 Remember: Always activate your virtual environment first!"
echo "   source heart_murmur_env/bin/activate"
echo ""
echo "🎵 Your M3 Mac is ready for heart murmur ML! 🍎" 