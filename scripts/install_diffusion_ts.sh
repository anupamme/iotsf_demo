#!/bin/bash
# Installation script for Diffusion-TS model
# This script clones and installs Diffusion-TS from GitHub

set -e  # Exit on error

echo "=================================="
echo "Diffusion-TS Installation Script"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Diffusion-TS is already installed
if python3 -c "import diffusion_ts" 2>/dev/null; then
    echo "✅ Diffusion-TS is already installed!"
    echo ""
    echo "To reinstall, first uninstall:"
    echo "  pip uninstall diffusion-ts"
    echo "Then run this script again."
    exit 0
fi

echo ""
echo "📦 Diffusion-TS not found. Installing..."
echo ""

# Create temp directory
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Clone repository
echo ""
echo "Cloning Diffusion-TS repository..."
cd "$TEMP_DIR"
git clone https://github.com/Y-debug-sys/Diffusion-TS.git

# Install package
echo ""
echo "Installing Diffusion-TS..."
cd Diffusion-TS
pip install -e .

# Verify installation
echo ""
echo "Verifying installation..."
if python3 -c "import diffusion_ts; print(f'✅ Diffusion-TS installed successfully!')" 2>/dev/null; then
    echo ""
    echo "=================================="
    echo "✅ Installation Complete!"
    echo "=================================="
    echo ""
    echo "You can now use the real Diffusion-TS model."
    echo "Test with:"
    echo "  python3 -c 'from src.models import IoTDiffusionGenerator; g = IoTDiffusionGenerator(); g.initialize(); print(\"Works!\")'"
else
    echo ""
    echo "❌ Installation verification failed."
    echo "Please check the error messages above."
    exit 1
fi

# Cleanup
echo ""
echo "Cleaning up temporary files..."
cd ~
rm -rf "$TEMP_DIR"

echo ""
echo "Done!"
