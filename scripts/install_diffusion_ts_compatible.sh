#!/bin/bash
# Compatible Diffusion-TS Installation Script
# Installs Diffusion-TS with dependencies compatible with iotsf_demo

set -e

echo "=========================================="
echo "Compatible Diffusion-TS Installation"
echo "=========================================="
echo ""

# Check if already installed
if python3 -c "import sys; sys.path.insert(0, 'lib/Diffusion-TS'); import Models.interpretable_diffusion.gaussian_diffusion" 2>/dev/null; then
    echo "✅ Diffusion-TS is already installed!"
    exit 0
fi

echo "📦 Installing Diffusion-TS with compatible dependencies..."
echo ""

# Create lib directory if it doesn't exist
mkdir -p lib

# Clone or update Diffusion-TS
if [ -d "lib/Diffusion-TS" ]; then
    echo "Updating existing Diffusion-TS..."
    cd lib/Diffusion-TS
    git pull
    cd ../..
else
    echo "Cloning Diffusion-TS repository..."
    cd lib
    git clone https://github.com/Y-debug-sys/Diffusion-TS.git
    cd ..
fi

echo ""
echo "Installing minimal required dependencies..."

# Install only the core dependencies with compatible versions
# Skip the full requirements.txt and install manually

pip install --upgrade \
    "einops>=0.6.0" \
    "ema-pytorch>=0.2.0" \
    "tqdm>=4.64.0" \
    "pyyaml>=6.0"

echo ""
echo "✅ Core dependencies installed!"
echo ""
echo "Testing Diffusion-TS import..."

# Test import
python3 << 'EOF'
import sys
from pathlib import Path

# Add Diffusion-TS to path
diffusion_ts_path = Path("lib/Diffusion-TS")
if diffusion_ts_path.exists():
    sys.path.insert(0, str(diffusion_ts_path))

try:
    from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
    from Models.interpretable_diffusion.transformer import Transformer
    print("✅ Diffusion-TS core modules imported successfully!")
    print("")
    print("The following classes are available:")
    print("  - Models.interpretable_diffusion.gaussian_diffusion.Diffusion_TS")
    print("  - Models.interpretable_diffusion.transformer.Transformer")
    print("  - Models.interpretable_diffusion.model_utils")
except Exception as e:
    print(f"❌ Import failed: {e}")
    print("")
    print("You may need to install additional dependencies.")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Installation Complete!"
    echo "=========================================="
    echo ""
    echo "Diffusion-TS is installed in: lib/Diffusion-TS"
    echo ""
    echo "Note: This is a lightweight installation that avoids"
    echo "dependency conflicts. The core diffusion model is available."
    echo ""
    echo "To use in your code, add to sys.path:"
    echo '  sys.path.insert(0, "lib/Diffusion-TS")'
else
    echo ""
    echo "❌ Installation verification failed."
    exit 1
fi
