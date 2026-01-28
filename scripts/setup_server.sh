#!/bin/bash
# Server Setup Script for IoT Security Demo
# This script installs all dependencies required for Moirai and Diffusion-TS

set -e  # Exit on error

echo "=============================================="
echo "IoT Security Demo - Server Setup"
echo "=============================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Detected Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    echo "WARNING: Python 3.12 is required for uni2ts (Moirai)"
    echo "Current version: $PYTHON_VERSION"
    echo ""
    echo "Please install Python 3.12 and run:"
    echo "  python3.12 -m venv .venv312"
    echo "  source .venv312/bin/activate"
    echo "  ./scripts/setup_server.sh"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv312" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv312
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv312/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
if [ -f "requirements-py312.txt" ]; then
    pip install -r requirements-py312.txt
else
    echo "ERROR: requirements-py312.txt not found!"
    exit 1
fi

# Setup Diffusion-TS
echo ""
echo "Setting up Diffusion-TS..."
mkdir -p lib
if [ ! -d "lib/Diffusion-TS" ]; then
    echo "Cloning Diffusion-TS repository..."
    cd lib
    git clone https://github.com/Y-debug-sys/Diffusion-TS.git
    cd ..
else
    echo "Diffusion-TS already exists, updating..."
    cd lib/Diffusion-TS
    git pull
    cd ../..
fi

# Verify installations
echo ""
echo "=============================================="
echo "Verifying installations..."
echo "=============================================="

# Check uni2ts (Moirai)
echo -n "Checking uni2ts (Moirai)... "
if python3 -c "from uni2ts.model.moirai import MoiraiForecast, MoiraiModule; print('OK')" 2>/dev/null; then
    echo "SUCCESS"
else
    echo "FAILED - Moirai will run in mock mode"
fi

# Check Diffusion-TS
echo -n "Checking Diffusion-TS... "
if python3 -c "import sys; sys.path.insert(0, 'lib/Diffusion-TS'); from Models.interpretable_diffusion.model_utils import *; print('OK')" 2>/dev/null; then
    echo "SUCCESS"
else
    echo "FAILED - Diffusion-TS will run in mock mode"
fi

# Check PyTorch
echo -n "Checking PyTorch... "
if python3 -c "import torch; print(f'OK (CUDA: {torch.cuda.is_available()})')" 2>/dev/null; then
    :
else
    echo "FAILED"
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To run the demo:"
echo "  source .venv312/bin/activate"
echo "  streamlit run app/main.py"
echo ""
