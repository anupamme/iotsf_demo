#!/usr/bin/env bash
# Setup script for g5 GPU instances (A10G)
# Installs Python deps and clones/syncs the repo

set -eu

echo "=== GPU Server Setup ==="
echo "$(date)"

# Create venv if not exists
if [ ! -d ~/venv ]; then
    python3 -m venv ~/venv
    echo "Created venv"
fi
source ~/venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install core deps
pip install -q \
    numpy">=1.26,<2.0" \
    pandas">=2.1" \
    scipy">=1.11,<1.14" \
    scikit-learn">=1.5" \
    torch">=2.4" \
    einops">=0.7,<0.8" \
    transformers">=4.40" \
    loguru">=0.7" \
    tqdm">=4.60" \
    pyyaml">=6.0"

# Install uni2ts (Moirai)
pip install -q "uni2ts>=2.0.0"

echo "Core packages installed."

# Verify CUDA
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Verify Moirai
python3 -c "from uni2ts.model.moirai import MoiraiForecast, MoiraiModule; print('uni2ts/Moirai OK')"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source ~/venv/bin/activate"
