#!/bin/bash
# Setup and run moderate-gate-band screening on AWS A10G.
# Usage: scp this + moderate_gate_screen.py to instance, then run.
set -e

echo "=== Setting up environment ==="

# Create venv if not exists
if [ ! -d ~/mvenv ]; then
    python3 -m venv ~/mvenv
fi
source ~/mvenv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn chronos-forecasting transformers accelerate

echo "=== Verifying GPU ==="
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "=== PHASE 1: Gate Screen ==="
python3 ~/src/moderate_gate_screen.py --phase screen --device cuda --results-dir ~/results/moderate_gate

echo ""
echo "=== PHASE 2: Fine-tune moderate-gate cells ==="
python3 ~/src/moderate_gate_screen.py --phase finetune-all --device cuda \
    --results-dir ~/results/moderate_gate \
    --epochs 25 --max-train-samples 2000 --n-seeds 5 --patience 7

echo ""
echo "=== DONE ==="
echo "Results in ~/results/moderate_gate/"
ls -la ~/results/moderate_gate/
