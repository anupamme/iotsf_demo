#!/usr/bin/env bash
# Setup script for g5.2xlarge (Experiment 2: Chronos + MOMENT)
# Additional deps beyond the base setup

set -eu

echo "=== GPU Server Setup (Exp 2: Chronos + MOMENT) ==="
source ~/venv/bin/activate

# Install Chronos
pip install -q "chronos-forecasting[training]>=1.2"

# Install MOMENT
pip install -q "momentfm>=0.1"

echo "Verifying installations..."
python3 -c "from chronos import ChronosPipeline; print('Chronos OK')"
python3 -c "from momentfm import MOMENTPipeline; print('MOMENT OK')"

echo ""
echo "=== Exp 2 Setup Complete ==="
