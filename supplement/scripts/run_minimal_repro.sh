#!/usr/bin/env bash
# Minimal reproducibility pipeline.
#
# Usage:
#   bash scripts/run_minimal_repro.sh configs/etth2_small_n500.yaml
#   bash scripts/run_minimal_repro.sh configs/etth2_small_n10k.yaml
#
# Estimated runtimes (A10G GPU):
#   n=500, 4 conditions x 1 seed:  ~15 minutes
#   n=10k, 4 conditions x 10 seeds: ~4 hours
#
# NOTE: Exact deterministic replication verified on A10G (Ampere).
# Results on other GPU architectures may differ slightly due to
# non-associative floating-point operations in cuDNN/cuBLAS.

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml>}"
DEVICE="${2:-auto}"
SEEDS_FILE="${3:-}"
RESULTS_DIR="runs/$(basename "$CONFIG" .yaml)"

echo "============================================================"
echo "  Reproducibility Pipeline"
echo "  Config: $CONFIG"
echo "  Device: $DEVICE"
echo "  Results: $RESULTS_DIR"
echo "============================================================"

# Step 0: Ensure data is available
echo -e "\n[Step 0] Preparing data..."
python scripts/00_download_or_prepare_data.py

# Step 1: Value gate
echo -e "\n[Step 1] Computing value gate..."
python scripts/01_value_gate.py --config "$CONFIG" --device "$DEVICE"

# Step 2: Fine-tune with full diagnostic protocol
echo -e "\n[Step 2] Fine-tuning with diagnostics..."
SEEDS_ARG=""
if [ -n "$SEEDS_FILE" ]; then
    SEEDS_ARG="--seeds-file $SEEDS_FILE"
fi
python scripts/02_finetune_moirai.py \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --results-dir "$RESULTS_DIR" \
    $SEEDS_ARG

# Step 3: CKA summary
echo -e "\n[Step 3] CKA summary..."
python scripts/03_compute_cka.py --results-dir "$RESULTS_DIR"

# Step 4: Probe summary
echo -e "\n[Step 4] Probe summary..."
python scripts/04_run_probes.py --results-dir "$RESULTS_DIR"

# Step 5: Forgetting aggregation
echo -e "\n[Step 5] Forgetting statistics..."
python scripts/05_compute_forgetting.py --results-dir "$RESULTS_DIR"

# Step 6: Generate tables
echo -e "\n[Step 6] Generating tables..."
python scripts/06_make_tables.py --input "$RESULTS_DIR" --output expected_outputs/

echo -e "\n============================================================"
echo "  Pipeline complete. Check expected_outputs/ for CSVs."
echo "============================================================"
