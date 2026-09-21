#!/usr/bin/env bash
# Per-layer CKA for the re-run N=6 (condition B) arm of tab:layerunfreeze.
#
# WHY: the per-layer CKA table in app:layerunfreeze had no run record. The script
# that produced it printed to stdout only, and its docstring restated the paper's
# own numbers as the thing to reconcile against -- circular provenance. It also
# read the v19 CUDA checkpoints, whose global CKA (0.518) contradicts the N=6 row
# printed directly above the per-layer table.
#
# This recomputes the breakdown from the v49 N=6 checkpoints, so the per-layer
# values and the global CKA in the row above come from the same 10 runs, and
# writes JSON so the numbers can be re-checked.
#
# Inference only, no training: ~200 val windows through 10 checkpoints.

set -u
# Repo root derived from this script's own location, not hard-coded: the absolute path
# that used to be here named a home directory (a deanonymisation vector in a
# supplementary bundle) and broke outright when the repository moved -- two of these
# runners still pointed at a location that has not existed since July.
cd "$(dirname "$0")/.." || exit 1

export PYTORCH_ENABLE_MPS_FALLBACK=1
export CKA_DEV=mps
export CKA_CKPTS='results/v49_layerunfreeze_10seed/N6_seed*/best_encoder.pt'
export CKA_OUT=results/v49_layerunfreeze_10seed/per_layer_cka_N6.json

echo "start $(date '+%Y-%m-%d %H:%M:%S')"
.venv12/bin/python scripts/layerwise_cka_from_ckpt.py
echo "exit=$? finish $(date '+%Y-%m-%d %H:%M:%S')"
