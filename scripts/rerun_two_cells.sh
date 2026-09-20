#!/bin/bash
# Re-run two cells from raw data and report the deviation from their stored run records.
#
# WHY TWO, AND WHY THESE TWO. scripts/rederive_all.sh proves the tables follow from the run records.
# It cannot prove the run records follow from the code -- for that something has to actually train.
# Retraining the whole 31-cell matrix would take weeks of MPS time and perturb every number in the
# paper, so this is a BOUNDED check: one cell per script path that the paper's headline depends on.
#
#   Moirai-Small / ETTh2 h96, n=500, seed 42, condition B   scripts/finetune_forecasting.py
#     The cheapest cell on the path that produces 22 of the 31 rows, and it is on ETTh2 -- the one
#     series where the gate passes and therefore the only series whose cells carry the degradation
#     test. If any single cell has to stand for the Moirai arm, it is one of these.
#
#   TimesFM / ETTh1 h24, n=1000, seed 42, condition B       scripts/finetune_timesfm.py
#     A different backbone, a different script, and a different scoring path (TimesFM's own output
#     head, no attached regression head), so a defect shared between the two would have to be in the
#     data loader rather than in either trainer. ETTh1 because it is the TimesFM cell the analysis
#     section quotes.
#
# The Chronos arm is deliberately NOT re-run here: its cells are n=8000 over 30 epochs and were run
# on CUDA, so a re-run costs hours and lands on a different backend, which makes it the weakest check
# per hour of the three. That omission is stated rather than hidden.
#
# WHAT A DEVIATION MEANS. Not a failure. MPS has no deterministic mode, the backends have moved since
# the earliest cells, and early stopping turns a low-order-bit difference into a whole-epoch
# difference in checkpoint choice. scripts/compare_rerun.py therefore reports signed relative
# deviations and adjudicates only the SIGN of the forgetting fields, which is what the paper's claims
# rest on. Whatever it prints goes in the paper as printed.
#
# ~1 hour total on this machine. Serialised; concurrent MPS jobs contend badly.
#
# Usage:  nohup bash scripts/rerun_two_cells.sh > logs/rerun_two_cells.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv-probe/bin/python          # torch 2.13 + MPS; the analysis venv has no torch
PYA=.venv12/bin/python             # the comparator needs no torch
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1
OUT=results/v55_rerun_check
mkdir -p "$OUT" logs

# The benchmark CSVs are gitignored, so a fresh clone can produce a plausible-looking run against a
# DIFFERENT variant of ETTh1. Verify identity before spending an hour of compute on it.
if ! $PYA scripts/data_manifest.py --check; then
  echo "STOPPED: the benchmark CSVs do not match results/data_manifest.json. A differently-parsed"
  echo "CSV changes every window, so a re-run against it would compare two different experiments."
  exit 2
fi

echo "=== Moirai-Small / ETTh2 h96 n=500 seed 42 cond B   $(date +%H:%M:%S)"
$PY -u scripts/finetune_forecasting.py --data-path data/forecasting/ETTh2.csv \
  --condition B --model-size small --horizon 96 --epochs 20 --seed 42 \
  --max-train-samples 500 --device mps --results-dir "$OUT/moirai" 2>&1 \
  | grep -aE 'Zero-shot|forgetting|CKA|Traceback|Error' | tail -5

echo "=== TimesFM / ETTh1 h24 n=1000 seed 42 cond B       $(date +%H:%M:%S)"
$PY -u scripts/finetune_timesfm.py --dataset ETTh1 --condition B --seed 42 \
  --epochs 20 --batch-size 16 --max-train-samples 1000 --device mps \
  --results-dir "$OUT/timesfm" 2>&1 \
  | grep -aE 'Zero-shot|forgetting|CKA|Traceback|Error' | tail -5

echo
echo "=== DEVIATION FROM THE STORED RUN RECORDS ==========================================="
rc=0
$PYA scripts/compare_rerun.py --label "Moirai-Small / ETTh2 h96 n=500 s42 cond B" \
  --stored results/v5_etth2_sweep/n500/condition_B_h96_s42.json \
  --fresh  "$OUT/moirai/condition_B_h96_s42.json" || rc=$?
$PYA scripts/compare_rerun.py --label "TimesFM / ETTh1 h24 n=1000 s42 cond B" \
  --stored results/v46_timesfm/ETTh1_h24/condition_B/condition_B_h24_s42.json \
  --fresh  "$OUT/timesfm/condition_B_h24_s42.json" || rc=$?

echo
echo "TWO-CELL CHECK DONE $(date +%H:%M:%S)   (exit $rc; nonzero = a sign flipped or a run is missing)"
exit "$rc"
