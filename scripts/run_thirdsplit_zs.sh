#!/bin/bash
# STEP 1 of 2 for the third-split confirmation arm.  Zero-shot only (condition A), no training.
#
# WHY A THIRD REGION.  The paper's primary value gate is scored on the SELECTION split, which is
# disjoint from the windows the B-D outcome is scored on.  One residual objection survives that:
# the selection split also drove early stopping and checkpoint choice, so the outcome depends on it
# indirectly.  This arm scores the gate on the last 20% of the TRAIN region instead -- windows that
# fitted no baseline (the ladder is fitted on the head), selected no checkpoint, and carried no
# outcome.  Fine-tuned checkpoints are not retained, so the outcome cannot be re-scored on a new
# region; the third split therefore has to be carved on the gate side, which is what this does.
#
# The tail is carved by explicit INDEX, not by which train windows a run happened to skip:
# finetune_forecasting.py subsamples train windows with np.random.choice, so the unused ones are
# seed-dependent and are not a clean held-back region.
#
# No pipes on the run line.  `... | grep | tail` reports the exit status of the right-hand side, so
# a crashed cell logs as a success -- a 16h window has already been lost to exactly that.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1     # Moirai's sampling path hits aten::poisson
OUT=results/v49_thirdsplit
LOG=logs/thirdsplit_zs.log
FRAC=0.2
mkdir -p "$OUT" logs
: > "$LOG"

# Preflight: the loader reads the benchmark CSVs, so verify them before spending the compute.
if ! .venv12/bin/python scripts/data_manifest.py --check >> "$LOG" 2>&1; then
  echo "PREFLIGHT FAILED: benchmark CSVs do not match the manifest" >> "$LOG"
  exit 1
fi

fails=0
run() {  # size dataset horizon
  local S=$1 DS=$2 H=$3
  local KEY="${S}_${DS}_h${H}"
  local D="$OUT/$KEY"
  if [ -f "$D/condition_A_traintail_h${H}_s42.json" ]; then
    echo "skip $KEY (already recorded)" >> "$LOG"
    return
  fi
  echo "=== $KEY  $(date +%H:%M:%S)" >> "$LOG"
  $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/${DS}.csv" \
    --model-size "$S" --horizon "$H" --condition A --zs-windows traintail \
    --traintail-frac "$FRAC" --seed 42 --device mps --results-dir "$D" >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAILED $KEY rc=$rc" >> "$LOG"
    fails=$((fails + 1))
  fi
}

# The 21 Moirai cells of gate_all_cells.MOIRAI_CELLS, in that order.  The non-Moirai arms are not
# covered here: their numerators come from separate evaluators, and the confirmation arm is reported
# as Moirai-only rather than silently presented as matrix-wide.
run small ETTh1        96
run small ETTh1        192
run small ETTh2        96
run small ETTh2        192
run small ETTm2        96
run small ETTm2        192
run small Weather      96
run small Weather      192
run base  ETTh1        96
run base  ETTh1        192
run base  ETTh2        96
run base  ETTh2        192
run large ETTh2        96
run base  Weather      96
run base  Weather      192
run base  ETTm2        96
run base  ETTm2        192
run small Electricity7 96
run small Electricity7 192
run large ETTh1        96
run large Weather      96

n=$(ls "$OUT"/*/condition_A_traintail_*.json 2>/dev/null | wc -l | tr -d ' ')
echo "=== EXIT $fails ===" >> "$LOG"
echo "BANNER cells_recorded=$n expected=21 failures=$fails frac=$FRAC" >> "$LOG"
exit $fails
