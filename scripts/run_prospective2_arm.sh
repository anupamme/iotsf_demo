#!/bin/bash
# The prospective arm: conditions B and D on eight cells that played no part in defining the six
# degradation cells. Launched only AFTER results/v48_prospective2/preregistration_v2.json is committed.
#
# WHY: the six degradation cells are identified using the test split, so their discovery is
# retrospective. Showing that a validation-side gate selects the same six answers a narrower
# question than a reviewer asks. These eight cells have never been scored, so the pre-registered
# predictions in preregistration_v2.json are a genuine out-of-sample test of the criterion.
#
# PRIORITY ORDER, fixed before any run and NOT by outcome:
#   1-2 base/Weather      -- dataset predicts degradation, new model size on the strongest cell
#   3-4 base/ETTm2        -- gate says pass, dataset says no degradation: the rules DISAGREE here,
#                            which makes these two the most informative cells in the set
#   5-6 small/Electricity7 -- third dataset (7 of 370 series; protocol deviation, stated in the paper)
#   7-8 large/{ETTh1,Weather} -- third model size on the two degradation datasets
# Whatever is missing at write-up time is reported, not silently dropped.
#
# SEED PAIRING: B and D for the SAME seed run back to back, because cell_matrix.py pairs on seed;
# truncating this script therefore leaves complete pairs, never half a cell.
set -uo pipefail
# Repo root derived from this script's own location, not hard-coded: the absolute path
# that used to be here named a home directory (a deanonymisation vector in a
# supplementary bundle) and broke outright when the repository moved -- two of these
# runners still pointed at a location that has not existed since July.
cd "$(dirname "$0")/.." || exit 1
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1     # Moirai's sampling path hits aten::poisson
OUT=results/v48_prospective2
SEEDS="42 123 456"

cell() {  # size dataset horizon
  local SZ=$1 DS=$2 H=$3
  local BASE="$OUT/${SZ}_${DS}_h${H}"
  for S in $SEEDS; do
    for C in B D; do
      local D_="$BASE/condition_$C"
      [ -f "$D_/condition_${C}_h${H}_s${S}.json" ] && continue
      echo "=== ${SZ}_${DS}_h${H} cond $C seed $S  $(date +%H:%M:%S)"
      $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/${DS}.csv" \
        --model-size "$SZ" --horizon "$H" --condition "$C" --seed "$S" \
        --epochs 20 --max-train-samples 1000 --max-eval-sequences 300 --device mps \
        --results-dir "$D_" 2>&1 \
        | grep -aE 'Zero-shot|CKA|val MSE|Test|forget|Traceback|Error' | tail -6
    done
  done
  echo "--- ${SZ}_${DS}_h${H} done $(date +%H:%M:%S)"
}

cell large ETTm2        96
cell large ETTm2        192
cell large Electricity7 96
cell large ETTh1        192
cell base  Electricity7 96
cell base  Electricity7 192
cell small Traffic      96
cell base  Traffic      96
echo "BATCH 2 ARM DONE $(date +%H:%M:%S)"
