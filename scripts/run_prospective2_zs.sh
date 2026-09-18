#!/bin/bash
# BATCH 2, STEP 1 of 3.  Condition A (zero-shot only, no training) for eight NEW cells.
# Feeds the VALIDATION-side gate that preregister_prospective2.py freezes.
#
# Cells were chosen before any of them was run, on one rule: never scored in any
# earlier arm.  Two add a dataset the study has not used at all (Traffic), four
# add Moirai-Large on datasets where only Small/Base have been measured, and two
# add Base/Electricity7.  No cell was selected by looking at an outcome.
set -uo pipefail
cd /Users/mediratta/code/paper_writing/iotsf_demo
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1     # Moirai's sampling path hits aten::poisson
OUT=results/v48_prospective2

run() {  # size dataset horizon
  local S=$1 DS=$2 H=$3
  local D="$OUT/${S}_${DS}_h${H}/condition_A"
  [ -f "$D/condition_A_h${H}_s42.json" ] && { echo "skip ${S}_${DS}_h${H}"; return; }
  echo "=== A ${S}_${DS}_h${H}  $(date +%H:%M:%S)"
  $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/${DS}.csv" \
    --model-size "$S" --horizon "$H" --condition A --seed 42 --device mps \
    --results-dir "$D" 2>&1 | grep -aE "Zero-shot|Traceback|Error" | tail -3
}
run large ETTm2        96
run large ETTm2        192
run large Electricity7 96
run large ETTh1        192
run base  Electricity7 96
run base  Electricity7 192
run small Traffic      96
run base  Traffic      96
echo "ALL ZS DONE $(date +%H:%M:%S)"
