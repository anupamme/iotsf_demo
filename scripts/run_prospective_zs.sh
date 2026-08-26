#!/bin/bash
# Condition A (zero-shot only, no training) for the 8 prospective cells.
# Feeds the VALIDATION-side gate that the pre-registration is computed from.
set -uo pipefail
cd /Users/mediratta/code/paper_writing/iotsf_demo
PY=.venv-probe/bin/python
# Moirai's sampling path hits aten::poisson, unimplemented on MPS -- same fallback the
# TimesFM runner sets. Without it every run dies at the zero-shot evaluation.
export PYTORCH_ENABLE_MPS_FALLBACK=1
OUT=results/v47_prospective
run() {  # size dataset horizon
  local S=$1 DS=$2 H=$3
  local D="$OUT/${S}_${DS}_h${H}/condition_A"
  [ -f "$D/condition_A_h${H}_s42.json" ] && { echo "skip ${S}_${DS}_h${H}"; return; }
  echo "=== A ${S}_${DS}_h${H}  $(date +%H:%M:%S)"
  $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/${DS}.csv" \
    --model-size "$S" --horizon "$H" --condition A --seed 42 --device mps \
    --results-dir "$D" 2>&1 | grep -aE "Zero-shot|Traceback|Error" | tail -3
}
run base  Weather      96
run base  Weather      192
run base  ETTm2        96
run base  ETTm2        192
run small Electricity7 96
run small Electricity7 192
run large ETTh1        96
run large Weather      96
echo "ALL ZS DONE $(date +%H:%M:%S)"
