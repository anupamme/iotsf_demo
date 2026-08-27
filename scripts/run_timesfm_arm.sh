#!/bin/bash
# The third-backbone intervention: TimesFM 2.5 (200M) conditions A / B / D at h=24.
#
# WHY: 14 of the paper's 19 cells are Moirai and 5 are Chronos, and the Chronos arm needs an
# attached MSE head for a matched B/D comparison. This arm adds a backbone trained through its OWN
# output head and scored through its OWN inference path, so the generality claim does not rest on
# Moirai and is not open to the "that is not normal TSFM fine-tuning" objection.
#
# LICENSED BY OUR OWN SCREEN, checked BEFORE these runs: at h=24 all five datasets gate-PASS
# (results/gate_test_side.json, timesfm_*: ettm2 +0.703, weather +0.506, etth2 +0.472,
# etth1 +0.401, electricity +0.214). At h=96 TimesFM gate-FAILS (-0.139/-0.138), which is why the
# horizon here is 24 and not 96.
#
# PRIORITY ORDER, fixed before any run and NOT by outcome: datasets are ordered by correspondence
# to the six existing degradation cells, so that whatever the window buys is maximally comparable
# to what the paper already reports. ETTh1 first (4 of the 6 degradation cells are ETTh1), then
# Weather (the other 2), then ETTm2 (widest gate margin), then ETTh2. Whatever is missing at
# write-up time is stated in the paper, not silently dropped.
#
# SEED PAIRING: within a cell the loop runs B and D for the SAME seed back to back, because
# cell_matrix.py pairs B and D on seed -- an unpaired B contributes nothing. Truncating this script
# at any point therefore leaves complete pairs, never half a cell.
#
# Resumes by output-file existence. Serialized: concurrent MPS jobs contend badly.
# Timings from the smoke test (64 windows, batch 8): B ~1.2 s/step, D ~0.35 s/step. At 1000 windows
# and batch 16 that is ~55 min per B run and ~20 min per D run, so ~3.7 h per cell.
set -uo pipefail
cd /Users/mediratta/code/paper_writing/iotsf_demo
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1
OUT=results/v46_timesfm
SEEDS="42 123 456"

cell() {  # dataset
  local DS=$1
  local BASE="$OUT/${DS}_h24"
  if [ ! -f "$BASE/condition_A/condition_A_h24_s42.json" ]; then
    echo "=== $DS cond A  $(date +%H:%M:%S)"
    $PY -u scripts/finetune_timesfm.py --dataset "$DS" --condition A --seed 42 \
      --device mps --results-dir "$BASE/condition_A" 2>&1 \
      | grep -aE 'ZS |native-path|Traceback|Error' | tail -3
  fi
  for S in $SEEDS; do
    for C in B D; do
      D_="$BASE/condition_$C"
      [ -f "$D_/condition_${C}_h24_s$S.json" ] && continue
      echo "=== $DS cond $C seed $S  $(date +%H:%M:%S)"
      $PY -u scripts/finetune_timesfm.py --dataset "$DS" --condition "$C" --seed "$S" \
        --epochs 20 --batch-size 16 --max-train-samples 1000 --device mps \
        --results-dir "$D_" 2>&1 \
        | grep -aE 'native-path|ZS val|params trainable|restoring|verified|CKA |val  |test |Traceback|Error' \
        | tail -8
    done
  done
  echo "--- $DS done $(date +%H:%M:%S)"
}

cell ETTh1      # ~3.7h -- 4 of the 6 degradation cells are ETTh1
cell Weather    # ~3.7h -- the other 2
cell ETTm2      # ~3.7h -- widest gate margin (+0.703)
cell ETTh2      # ~3.7h
echo "ALL TIMESFM DONE $(date +%H:%M:%S)"
