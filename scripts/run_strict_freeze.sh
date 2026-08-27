#!/bin/bash
# The strict-freeze control (condition H) on the six degradation cells, plus ILI.
#
# WHY: condition D freezes the encoder's weights but leaves in_proj and mask_encoding trainable, so
# what the encoder receives keeps changing and its output is not a fixed function of the input --
# on ILI that shows up as D reaching CKA 0.76-0.90 rather than 1.0. A reader can therefore object
# that "freezing wins" in the degradation cells is partly the input projection re-fitting rather
# than the encoder being held still. Condition H removes the objection: everything upstream of the
# encoder is frozen, CKA is 1.0 by construction, and only param_proj (2.96M params for small)
# trains.
#
# READING RULE, fixed before the runs: the claim is that the SIGN and READING of B-D survive strict
# freezing. B-H is reported beside B-D for every cell. Agreement => the degradation result is
# robust. Divergence on any cell => that cell's "freezing wins" is partly input re-fitting, and it
# gets said in the body rather than buried.
#
# Seeds 42/123/456 match the existing condition-B runs cell for cell; cell_matrix.py pairs on seed.
# Serialized -- concurrent MPS jobs contend badly. Resumes by output-file existence.
#
# Priority order matters: the window is ~12h and the list is ~12.3h. Cells are ordered so that if
# the window closes, what is lost is the h=192 half of each pair, never a whole cell. Whatever is
# missing at write-up time is stated explicitly, not silently dropped.
set -uo pipefail
cd /Users/mediratta/code/paper_writing/iotsf_demo
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1
OUT=results/v45_strict_freeze

cell() {  # size dataset horizon
  for S in 42 123 456; do
    D_="$OUT/$1_$2_h$3/condition_H"
    [ -f "$D_/condition_H_h$3_s$S.json" ] && continue
    echo "=== $1/$2 h$3 cond H seed $S  $(date +%H:%M:%S)"
    $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/$2.csv" \
      --condition H --model-size "$1" --horizon "$3" --epochs 20 --seed "$S" \
      --max-train-samples 1000 --device mps --results-dir "$D_" 2>&1 \
      | grep -aE 'Zero-shot|forgetting|CKA|Strict freeze|Traceback|Error' | tail -5
  done
}

ili() {  # ILI runs its own script; 10 seeds to match the published condition B/D there
  for S in 42 101 123 202 303 456 777 789 888 999; do
    [ -f "$OUT/ili/condition_H_seed$S.json" ] && continue
    echo "=== ILI cond H seed $S  $(date +%H:%M:%S)"
    $PY -u scripts/finetune_ili.py --condition H --seed "$S" --epochs 20 --device cpu \
      --results-dir "$OUT/ili" 2>&1 \
      | grep -aE 'AGGREGATE|Mean|frozen|Traceback|Error' | tail -4
  done
}

cell small Weather 96      # ~1.8h
cell base  ETTh1   96      # ~2.5h
cell small ETTh1   96      # ~1.0h
ili                        # ~0.5h -- the cell where D drifts most, so most likely to disagree
cell small Weather 192     # ~3.0h
cell base  ETTh1   192     # ~2.5h
cell small ETTh1   192     # ~1.0h
echo "ALL STRICT-FREEZE DONE $(date +%H:%M:%S)"
