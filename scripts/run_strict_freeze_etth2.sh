#!/bin/bash
# The strict-freeze control (condition H) on the FIVE GATE-PASSING cells, all ETTh2.
#
# WHY THIS RUN EXISTS: results/v45_strict_freeze covers ETTh1, Weather and ILI -- none of which
# clears the value gate. So the published strict-freeze control speaks only to cells where there is
# no demonstrated pre-trained advantage in the first place. The five cells the paper's conclusions
# actually rest on (Moirai-S/B/L on ETTh2) had NO condition-H arm at all. This run closes that gap.
#
# Condition D freezes the encoder's weights but leaves in_proj and mask_encoding trainable, so what
# the encoder receives keeps changing and its output is not a fixed function of the input -- D lands
# at CKA 0.76-0.99 rather than 1.0. A reader can therefore object that "freezing wins" on these
# cells is partly the input projection re-fitting rather than the encoder being held still.
# Condition H removes the objection: everything upstream of the encoder is frozen, CKA is 1.0 by
# construction, and only param_proj trains.
#
# READING RULE, fixed before the runs: the claim is that the SIGN and READING of B-D survive strict
# freezing. B-H is reported beside B-D for every cell. Agreement => the five survivors' readings are
# not artefacts of input-projection re-fitting. Divergence on any cell => that cell's "freezing
# wins" is partly input re-fitting, and it gets said in the body rather than buried.
#
# PROTOCOL MATCHING -- the reason n is a per-cell argument and not a constant.
# run_strict_freeze.sh hardcodes --max-train-samples 1000, which was correct there because every
# v45 cell's B/D arm ran at n=1000. It is NOT correct here: these five cells split 500/1000, so a
# constant 1000 would have silently produced an unmatched comparison on three of the five. Values
# below were read off the stored B/D runs, per cell:
#
#   cell                  B/D dir                       n     epochs  early_stop  B/D seeds
#   small ETTh2 h96       forecasting_finetune_20ep      500   20      absent      10
#   small ETTh2 h192      forecasting_finetune_20ep      500   20      absent      10
#   base  ETTh2 h96       v5_etth2_base/h96              1000  20      absent      3 (42/123/456)
#   base  ETTh2 h192      v5_etth2_base/h192             1000  20      absent      3 (42/123/456)
#   large ETTh2 h96       v8_etth2_large                 500   20      absent      3 (42/123/456)
#
# "early_stop absent" == no early_stopping block in those JSONs, i.e. a fixed 20-epoch budget, which
# is what condition H does too (H writes early_stopping={'enabled': False}). Batch size and learning
# rate are left at the script defaults, as in run_strict_freeze.sh; the stored B/D records do not
# carry either field, so they cannot be verified from the run records and are matched by convention.
# All five B/D arms postdate the 1.0-R -> 1.1-R checkpoint switch (commit f2c4da7, 2026-01-25; runs
# 2026-07-24), so H loads the same weights those runs loaded.
#
# Seeds 42/123/456: cell_matrix.strict_freeze_cells() pairs on seed and intersects B & D & H, so
# three H seeds against ten B/D seeds is fine and is the already-documented reason strictfreeze.tex
# and heldout_all.tex print different B-D for one cell.
#
# Serialized -- concurrent MPS jobs contend badly. Resumes by output-file existence.
#
# ORDER: cheapest first, Moirai-Large last. If the window closes, what is lost is whole cells at the
# tail, never half of every cell. Whatever is missing at write-up time is stated explicitly.
set -uo pipefail
# Repo root derived from this script's own location, not hard-coded: the absolute path
# that used to be here named a home directory (a deanonymisation vector in a
# supplementary bundle) and broke outright when the repository moved -- two of these
# runners still pointed at a location that has not existed since July.
cd "$(dirname "$0")/.." || exit 1
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1
OUT=results/v51_strictfreeze_etth2

cell() {  # size dataset horizon n_train
  for S in 42 123 456; do
    D_="$OUT/$1_$2_h$3/condition_H"
    [ -f "$D_/condition_H_h$3_s$S.json" ] && continue
    echo "=== $1/$2 h$3 n=$4 cond H seed $S  $(date +%H:%M:%S)"
    $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/$2.csv" \
      --condition H --model-size "$1" --horizon "$3" --epochs 20 --seed "$S" \
      --max-train-samples "$4" --device mps --results-dir "$D_" 2>&1 \
      | grep -aE 'Zero-shot|forgetting|CKA|Strict freeze|Traceback|Error' | tail -5
  done
}

cell small ETTh2  96  500    # ~0.5h  (n=500, half the v45 budget)
cell small ETTh2 192  500    # ~1.0h
cell base  ETTh2  96 1000    # ~2.5h
cell base  ETTh2 192 1000    # ~2.5h
cell large ETTh2  96  500    # ~3-4h  <- first thing lost if the window closes
echo "ALL ETTh2 STRICT-FREEZE DONE $(date +%H:%M:%S)"
