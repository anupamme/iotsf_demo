#!/bin/bash
# The strict-freeze control (condition H) on the FOUR Moirai/ETTm2 value-cells.
#
# WHY THIS RUN EXISTS: condition H now covers ETTh1/Weather/ILI (v45) and the five ETTh2 cells that
# clear the fitted-linear gate (v51). But the gate is only one hypothesis class. Scored against the
# full baseline ladder, 13 of 32 cells clear R2_task = 0.20 under at least one admissible baseline,
# and four of the uncovered ones are Moirai/ETTm2 (Small and Base, h96 and h192). Those four rest on
# condition D alone, and D is not a strict freeze: it leaves in_proj (247,680 params Small /
# 496,128 Base) and mask_encoding (384) trainable, so the encoder's output is not a fixed function of
# its input and D lands at CKA 0.969-0.9999 rather than 1.0. A reader can therefore object that
# "freezing wins" on these cells is partly the input projection re-fitting. Condition H removes the
# objection: everything upstream of the encoder is frozen, only param_proj trains, CKA is exactly
# 1.0000 by construction. This run takes H coverage of the 13 value-cells from 6 to 10.
#
# READING RULE, fixed before the runs: the claim is that the SIGN and READING of Delta_encoder
# (= L_full - L_frozen, what the paper has been calling B-D) survive strict freezing. B-H is reported
# beside B-D for every cell. Agreement => these cells' readings are not artefacts of
# input-projection re-fitting. Divergence on any cell => that cell's "freezing wins" is partly input
# re-fitting, and it gets said in the body rather than buried.
#
# PROTOCOL MATCHING -- read off the stored B/D runs, per cell, before writing this file:
#
#   cell                B/D dir                            n     epochs  early_stop      B/D seeds
#   small ETTm2 h96     v5_ettm2/h96                       1000  20      absent          5 (42/123/456/789/999)
#   small ETTm2 h192    v5_ettm2/h192                      1000  20      absent          5 (42/123/456/789/999)
#   base  ETTm2 h96     v47_prospective/base_ETTm2_h96     1000  20      {enabled:False}  3 (42/123/456)
#   base  ETTm2 h192    v47_prospective/base_ETTm2_h192    1000  20      {enabled:False}  3 (42/123/456)
#
# Unlike the ETTh2 five, all four of these ran at n=1000, so n is constant here -- but it is still
# passed per cell rather than hardcoded, because the next cell added may not be.
# "early_stop absent" == no early_stopping block in those JSONs, i.e. a fixed 20-epoch budget, which
# is what {'enabled': False} records explicitly and what condition H does too. Batch size and
# learning rate are left at the script defaults, as in run_strict_freeze.sh and
# run_strict_freeze_etth2.sh; the stored B/D records do not carry either field, so they cannot be
# verified from the run records and are matched by convention.
#
# CHECKPOINT PARITY, checked rather than assumed: the v5_ettm2 B/D runs are dated 2026-07-24 (the
# same day as the ETTh2 spectrum arms, so after the 1.0-R -> 1.1-R switch in f2c4da7, 2026-01-25),
# and their zero-shot MSE (0.17501-0.17541 at h96) agrees with the independent v41_zs_test
# measurement (0.17494-0.17501) to 0.03%. H therefore loads the weights those runs loaded.
#
# Seeds 42/123/456: cell_matrix.strict_freeze_cells() pairs on seed and intersects B & D & H, so
# three H seeds against five B/D seeds is fine, and B-D is re-averaged on the H seeds only so the
# printed comparison stays like for like.
#
# Serialized -- concurrent MPS jobs contend badly. Resumes by output-file existence.
#
# ORDER: by measured cost, cheapest first, so a closed window loses whole cells at the tail rather
# than halves of every cell. Per-seed estimates scaled from the v51 log (small/ETTh2 h96 n=500 took
# 21 min/seed, h192 27 min; base/ETTh2 n=1000 took 46 min at h96 and 63 min at h192):
# Whatever is missing at write-up time is stated explicitly rather than quietly dropped.
set -uo pipefail
cd /Users/mediratta/code/paper_writing/iotsf_demo
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1
OUT=results/v53_strictfreeze_ettm2

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

cell small ETTm2  96 1000    # ~2.1h  (42 min/seed)
cell base  ETTm2  96 1000    # ~2.3h  (46 min/seed)
cell small ETTm2 192 1000    # ~2.8h  (55 min/seed)
cell base  ETTm2 192 1000    # ~3.2h  (63 min/seed)  <- first thing lost if the window closes
echo "ALL ETTm2 STRICT-FREEZE DONE $(date +%H:%M:%S)"
