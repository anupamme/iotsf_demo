#!/bin/bash
# BATCH 3 of the prospective arm: conditions B and D for the ten cells registered in
# results/v57_prospective3/preregistration_v3.json.  Launch ONLY after that file's commit is in
# `git log` -- the registration script refuses to write once a B/D record exists here, and the
# ordering in the history is the claim the arm rests on.
#
# The cell list, the seeds, the epoch budget, the sample cap and the ORDER below are all copied from
# the registration rather than chosen here: the two gate-PASSING Moirai cells run first because they
# are the only cells where the gate predicts degradation and so the only cells that can produce a
# true positive.  Nothing in this file may be reordered by what the early cells return.
#
# HYPERPARAMETERS ARE THE PUBLISHED ARM'S, NOT NEW ONES.  Moirai: 20 epochs, 1,000 training windows,
# 300 evaluation sequences, no --early-stopping flag, exactly as results/v47_prospective was run.
# TimesFM: 20 epochs, batch 16, 1,000 training windows, exactly as results/v46_timesfm was run, with
# --horizon 48 (the pool screened these five cells at h=48; the script's default is 24).
#
# SEED PAIRING: B and D for the SAME seed run back to back, because cell_matrix.py pairs on seed.
# Truncating this script therefore leaves complete pairs, never half a cell.
#
# MEASURED COST, from the mtime spacing of the runs above (not an a priori guess):
#   large/ETTh2 h192, large/ETTm2 h192   ~12 h each   (large h96 is 102 min B / 88 min D on an
#                                                      ETT-shaped CSV; h192 is ~1.3x)
#   large/ETTm2 h96                      ~9.5 h
#   base/Electricity7 h96, h192          ~5.6 h, ~7 h
#   five TimesFM h=48 cells              ~3.5 h total (v46/ETTh1 h24 ran A+3B+3D in 32 min; the
#                                                      ~55 min/run in run_timesfm_arm.sh's header is
#                                                      an estimate that never matched its own records)
# ~50 h in total, which is the cost_estimate_h the registration recorded.
#
# NO GREP FILTER ON THE CHILD'S OUTPUT AND NO UNCONDITIONAL SUCCESS LINE.  run_prospective_arm.sh
# has both, and the combination once reported "ALL PROSPECTIVE DONE" through a night in which the
# runs had crashed -- `grep` exits 0 on no match and it, not python, set the pipeline's status.  Here
# every run's exit code and the existence of its JSON are logged per run, and the closing line counts
# the records that exist against the number expected.  Poll the log; do not read a final line as
# success.
#
# Launch:  nohup bash scripts/run_prospective3.sh >/dev/null 2>&1 &
#          tail -f results/v57_prospective3/run.log
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1     # Moirai's sampling path hits aten::poisson
export HF_HUB_OFFLINE=1                  # TimesFM and Moirai weights are cached; never fetch mid-run
OUT=results/v57_prospective3
LOG=$OUT/run.log
SEEDS="42 123 456"
EXPECTED=60                              # 10 cells x 3 seeds x {B, D}
mkdir -p "$OUT"

if [ ! -f "$OUT/preregistration_v3.json" ]; then
  echo "no registration at $OUT/preregistration_v3.json -- refusing to run" | tee -a "$LOG"
  exit 1
fi

log() { echo "$*" | tee -a "$LOG"; }

# One fine-tuning run.  Skips if its record already exists, so the script is restartable after a
# crash or a deliberate kill without redoing finished seeds.
one() {  # cell_dir condition horizon seed -- remaining args are the model command
  local CELL=$1 C=$2 H=$3 S=$4; shift 4
  local D_="$OUT/$CELL/condition_$C"
  local J="$D_/condition_${C}_h${H}_s${S}.json"
  if [ -f "$J" ]; then
    log "skip $CELL $C s$S (record present)"
    return
  fi
  log "=== $CELL cond $C seed $S  start $(date '+%m-%d %H:%M:%S')"
  "$@" --condition "$C" --seed "$S" --results-dir "$D_" >>"$LOG" 2>&1
  local rc=$?
  log "=== $CELL cond $C seed $S  exit $rc  $(date '+%m-%d %H:%M:%S')  json=$( [ -f "$J" ] && echo yes || echo NO )"
}

moirai() {  # size dataset horizon
  local SZ=$1 DS=$2 H=$3
  for S in $SEEDS; do
    for C in B D; do
      one "${SZ}_${DS}_h${H}" "$C" "$H" "$S" \
        $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/${DS}.csv" \
        --model-size "$SZ" --horizon "$H" \
        --epochs 20 --max-train-samples 1000 --max-eval-sequences 300 --device mps
    done
  done
}

timesfm() {  # dataset
  local DS=$1 H=48
  local KEY="timesfm_$(echo "$DS" | tr '[:upper:]' '[:lower:]')_h$H"
  for S in $SEEDS; do
    for C in B D; do
      one "$KEY" "$C" "$H" "$S" \
        $PY -u scripts/finetune_timesfm.py --dataset "$DS" --horizon "$H" \
        --epochs 20 --batch-size 16 --max-train-samples 1000 --device mps
    done
  done
}

log "### batch 3 start $(date '+%m-%d %H:%M:%S')  $EXPECTED runs expected"

# Registered run order.  Do not reorder.
moirai  large ETTh2        192      # gate +0.924  PASS
moirai  large ETTm2        192      # gate +0.241  PASS
timesfm ETTh1                       # gate -0.050
timesfm ETTh2                       # gate -0.083
timesfm ETTm2                       # gate -0.011
timesfm Weather                     # gate -0.082
timesfm Electricity                 # gate +0.041
moirai  base  Electricity7 96       # gate -0.592
moirai  base  Electricity7 192      # gate -0.512
moirai  large ETTm2        96       # gate -0.278

n=$(ls -1 "$OUT"/*/condition_[BD]/condition_[BD]_h*_s*.json 2>/dev/null | wc -l | tr -d ' ')
log "### batch 3 stopped $(date '+%m-%d %H:%M:%S'): $n of $EXPECTED records exist"
