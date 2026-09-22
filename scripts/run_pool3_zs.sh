#!/bin/bash
# BATCH-3 POOL, step 1.  Condition A (zero-shot only, no training) for the three Moirai pool cells
# that have no zero-shot record yet.  The other six come from results/v48_prospective2/, whose
# condition_A runs predate this pool and are disclosed as such in the registration.
#
# These are NOT outcomes and cannot become outcomes: condition A loads the released checkpoint and
# measures it.  Running them before the registration is what lets the registration name a predictor
# value per cell instead of a promise to compute one.
#
# Cell rule, fixed before any of these ran: the complement of gate_all_cells.MOIRAI_CELLS within the
# display grid of tables/r2task.tex.  Traffic is excluded because data/forecasting/Traffic.csv is a
# 14-byte HTTP 404 page with no downloader in the repository.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1     # Moirai's sampling path hits aten::poisson
OUT=results/v56_pool3
LOG=$OUT/zs_run.log
mkdir -p "$OUT"

# No grep filter on the child's output and no unconditional "done" line at the end.  A runner in
# this repository once reported success for a whole night because a grep pipe swallowed the crash
# and `set -o pipefail` was not enough to surface it: the per-cell exit status below is the record,
# and the summary at the end counts the JSON files that actually exist.
run() {  # size dataset horizon
  local S=$1 DS=$2 H=$3
  local D="$OUT/${S}_${DS}_h${H}/condition_A"
  if [ -f "$D/condition_A_h${H}_s42.json" ]; then
    echo "skip ${S}_${DS}_h${H} (already present)" | tee -a "$LOG"
    return
  fi
  echo "=== A ${S}_${DS}_h${H}  start $(date +%H:%M:%S)" | tee -a "$LOG"
  $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/${DS}.csv" \
    --model-size "$S" --horizon "$H" --condition A --seed 42 --device mps \
    --results-dir "$D" >>"$LOG" 2>&1
  local rc=$?
  echo "=== A ${S}_${DS}_h${H}  exit $rc  $(date +%H:%M:%S)  json=$( [ -f "$D/condition_A_h${H}_s42.json" ] && echo yes || echo NO )" | tee -a "$LOG"
}

run large ETTh2        192
run large Weather      192
run large Electricity7 192

n=$(ls -1 "$OUT"/*/condition_A/condition_A_h*_s42.json 2>/dev/null | wc -l | tr -d ' ')
echo "pool-3 zero-shot: $n of 3 cells have a record  $(date +%H:%M:%S)" | tee -a "$LOG"
