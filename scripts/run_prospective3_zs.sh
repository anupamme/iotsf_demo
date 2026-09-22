#!/bin/bash
# BATCH 3, step 0: two MORE condition-A seeds for each of the five batch-3 Moirai cells.
#
# WHY, AND WHY IT IS NOT A CHANGE TO THE REGISTRATION.  The registered primary outcome propagates
# "the zero-shot reference's own SEM into both intervals where the arm is unpaired", and the arm IS
# unpaired on every Moirai cell: B and D divide by a per-cell mean over condition-A seeds rather than
# by a seed's own zero-shot.  Each batch-3 Moirai cell currently has exactly ONE condition-A record
# (seed 42, from results/v48_prospective2 or results/v56_pool3), and paired_inference.py says what
# that means in as many words -- with one seed the reference's error is UNESTIMATED rather than zero,
# so nothing is propagated and the B-ZS and D-ZS intervals come out narrower than the rule intends.
# Narrower intervals make degradation EASIER to declare, so leaving this alone would not be the
# conservative choice; it would quietly favour positives on the eight gate-failing cells, which is
# the class the batch exists to measure.
#
# Batch 1 (results/v47_prospective) has the same one-seed reference on all eight of its cells. That
# is reported as a limitation of batch 1 rather than repaired, because repairing it would change a
# denominator after its outcomes were known. Here nothing is known yet: no condition B or D record
# exists for any batch-3 cell, which is what makes this runnable now and not later.
#
# CONDITION A IS NOT AN OUTCOME.  It loads the released checkpoint and measures it; no training
# happens and no fine-tuned quantity is produced. The predictor frozen in preregistration_v3.json is
# also untouched: it is the SELECTION-split ridge gate computed by the pool screen, and these runs
# add held-out zero-shot seeds, which enter no gate.
#
# Flags match the existing seed-42 records exactly -- default eval caps, n_test_eval 300 in every
# condition-A record on disk -- so the mean over seeds is a mean over one measurement repeated at
# three window draws and not a mix of two protocols.
#
# Run BEFORE scripts/run_prospective3.sh; that script refuses to start without these.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1
OUT=results/v57_prospective3
LOG=$OUT/zs_topup.log
SEEDS="123 456"
mkdir -p "$OUT"

log() { echo "$*" | tee -a "$LOG"; }

existing=$(ls -1 "$OUT"/*/condition_[BD]/*.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$existing" != "0" ]; then
  log "refusing to run: $existing condition B/D records already exist under $OUT, so the zero-shot"
  log "denominator is no longer being fixed in advance of the outcomes"
  exit 1
fi

cell() {  # size dataset horizon
  local SZ=$1 DS=$2 H=$3
  local D_="$OUT/${SZ}_${DS}_h${H}/condition_A"
  for S in $SEEDS; do
    local J="$D_/condition_A_h${H}_s${S}.json"
    if [ -f "$J" ]; then log "skip ${SZ}_${DS}_h${H} A s$S (record present)"; continue; fi
    log "=== A ${SZ}_${DS}_h${H} seed $S  start $(date '+%m-%d %H:%M:%S')"
    $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/${DS}.csv" \
      --model-size "$SZ" --horizon "$H" --condition A --seed "$S" --device mps \
      --results-dir "$D_" >>"$LOG" 2>&1
    local rc=$?
    log "=== A ${SZ}_${DS}_h${H} seed $S  exit $rc  $(date '+%m-%d %H:%M:%S')  json=$( [ -f "$J" ] && echo yes || echo NO )"
  done
}

log "### batch-3 zero-shot top-up start $(date '+%m-%d %H:%M:%S')  10 runs expected"

cell large ETTh2        192
cell large ETTm2        192
cell base  Electricity7 96
cell base  Electricity7 192
cell large ETTm2        96

n=$(ls -1 "$OUT"/*/condition_A/condition_A_h*_s*.json 2>/dev/null | wc -l | tr -d ' ')
log "### zero-shot top-up stopped $(date '+%m-%d %H:%M:%S'): $n of 10 records exist under $OUT"
