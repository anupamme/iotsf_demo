#!/usr/bin/env bash
# Phase F2 -- run EXACTLY the 26 fine-tunes registered in
# results/power_topup/preregistration_power.json, in the order that file lists them.
#
# THE RUNNER HAS NO OPINIONS.  Every command it executes is read out of the registration's
# `planned_runs` array at run time; nothing about which cell, which seed, which condition or which
# hyperparameter lives in this file.  That is deliberate: a top-up is only not optional stopping if
# the runner cannot be adjusted once a result is visible, and the way to guarantee that is to give it
# nothing to adjust.  If a run needs to change, the registration has to change first, and `git log`
# will show it changed after the results it predicted.
#
# WHERE THE RECORDS LAND, AND WHY IT MATTERS.  Each planned run writes into the cell's ORIGINAL
# results directory -- results/forecasting_finetune_20ep (flat), results/v5_ettm2/h192/condition_{B,D}
# and results/v5_etth2_base/h96/condition_{B,D}.  cell_matrix keys a Moirai cell by
# (results-subdirectory, size, horizon, n_train), so a seed written anywhere else would appear as a
# NEW CELL with n=1 rather than as an extra seed of an existing cell, and the 31-cell matrix would
# silently become a 34-cell matrix.  The registration stores the full path of every run for this
# reason and the paths are used verbatim.
#
# THE HYPERPARAMETERS ARE THE DEFAULTS THAT WERE IN FORCE FOR THE ORIGINAL SEEDS.  The registered
# commands pass --epochs 20, --max-train-samples (500 or 1000 per cell), --model-size, --horizon and
# --device mps, and take the script's defaults for --lr (1e-4), --batch-size (16),
# --max-eval-sequences (300) and --features (M).  Those three defaults were introduced in 5c89c2a
# (2026-04-24) and have not been touched since, so they are the same values the cells' existing seeds
# ran under; this script asserts them before launching anything rather than trusting that sentence.
#
# MEASURED COST (comparable runs, from record mtimes and results/positive_control/grid_status.tsv,
# not a priori estimates):
#   base_ETTh2_h96    4 runs  ~50 min each (v47 base/ETTm2 h96: 49-58 min)          ~3.5 h
#   small_ETTh2_h192  8 runs  ~25 min each (n=500, cheaper than the 1462-1837 s
#                                           n=1000 finetunes in grid_status.tsv)    ~3.5 h
#   small_ETTm2_h192 14 runs  ~30 min each (n=1000)                                 ~7 h
#                                                                            total  ~14 h
#
# MPS IS SINGLE-TENANT IN PRACTICE.  Phase E3's batch-3 runner holds the device for ~50 h; two
# Moirai fine-tunes sharing it do not just run slower, they change each other's timings and risk
# memory pressure. This script refuses to start while any finetune process is alive.
#
# NO GREP FILTER ON THE LOG AND NO UNCONDITIONAL SUCCESS LINE.  scripts/run_prospective_arm.sh piped
# its runs through `grep -aE ... | tail -6` and ended with `echo "ALL PROSPECTIVE DONE"`; a crash
# inside the pipe cost a 16-hour window and still printed the done line. Here every run's exit code
# and the presence of its record are logged individually, and the last line is a COUNT of the records
# that exist -- which is a fact, not a claim of success.
#
#   nohup bash scripts/run_power_topup.sh >/dev/null 2>&1 &
#   tail -f results/power_topup/run.log     # then verify "26 of 26 records exist"
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY=.venv-probe/bin/python
PY12=.venv12/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1

REG=results/power_topup/preregistration_power.json
LOG=results/power_topup/run.log
mkdir -p "$(dirname "$LOG")"
log() { echo "$*" | tee -a "$LOG"; }

# --- preconditions -------------------------------------------------------------------------------
[ -f "$REG" ] || { log "FATAL: $REG is missing; run scripts/preregister_power.py first"; exit 1; }

if ! git diff --quiet HEAD -- "$REG" 2>/dev/null; then
  log "FATAL: $REG differs from HEAD. A registration must be committed BEFORE the runs it predicts."
  exit 1
fi
if ! git log -1 --format=%H -- "$REG" | grep -q .; then
  log "FATAL: $REG has no commit. A registration must be committed BEFORE the runs it predicts."
  exit 1
fi

# The defaults the registered commands rely on, asserted rather than assumed.
for pair in "'--lr', type=float, default=1e-4" \
            "'--batch-size', type=int, default=16" \
            "'--max-eval-sequences', type=int, default=300"; do
  grep -qF "$pair" scripts/finetune_forecasting.py || {
    log "FATAL: scripts/finetune_forecasting.py no longer has the default [$pair] that the cells'"
    log "       existing seeds ran under; the added seeds would not be exchangeable with them."
    exit 1
  }
done

live=$(pgrep -f 'finetune_forecasting\.py|finetune_timesfm\.py|finetune_chronos' | wc -l | tr -d ' ')
if [ "$live" -gt 0 ]; then
  log "FATAL: $live fine-tune process(es) already running (Phase E3 holds MPS). Wait for them."
  exit 1
fi

# --- the registered runs, read out of the registration -------------------------------------------
# One tab-separated line per planned run: path, then the command with `python` replaced by the
# probe venv's interpreter. Nothing is reordered and nothing is filtered.
SPEC=$("$PY12" - "$REG" <<'EOF'
import json, sys
reg = json.load(open(sys.argv[1]))
for r in reg["planned_runs"]:
    cmd = r["cmd"]
    assert cmd.startswith("python scripts/finetune_forecasting.py "), cmd
    print("\t".join([r["ref"], str(r["seed"]), r["condition"], r["path"],
                     cmd.replace("python ", "", 1)]))
EOF
) || { log "FATAL: could not read planned_runs out of $REG"; exit 1; }

EXPECTED=$(printf '%s\n' "$SPEC" | grep -c '^')
log "### power top-up start $(date '+%m-%d %H:%M:%S')  $EXPECTED runs registered in $REG"

while IFS=$'\t' read -r REF SEED COND PATH_ ARGS; do
  [ -n "${REF:-}" ] || continue
  if [ -f "$PATH_" ]; then
    log "skip $REF cond $COND s$SEED (record present)"
    continue
  fi
  mkdir -p "$(dirname "$PATH_")"
  log "=== $REF cond $COND seed $SEED  start $(date '+%m-%d %H:%M:%S')"
  # shellcheck disable=SC2086  -- ARGS is a registered command line, word splitting is intended
  $PY -u $ARGS >>"$LOG" 2>&1
  rc=$?
  log "=== $REF cond $COND seed $SEED  exit $rc  end $(date '+%m-%d %H:%M:%S')  json=$( [ -f "$PATH_" ] && echo yes || echo NO )"
done < <(printf '%s\n' "$SPEC")

n=$("$PY12" - "$REG" <<'EOF'
import json, os, sys
reg = json.load(open(sys.argv[1]))
print(sum(os.path.exists(r["path"]) for r in reg["planned_runs"]))
EOF
)
log "### power top-up stopped $(date '+%m-%d %H:%M:%S'): $n of $EXPECTED records exist"
