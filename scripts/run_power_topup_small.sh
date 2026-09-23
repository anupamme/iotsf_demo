#!/usr/bin/env bash
# Phase F2, MOIRAI-SMALL SUBSET ONLY -- the 22 registered runs of small_ETTh2_h192 and
# small_ETTm2_h192, launched CONCURRENTLY with Phase E3 rather than after it.
#
# WHY THIS FILE EXISTS INSTEAD OF A FLAG ON run_power_topup.sh.  That script refuses to start while
# any fine-tune is alive (its lines 35-37: "MPS IS SINGLE-TENANT IN PRACTICE"), and that guard is
# there to stop a later reader from assuming the recorded per-run costs were measured on an idle
# machine. Weakening it in place would erase the guard for every future invocation. This file waives
# it for one named subset, on the user's explicit decision of 2026-09-23, and says so in its own log
# so the waiver travels with the records rather than living in a chat transcript.
#
# WHAT CONCURRENCY DOES AND DOES NOT AFFECT.  It does not touch the numbers: each run takes its seed,
# its hyperparameters and its command verbatim from the registration, so the added seeds stay
# exchangeable with the cells' existing ones. What it does affect is wall-clock timing, so the cost
# comments in run_power_topup.sh no longer describe how these 22 runs were produced -- hence the
# CONCURRENT marker written into every start line below.
#
# WHY MOIRAI-SMALL AND NOT THE OTHER FOUR RUNS.  E3 is fine-tuning Moirai-Large, measured at a 5.58 GB
# phys_footprint on a 24 GB machine that is already using 5.9 GB of swap. Moirai-Small is the only
# part of the top-up small enough to sit beside that. The four base_ETTh2_h96 runs stay serialized
# behind E3, where run_power_topup.sh will pick them up and skip these 22 as "record present".
#
# THE MEMORY VALVE.  Before each run this script requires MIN_AVAIL_GB of free+inactive memory and
# otherwise sleeps, so a tight moment delays THIS chain instead of pushing E3's Large run into an MPS
# allocation failure. E3 is the critical path; this chain is the one that may wait.
#
# NO GREP FILTER ON THE LOG AND NO UNCONDITIONAL SUCCESS LINE, for the reason recorded in
# run_power_topup.sh: a pipeline whose status came from `grep` once printed "ALL PROSPECTIVE DONE"
# through a night of crashed runs. Every run logs its own exit code and whether its record exists,
# and the closing line is a COUNT, which is a fact rather than a claim of success.
#
#   nohup bash scripts/run_power_topup_small.sh >/dev/null 2>&1 &
#   tail -f results/power_topup/run_small.log    # then verify "22 of 22 subset records exist"
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY=.venv-probe/bin/python
PY12=.venv12/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1

REG=results/power_topup/preregistration_power.json
LOG=results/power_topup/run_small.log
MIN_AVAIL_GB=${MIN_AVAIL_GB:-6}
SUBSET_PREFIX=small_
mkdir -p "$(dirname "$LOG")"
log() { echo "$*" | tee -a "$LOG"; }

# --- preconditions, the same ones run_power_topup.sh asserts --------------------------------------
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

# --- available memory, in GB: free + inactive (inactive is reclaimable) ---------------------------
avail_gb() {
  vm_stat | awk '/page size of/{gsub(/[^0-9]/,"",$8); ps=$8}
                 /Pages free/{gsub(/\./,"",$3); f=$3}
                 /Pages inactive/{gsub(/\./,"",$3); i=$3}
                 END{printf "%d", (f+i)*ps/1073741824}'
}

# --- the registered runs of this subset, read out of the registration -----------------------------
SPEC=$("$PY12" - "$REG" "$SUBSET_PREFIX" <<'EOF'
import json, sys
reg, pre = json.load(open(sys.argv[1])), sys.argv[2]
for r in reg["planned_runs"]:
    if not r["ref"].startswith(pre):
        continue
    cmd = r["cmd"]
    assert cmd.startswith("python scripts/finetune_forecasting.py "), cmd
    assert "--model-size small" in cmd, cmd        # the waiver is for Moirai-Small and nothing else
    print("\t".join([r["ref"], str(r["seed"]), r["condition"], r["path"],
                     cmd.replace("python ", "", 1)]))
EOF
) || { log "FATAL: could not read planned_runs out of $REG"; exit 1; }

EXPECTED=$(printf '%s\n' "$SPEC" | grep -c '^')
[ "$EXPECTED" -eq 22 ] || { log "FATAL: subset is $EXPECTED runs, expected 22; check $REG"; exit 1; }

log "### power top-up (Moirai-Small subset) start $(date '+%m-%d %H:%M:%S')  $EXPECTED runs"
log "### CONCURRENT WITH PHASE E3 by the user's decision of 2026-09-23. run_power_topup.sh's"
log "### single-tenant-MPS guard is waived for this subset only; the four base_ETTh2_h96 runs stay"
log "### serialized behind E3. Wall-clock timings here are NOT idle-machine measurements."
log "### memory valve: each run waits for >= ${MIN_AVAIL_GB} GB free+inactive; E3 has right of way."

while IFS=$'\t' read -r REF SEED COND PATH_ ARGS; do
  [ -n "${REF:-}" ] || continue
  if [ -f "$PATH_" ]; then
    log "skip $REF cond $COND s$SEED (record present)"
    continue
  fi
  waited=0
  while [ "$(avail_gb)" -lt "$MIN_AVAIL_GB" ]; do
    [ "$waited" -eq 0 ] && log "wait $REF cond $COND s$SEED: $(avail_gb) GB avail < ${MIN_AVAIL_GB} GB"
    waited=$((waited + 300))
    sleep 300
    if [ "$waited" -ge 7200 ]; then
      log "### stopping: memory has been below ${MIN_AVAIL_GB} GB for 2 h. Re-run this script when"
      log "### E3 is done -- it skips records that exist, so nothing is repeated."
      exit 0
    fi
  done
  [ "$waited" -gt 0 ] && log "resume $REF cond $COND s$SEED after ${waited}s ($(avail_gb) GB avail)"
  mkdir -p "$(dirname "$PATH_")"
  log "=== $REF cond $COND seed $SEED  start $(date '+%m-%d %H:%M:%S')  CONCURRENT  $(avail_gb) GB avail"
  # shellcheck disable=SC2086  -- ARGS is a registered command line, word splitting is intended
  $PY -u $ARGS >>"$LOG" 2>&1
  rc=$?
  log "=== $REF cond $COND seed $SEED  exit $rc  end $(date '+%m-%d %H:%M:%S')  json=$( [ -f "$PATH_" ] && echo yes || echo NO )"
done < <(printf '%s\n' "$SPEC")

n=$("$PY12" - "$REG" "$SUBSET_PREFIX" <<'EOF'
import json, os, sys
reg, pre = json.load(open(sys.argv[1])), sys.argv[2]
print(sum(os.path.exists(r["path"]) for r in reg["planned_runs"] if r["ref"].startswith(pre)))
EOF
)
log "### power top-up (Moirai-Small subset) stopped $(date '+%m-%d %H:%M:%S'): $n of $EXPECTED subset records exist"
