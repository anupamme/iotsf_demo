#!/usr/bin/env bash
# The remaining runs before the ICLR 2027 data freeze, in one serial queue. Replaces the E3 chain
# (pid 40235 / run_prospective3.sh 56033) and watcher v3 (pid 92621), both killed when this started.
#
# WHY THE CHAIN HAD TO BE REPLACED, two independent reasons:
#
#  1. run_prospective3.sh iterates its whole registered cell list, skipping only records that already
#     exist. Its next cell after base_Electricity7_h192 is large_ETTm2_h96 -- ~16.5 h of Moirai-Large
#     that does not fit before the 2026-09-26 10:34 deadline and has been deliberately deferred
#     (results/v57_prospective3/deferred_cell_2026-09-24.json, committed BEFORE this script ran).
#     So the remaining runs cannot be driven by that runner at all: invoking it for the repair below
#     would also start the deferred cell. They are driven here instead, with the SAME commands.
#
#  2. watcher v3 gated the Phase F2 top-up on E3 reaching 60 of 60 records. With one cell deferred,
#     60 is unreachable, so v3 would have taken its else-branch and never launched F2 -- which has
#     already been dead at 15 of 26 since 2026-09-23 22:16, when run_power_topup_small.sh's 2-hour
#     memory valve fired. Two arms were heading for partial with nothing reporting it.
#
# WHY SERIAL AND NOT CONCURRENT. The plan assumed pausing another project's pid 82363 would free its
# ~39 GB. It does not: SIGSTOP suspends a process but it keeps its pages, and measurement confirmed
# free+inactive moved only 4.9 -> 5.5 GB with swap used unchanged at 21234 MB. Only exit frees that
# memory, and killing another project's 1d13h run was not authorised. 82363 stays STOPPED anyway --
# it halts the +0.47 GB/h swap growth that threatens a mid-flight Moirai-Large run -- and is resumed
# with SIGCONT at the freeze, not by this script.
#
# WHAT THIS SCRIPT DOES NOT CHANGE. Every command below is byte-for-byte the one run_prospective3.sh
# would have issued (its moirai() at lines 87-98), so the records stay exchangeable with the seeds
# already on disk. Verified against both preregistration_v3.json and the sibling condition_B_h192_s42
# record before this file was written. Execution ORDER differs from the registration's; that changes
# no number, because cells and seeds are independent runs.
#
# NO GREP FILTER AND NO UNCONDITIONAL SUCCESS LINE, for the reason recorded in run_power_topup.sh:
# a pipeline whose status came from `grep` once printed a done line through a night of crashed runs.
# Every run logs its own exit code and whether its record exists; the closing lines are COUNTS.
#
#   nohup bash scripts/run_deadline_tail.sh >/dev/null 2>&1 &
#   tail -f results/v57_prospective3/run.log
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY=.venv-probe/bin/python
PY12=.venv12/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1

OUT=results/v57_prospective3
LOG=$OUT/run.log
CHAIN=results/power_topup/chain.log
DEFERRED=$OUT/deferred_cell_2026-09-24.json
mkdir -p "$(dirname "$CHAIN")"
log()  { echo "$*" | tee -a "$LOG"; }
clog() { echo "$*" >> "$CHAIN"; }

# --- preconditions --------------------------------------------------------------------------------
# The deferral note must be COMMITTED before the queue that embodies the deferral runs, on the same
# principle as the registration guards in run_power_topup.sh: the reasoning has to be in the artifact
# before the fact it explains, not added afterwards.
[ -f "$DEFERRED" ] || { log "FATAL: $DEFERRED is missing; the deferral must be recorded first"; exit 1; }
if ! git diff --quiet HEAD -- "$DEFERRED" 2>/dev/null; then
  log "FATAL: $DEFERRED differs from HEAD. Commit the deferral note before running this queue."
  exit 1
fi
if ! git log -1 --format=%H -- "$DEFERRED" | grep -q .; then
  log "FATAL: $DEFERRED has no commit. Commit the deferral note before running this queue."
  exit 1
fi

# The defaults the already-recorded seeds ran under, asserted rather than assumed.
for pair in "'--lr', type=float, default=1e-4" \
            "'--max-eval-sequences', type=int, default=300"; do
  grep -qF "$pair" scripts/finetune_forecasting.py || {
    log "FATAL: scripts/finetune_forecasting.py no longer has the default [$pair]; the runs below"
    log "       would not be exchangeable with the seeds already on disk."
    exit 1
  }
done

clog "run_deadline_tail.sh start $(date '+%m-%d %H:%M:%S'): replaces the E3 chain (40235/56033) and"
clog "  watcher v3 (92621). Reason: large_ETTm2_h96 is deferred, so run_prospective3.sh can no longer"
clog "  drive the tail (it would start that cell) and v3's n-eq-60 gate on F2 is unreachable."
clog "  Queue: 4x base_Electricity7_h192, then the large_ETTm2_h192 cond B s123 repair, then"
clog "  run_power_topup.sh for F2's 11 remaining runs. SERIAL: pausing pid 82363 did not free memory."

# --- one fine-tuning run, same contract as run_prospective3.sh's one() ----------------------------
one() {  # cell_dir condition horizon seed -- remaining args are the model command
  local CELL=$1 C=$2 H=$3 S=$4; shift 4
  local D_="$OUT/$CELL/condition_$C"
  local J="$D_/condition_${C}_h${H}_s${S}.json"
  if [ -f "$J" ]; then
    log "skip $CELL $C s$S (record present)"
    return
  fi
  mkdir -p "$D_"
  log "=== $CELL cond $C seed $S  start $(date '+%m-%d %H:%M:%S')"
  "$@" --condition "$C" --seed "$S" --results-dir "$D_" >>"$LOG" 2>&1
  local rc=$?
  log "=== $CELL cond $C seed $S  exit $rc  $(date '+%m-%d %H:%M:%S')  json=$( [ -f "$J" ] && echo yes || echo NO )"
}

# Identical to run_prospective3.sh moirai(), for one (size,dataset,horizon,seed,condition).
moirai_one() {  # size dataset horizon seed condition
  local SZ=$1 DS=$2 H=$3 S=$4 C=$5
  one "${SZ}_${DS}_h${H}" "$C" "$H" "$S" \
    $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/${DS}.csv" \
    --model-size "$SZ" --horizon "$H" \
    --epochs 20 --max-train-samples 1000 --max-eval-sequences 300 --device mps
}

# --- wait for the orphaned in-flight run to finish ------------------------------------------------
# Killing the runner shell left its child (base_Electricity7_h192 cond D s42) running; it writes its
# own record. MPS is single-tenant in practice, so nothing starts until it exits.
waited=0
while [ "$(pgrep -f 'finetune_forecasting\.py|finetune_timesfm\.py|finetune_chronos' | wc -l | tr -d ' ')" -gt 0 ]; do
  [ "$waited" -eq 0 ] && log "### waiting for the in-flight run to exit before starting the queue"
  waited=$((waited + 60)); sleep 60
done
[ "$waited" -gt 0 ] && log "### in-flight run exited after ${waited}s; starting queue $(date '+%m-%d %H:%M:%S')"

log "### deadline tail start $(date '+%m-%d %H:%M:%S'). large_ETTm2_h96 is DEFERRED, see $DEFERRED"

# --- leg 1: finish base_Electricity7_h192, registered order (seeds 42 123 456, conditions B then D)
for S in 123 456; do
  for C in B D; do
    moirai_one base Electricity7 192 "$S" "$C"
  done
done

# --- leg 2: the repair. large_ETTm2_h192 cond B s123 was SIGTERMed on 09-23 at 20:08:59 with
# json=NO, leaving the seed unpaired; cell_matrix cannot use an unpaired seed. This is a gate-PASSING
# cell (gate +0.241), so it is worth the 4.45 h that the deferred gate-failing cell freed.
moirai_one large ETTm2 192 123 B

# --- leg 3: Phase F2, the pre-registered power top-up. run_power_topup.sh refuses to start while any
# fine-tune is alive, so it goes last and only once legs 1-2 have exited. It skips the 15 records that
# exist and runs the 7 small_ETTm2_h192 + 4 base_ETTh2_h96 that do not.
while [ "$(pgrep -f 'finetune_forecasting\.py|finetune_timesfm\.py|finetune_chronos' | wc -l | tr -d ' ')" -gt 0 ]; do
  sleep 60
done
log "### legs 1-2 done $(date '+%m-%d %H:%M:%S'); launching run_power_topup.sh for F2"
clog "legs 1-2 done $(date '+%m-%d %H:%M:%S'); launching run_power_topup.sh"
bash scripts/run_power_topup.sh >/dev/null 2>&1
clog "run_power_topup.sh returned $? at $(date '+%m-%d %H:%M:%S')"

# --- closing counts, which are facts rather than claims of success -------------------------------
n3=$(ls -1 "$OUT"/*/condition_[BD]/condition_[BD]_h*_s*.json 2>/dev/null | wc -l | tr -d ' ')
nf=$("$PY12" - results/power_topup/preregistration_power.json <<'EOF'
import json, os, sys
reg = json.load(open(sys.argv[1]))
print(sum(os.path.exists(r["path"]) for r in reg["planned_runs"]))
EOF
)
log "### deadline tail stopped $(date '+%m-%d %H:%M:%S'): E3 has $n3 B/D records (54 expected with"
log "###   large_ETTm2_h96 deferred, 60 registered); F2 has $nf of 26 records."
clog "deadline tail stopped $(date '+%m-%d %H:%M:%S'): E3 $n3 B/D records, F2 $nf of 26"
