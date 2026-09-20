#!/bin/bash
# Serialise the remaining MPS work behind whatever is already running.
#
# WHY A CHAIN AND NOT THREE LAUNCHES. Concurrent MPS jobs contend badly -- two fine-tuning runs on the
# same GPU do not take 2x as long, they take 3-4x and occasionally OOM the shorter one. So every
# runner in this project is internally serialised, and the only way to queue work behind a run that is
# ALREADY in flight is to wait on its pid. That is all this does.
#
# ORDER, and why. The two-cell check goes first even though it is the shorter job: its output is a
# number that has to go into the Reproducibility Statement, where there is currently a
# "% PENDING:" comment holding the place open (09_statements.tex). One hour unblocks paper text. The
# LoRA arm is twelve hours and unblocks a table, which can be regenerated whenever it lands.
#
# Usage:  nohup bash scripts/run_night2_chain.sh <pid-to-wait-for> > logs/night2_chain.log 2>&1 &
#         (omit the pid to start immediately)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "=== waiting for pid $WAIT_PID to finish   $(date +%H:%M:%S)"
  # `wait` only works on children of this shell, so poll. 60 s is far finer than the ~60 min job.
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "=== pid $WAIT_PID gone   $(date +%H:%M:%S)"
  sleep 30   # let the MPS allocator release before the next job grabs it
fi

echo "=== STEP 7.2: two-cell from-scratch check   $(date +%H:%M:%S)"
bash scripts/rerun_two_cells.sh > logs/rerun_two_cells.log 2>&1
echo "    rerun_two_cells.sh exit $?   (see logs/rerun_two_cells.log)"
tail -12 logs/rerun_two_cells.log | sed 's/^/    /'

echo
echo "=== STEP 0c: LoRA on the eight uncovered Moirai value-cells   $(date +%H:%M:%S)"
bash scripts/run_lora_valuecells.sh > logs/lora_valuecells.log 2>&1
echo "    run_lora_valuecells.sh exit $?   (see logs/lora_valuecells.log)"

echo
echo "=== CHAIN DONE $(date +%H:%M:%S)"
