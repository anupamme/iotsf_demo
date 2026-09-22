#!/usr/bin/env bash
# The six registered Chronos/M4-Monthly runs: {B,D} x seeds 42,43,44, n=500.
# Sequential on purpose -- one MPS device, and two concurrent Chronos fine-tunes on it are slower
# than two in series as well as making per-epoch timings uninterpretable.
# Launch detached:  nohup bash scripts/run_chronos_m4.sh > logs/chronos_m4.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
for seed in 42 43 44; do
  for cond in B D; do
    echo "=== condition $cond seed $seed  $(date -u +%H:%M:%S)"
    # -u, not a plain invocation: with stdout redirected to a file Python block-buffers,
    # so the log stays empty for the whole run and polling it cannot distinguish 'slow'
    # from 'hung'. An unpollable log is how a 16-hour window was lost here before.
    .venv-probe/bin/python -u scripts/chronos_m4_corrected.py \
      --condition "$cond" --seed "$seed" --device mps || echo "FAILED: $cond/$seed rc=$?"
  done
done
# No "ALL DONE" line. A runner's own success line has masked a crash here before; the completion test
# is six json files under results/chronos_m4/cond_{B,D}/seed*/, counted from outside.
echo "=== loop exited $(date -u +%H:%M:%S); expected 6 records, found: $(ls results/chronos_m4/cond_*/seed*/*.json 2>/dev/null | wc -l)"
