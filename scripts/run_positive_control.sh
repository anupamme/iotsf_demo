#!/bin/bash
# Run the pre-registered positive-control grid, and score every checkpoint on TASK A.
#
# THE GRID IS NOT CHOSEN HERE. It is read from results/positive_control/preregistration.json, which was
# committed before the first run existed. This script's only job is to execute it and to refuse to
# execute anything else -- the learning rates, conditions and seeds below are printed from that file
# rather than typed, so a divergence between what was registered and what ran is visible in the log.
#
# WHY EACH RUN IS SCORED TWICE. The fine-tuning run measures everything on TASK B, because that is what
# it trains on. The question is what happened to TASK A, so after each run both saved checkpoints --
# the early-stopped one (primary, for comparability with the main matrix) and the end-of-training one
# (the maximal-destruction bound) -- go through scripts/eval_retention.py against task A's held-out
# windows. When early stopping never fires, the two checkpoints are the same state and the two JSONs
# agree; that is information, not redundancy, and it is recorded rather than short-circuited.
#
# NO GREP PIPE ON THE RUNNER. A previous overnight sweep in this repo piped the runner through grep,
# which discarded the exit status, so a crash 40 minutes in was reported as success and cost a
# 16-hour window. Every run here writes a complete log, its exit status is captured directly, and the
# per-run verdict is appended to a summary file that the poller reads.
#
# Usage:  nohup bash scripts/run_positive_control.sh              > logs/pc_grid.log 2>&1 &
#         nohup bash scripts/run_positive_control.sh --extension  > logs/pc_ext.log  2>&1 &
#
# --extension runs ONLY the single declared-in-advance rung (lr=1e-1). The registration permits it
# once, if destruction was not achieved at 1e-2, and then the arm stops whatever it shows.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

PY=.venv-probe/bin/python          # torch 2.13 + MPS, the interpreter the main matrix ran on
PYA=.venv12/bin/python             # analysis only
REG=results/positive_control/preregistration.json
SUMMARY=results/positive_control/grid_status.tsv
mkdir -p logs results/positive_control

[ -f "$REG" ] || { echo "no $REG -- the grid may not run before it is registered"; exit 2; }

EXTENSION=0
for arg in "$@"; do
  case "$arg" in
    --extension) EXTENSION=1 ;;
    *) echo "unknown flag: $arg"; exit 2 ;;
  esac
done

# Read the grid out of the registration. Never typed here.
read -r LRS CONDS SEEDS EPOCHS NSAMP BATCH EXTLR < <($PYA - <<'EOF'
import json
r = json.load(open("results/positive_control/preregistration.json"))["grid"]
print(",".join(str(x) for x in r["learning_rates"]), ",".join(r["conditions"]),
      ",".join(str(s) for s in r["seeds"]), r["epochs"], r["max_train_samples"],
      r["batch_size"], r["declared_extension_lr"])
EOF
)
TASK_B=$($PYA -c "import json;print(json.load(open('$REG'))['task_b']['path'])")
TASK_A=$($PYA -c "import json;print(json.load(open('$REG'))['task_a']['data_path'])")
SIZE=$($PYA -c "import json;print(json.load(open('$REG'))['task_a']['model_size'])")
HZ=$($PYA -c "import json;print(json.load(open('$REG'))['task_a']['horizon'])")

if [ "$EXTENSION" = 1 ]; then
  LRS=$EXTLR
  echo "EXTENSION RUN: the single declared rung lr=$EXTLR only. The arm stops after this."
fi

echo "=============================================================================="
echo "PRE-REGISTERED POSITIVE CONTROL"
echo "=============================================================================="
echo "  registration   $REG"
echo "  task A         $TASK_A   (Moirai-$SIZE, h=$HZ)"
echo "  task B         $TASK_B"
echo "  learning rates $LRS"
echo "  conditions     $CONDS"
echo "  seeds          $SEEDS"
echo "  epochs $EPOCHS   n=$NSAMP   batch=$BATCH"
echo

# Task B is gitignored and regenerated, not downloaded. Verify it is the series that was registered
# BEFORE spending hours on it: a different seed or a changed generator would silently make this a
# different experiment from the one the paper describes.
if ! $PYA scripts/make_conflicting_series.py --check > /tmp/pc_taskb.txt 2>&1; then
  echo "STOPPED: task B does not match results/synthetic_task_b.json"; sed 's/^/    /' /tmp/pc_taskb.txt
  exit 2
fi
echo "  task B hash matches the registration"
if ! $PYA scripts/data_manifest.py --check > /tmp/pc_taska.txt 2>&1; then
  echo "STOPPED: task A's benchmark CSV does not match the manifest"; sed 's/^/    /' /tmp/pc_taska.txt
  exit 2
fi
echo "  task A CSV matches results/data_manifest.json"
echo

[ -f "$SUMMARY" ] || printf "cell\tstage\tstatus\tseconds\n" > "$SUMMARY"
t_all=$(date +%s)
fail=0

for lr in ${LRS//,/ }; do
  for cond in ${CONDS//,/ }; do
    for seed in ${SEEDS//,/ }; do
      cell="lr${lr}_${cond}_s${seed}"
      out="results/positive_control/$cell"
      log="logs/pc_${cell}.log"

      # Idempotent: the pilot is one of these cells and must not be re-run. Resume-safe if the
      # overnight job is interrupted.
      if [ -f "$out/condition_${cond}_h${HZ}_s${seed}.json" ]; then
        echo "-- $cell  fine-tune already present, skipping"
      else
        echo "-- $cell  fine-tuning on task B   $(date +%H:%M:%S)"
        t0=$(date +%s)
        # No pipe: the exit status must be the runner's.
        PYTORCH_ENABLE_MPS_FALLBACK=1 $PY -u scripts/finetune_forecasting.py \
          --data-path "$TASK_B" --condition "$cond" --model-size "$SIZE" --horizon "$HZ" \
          --epochs "$EPOCHS" --seed "$seed" --lr "$lr" --max-train-samples "$NSAMP" \
          --batch-size "$BATCH" --device mps \
          --early-stopping --save-best-encoder --save-final-state \
          --results-dir "$out" > "$log" 2>&1
        rc=$?
        dt=$(( $(date +%s) - t0 ))
        if [ "$rc" != 0 ]; then
          echo "   FAILED rc=$rc after ${dt}s -- see $log"
          tail -15 "$log" | sed 's/^/      /'
          printf "%s\tfinetune\tFAILED(rc=%d)\t%d\n" "$cell" "$rc" "$dt" >> "$SUMMARY"
          fail=$((fail + 1))
          continue
        fi
        printf "%s\tfinetune\tok\t%d\n" "$cell" "$dt" >> "$SUMMARY"
        echo "   fine-tune ok in ${dt}s"
      fi

      # Score BOTH checkpoints on task A.
      for ck in final_state best_encoder; do
        src="$out/${ck}.pt"
        dst="$out/retention_A_${ck}.json"
        [ -f "$dst" ] && { echo "   retention($ck) already present"; continue; }
        if [ ! -f "$src" ]; then
          echo "   NO CHECKPOINT $src -- cannot score $ck on task A"
          printf "%s\tretention_%s\tMISSING_CKPT\t0\n" "$cell" "$ck" >> "$SUMMARY"
          fail=$((fail + 1)); continue
        fi
        t0=$(date +%s)
        PYTORCH_ENABLE_MPS_FALLBACK=1 $PY -u scripts/eval_retention.py \
          --state "$src" --task-a-data "$TASK_A" --model-size "$SIZE" --horizon "$HZ" \
          --device mps --out "$dst" > "logs/pc_${cell}_ret_${ck}.log" 2>&1
        rc=$?
        dt=$(( $(date +%s) - t0 ))
        if [ "$rc" != 0 ]; then
          echo "   retention($ck) FAILED rc=$rc -- see logs/pc_${cell}_ret_${ck}.log"
          tail -10 "logs/pc_${cell}_ret_${ck}.log" | sed 's/^/      /'
          printf "%s\tretention_%s\tFAILED(rc=%d)\t%d\n" "$cell" "$ck" "$rc" "$dt" >> "$SUMMARY"
          fail=$((fail + 1)); continue
        fi
        printf "%s\tretention_%s\tok\t%d\n" "$cell" "$ck" "$dt" >> "$SUMMARY"
        r=$($PYA -c "import json;print('%+.2f' % json.load(open('$dst'))['retention_A_pct'])")
        c=$($PYA -c "import json;print('%.4f' % json.load(open('$dst'))['cka_on_task_a_inputs'])")
        echo "   retention($ck): retention_A=${r}%  CKA=${c}  (${dt}s)"
      done
    done
  done
done

echo
echo "=============================================================================="
printf "TOTAL %s   failures: %s\n" "$(( ($(date +%s) - t_all) / 60 )) min" "$fail"
if [ "$fail" != 0 ]; then
  echo "THE GRID DID NOT COMPLETE. $fail step(s) failed -- read $SUMMARY and the per-run logs."
  echo "Do NOT read the emitted table as the full grid until this is zero."
fi
echo "ALL RUNS ACCOUNTED FOR (this line means the loop ended, NOT that every run succeeded --"
echo "the failure count above is the thing to read)."
exit "$fail"
