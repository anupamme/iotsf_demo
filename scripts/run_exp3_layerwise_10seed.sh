#!/usr/bin/env bash
# Experiment 3: Extend layer-wise unfreeze to 10 seeds
# Config: ETTh2, n=10k, condition D with --unfreeze-top-n-layers 3 and 6 (full B)
# Seeds: 202/303/456/777/789/888/999 (7 additional; existing: 42/101/123)
# Hardware: g5.xlarge (A10G)
# Expected time: ~6 GPU-hours (2 conditions × 7 seeds × ~25 min)

set -eu
cd ~/iotsf_demo

SEEDS=(202 303 456 777 789 888 999)
OUT_ROOT=results/exp3_layerwise_10seed
LOG_ROOT=logs/exp3_layerwise
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "=== Phase 1: N=3 (top-3 layers unfrozen) ==="
for seed in "${SEEDS[@]}"; do
  seed_dir="$OUT_ROOT/N3_seed${seed}"
  log_file="$LOG_ROOT/N3_seed${seed}.log"
  if [ -f "$seed_dir/condition_D_h96_s${seed}.json" ]; then
    echo "[skip] N=3 seed $seed already done"
    continue
  fi
  mkdir -p "$seed_dir"
  echo "[start] $(date +%H:%M:%S) N=3 seed=$seed"
  python3 scripts/finetune_forecasting.py \
    --data-path data/forecasting/ETTh2.csv \
    --condition D --model-size small --horizon 96 \
    --epochs 20 --max-train-samples 10000 --eval-every 1 --device cuda \
    --seed "$seed" --early-stopping --save-best-encoder --deterministic \
    --unfreeze-top-n-layers 3 \
    --results-dir "$seed_dir" \
    > "$log_file" 2>&1
  status=$?
  echo "[done] $(date +%H:%M:%S) N=3 seed=$seed exit=$status"
done

echo ""
echo "=== Phase 2: N=6 (full unfreeze, condition B) ==="
for seed in "${SEEDS[@]}"; do
  seed_dir="$OUT_ROOT/N6_seed${seed}"
  log_file="$LOG_ROOT/N6_seed${seed}.log"
  if [ -f "$seed_dir/condition_B_h96_s${seed}.json" ]; then
    echo "[skip] N=6 seed $seed already done"
    continue
  fi
  mkdir -p "$seed_dir"
  echo "[start] $(date +%H:%M:%S) N=6 seed=$seed"
  python3 scripts/finetune_forecasting.py \
    --data-path data/forecasting/ETTh2.csv \
    --condition B --model-size small --horizon 96 \
    --epochs 20 --max-train-samples 10000 --eval-every 1 --device cuda \
    --seed "$seed" --early-stopping --save-best-encoder --deterministic \
    --results-dir "$seed_dir" \
    > "$log_file" 2>&1
  status=$?
  echo "[done] $(date +%H:%M:%S) N=6 seed=$seed exit=$status"
done

echo ""
echo "=== Running probes ==="
for mode in N3 N6; do
  cond="D"
  [ "$mode" = "N6" ] && cond="B"
  for seed in "${SEEDS[@]}"; do
    seed_dir="$OUT_ROOT/${mode}_seed${seed}"
    if [ ! -f "$seed_dir/best_encoder.pt" ]; then
      echo "[warn] No best_encoder.pt for $mode seed $seed"
      continue
    fi
    echo "[probe] $mode seed=$seed"
    python3 scripts/reprobe_saved_encoders.py \
      --encoder-dir "$seed_dir" \
      --probe-types ridge \
      --head-types forecast96 \
      --data-path data/forecasting/ETTh2.csv \
      --out-path "$OUT_ROOT/probe_${mode}_seed${seed}.json" \
      2>&1 | tail -3
  done
done

# ZS baseline for reference
echo "[probe] zero-shot baseline"
python3 scripts/reprobe_saved_encoders.py \
  --zero-shot \
  --probe-types ridge \
  --head-types forecast96 \
  --data-path data/forecasting/ETTh2.csv \
  --out-path "$OUT_ROOT/probe_zeroshot.json" \
  2>&1 | tail -3

echo ""
echo "=== Experiment 3 Complete ==="
ls -la "$OUT_ROOT"/*.json 2>/dev/null | wc -l
echo " JSON files produced."
