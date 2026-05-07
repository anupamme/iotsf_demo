#!/usr/bin/env bash
# Experiment 1: 5 additional ETTm2 seeds (456/777/789/888/999)
# Config: ETTm2, n=10k, condition B (NLL-only), h=96, 20 epochs, AdamW lr=1e-4
# Hardware: g5.xlarge (A10G)
# Expected time: ~4 GPU-hours

set -eu
cd ~/iotsf_demo

SEEDS=(456 777 789 888 999)
OUT_ROOT=results/exp1_ettm2_n10k_es
LOG_ROOT=logs/exp1_ettm2
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

for seed in "${SEEDS[@]}"; do
  seed_dir="$OUT_ROOT/seed${seed}"
  log_file="$LOG_ROOT/seed${seed}.log"
  if [ -f "$seed_dir/condition_B_h96_s${seed}.json" ]; then
    echo "[skip] seed $seed already done"
    continue
  fi
  mkdir -p "$seed_dir"
  echo "[start] $(date +%H:%M:%S) seed=$seed"
  python3 scripts/finetune_forecasting.py \
    --data-path data/forecasting/ETTm2.csv \
    --condition B --model-size small --horizon 96 \
    --epochs 20 --max-train-samples 10000 --eval-every 1 --device cuda \
    --seed "$seed" --early-stopping --save-best-encoder --deterministic \
    --results-dir "$seed_dir" \
    > "$log_file" 2>&1
  status=$?
  echo "[done] $(date +%H:%M:%S) seed=$seed exit=$status"
done

echo ""
echo "=== Experiment 1 Complete ==="
echo "Running probes on saved encoders..."

# Run reprobe on each saved encoder
for seed in "${SEEDS[@]}"; do
  seed_dir="$OUT_ROOT/seed${seed}"
  if [ ! -f "$seed_dir/best_encoder.pt" ]; then
    echo "[warn] No best_encoder.pt for seed $seed"
    continue
  fi
  echo "[probe] seed=$seed"
  python3 scripts/reprobe_saved_encoders.py \
    --encoder-dir "$seed_dir" \
    --probe-types ridge,gbm \
    --head-types forecast96,delta1 \
    --data-path data/forecasting/ETTm2.csv \
    --out-path "$OUT_ROOT/probe_seed${seed}.json" \
    2>&1 | tail -3
done

# Also probe ZS baseline for reference
echo "[probe] zero-shot baseline"
python3 scripts/reprobe_saved_encoders.py \
  --zero-shot \
  --probe-types ridge,gbm \
  --head-types forecast96,delta1 \
  --data-path data/forecasting/ETTm2.csv \
  --out-path "$OUT_ROOT/probe_zeroshot.json" \
  2>&1 | tail -3

echo ""
echo "=== All probes complete ==="
echo "Results in: $OUT_ROOT/"
ls -la "$OUT_ROOT"/*.json 2>/dev/null | wc -l
echo " JSON files produced."
