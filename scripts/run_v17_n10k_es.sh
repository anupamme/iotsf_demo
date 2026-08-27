#!/usr/bin/env bash
# V17 Phase 1B/1C: n=10k, condition B, Moirai-Small, early-stopping, 6 seeds (serial).
# Seeds 202, 303, 999, 777, 888 (fresh) + 101 (V16 outlier rerun).
# Empirical clock: ~1.5h/seed on MPS → ~9h total.

set -u
cd /Users/mediratta/code/iotsf_demo
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate iotsf

SEEDS=(202 303 999 777 888 101)
OUT_ROOT=results/v17_etth2_n10k_es
LOG_ROOT=logs/v17_n10k_es
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
  PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/finetune_forecasting.py \
    --data-path data/forecasting/ETTh2.csv \
    --condition B --model-size small --horizon 96 \
    --epochs 20 --max-train-samples 10000 --eval-every 1 --device mps \
    --seed "$seed" --early-stopping \
    --results-dir "$seed_dir" \
    > "$log_file" 2>&1
  status=$?
  echo "[done] $(date +%H:%M:%S) seed=$seed exit=$status"
done

echo "[all_done] $(date +%H:%M:%S)"
