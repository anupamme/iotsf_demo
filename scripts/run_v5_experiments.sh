#!/bin/bash
# V5 Revision Experiments — All phases
# Usage: conda run -n iotsf bash scripts/run_v5_experiments.sh [DEVICE]
# DEVICE defaults to 'cpu', pass 'cuda' or 'mps' for GPU

set -e
DEVICE=${1:-cpu}
SCRIPT="python scripts/finetune_forecasting.py"
SEEDS="42 123 456 789 999"
SEEDS3="42 123 456"

echo "=== V5 Experiments — Device: $DEVICE ==="
echo "Started: $(date)"

# ============================================================
# Phase 1a: ETTh1 — Conditions A/B/C/D, h=96/192, 5 seeds
# ============================================================
echo "--- Phase 1a: ETTh1 ---"
for H in 96 192; do
  for COND in A B C D; do
    for S in $SEEDS; do
      echo "ETTh1 h=$H cond=$COND seed=$S"
      $SCRIPT --data-path data/forecasting/ETTh1.csv --condition $COND \
        --horizon $H --epochs 20 --seed $S --device $DEVICE \
        --results-dir results/v5_etth1/h${H}/condition_${COND}
    done
  done
done

# ============================================================
# Phase 1a: ETTm2 — Conditions A/B/C/D, h=96/192, 5 seeds
# ============================================================
echo "--- Phase 1a: ETTm2 ---"
for H in 96 192; do
  for COND in A B C D; do
    for S in $SEEDS; do
      echo "ETTm2 h=$H cond=$COND seed=$S"
      $SCRIPT --data-path data/forecasting/ETTm2.csv --condition $COND \
        --horizon $H --epochs 20 --seed $S --device $DEVICE \
        --results-dir results/v5_ettm2/h${H}/condition_${COND}
    done
  done
done

# ============================================================
# Phase 1b: Moirai-Base on ETTh2 — Conditions A/B/D, h=96/192, 3 seeds
# ============================================================
echo "--- Phase 1b: Moirai-Base ---"
for H in 96 192; do
  for COND in A B D; do
    for S in $SEEDS3; do
      echo "ETTh2-Base h=$H cond=$COND seed=$S"
      $SCRIPT --data-path data/forecasting/ETTh2.csv --condition $COND \
        --model-size base --horizon $H --epochs 20 --seed $S --device $DEVICE \
        --results-dir results/v5_etth2_base/h${H}/condition_${COND}
    done
  done
done

# ============================================================
# Phase 1c: Training-set size sweep on ETTh2 — Condition B, h=96
# ============================================================
echo "--- Phase 1c: Sample size sweep ---"
for N in 200 500 1000 2000; do
  for S in $SEEDS3; do
    echo "ETTh2 samples=$N seed=$S"
    $SCRIPT --data-path data/forecasting/ETTh2.csv --condition B \
      --horizon 96 --epochs 20 --max-train-samples $N --seed $S --device $DEVICE \
      --results-dir results/v5_etth2_sweep/n${N}
  done
done

# ============================================================
# Phase 2a: LoRA on ETTh2 — h=96/192, 5 seeds
# ============================================================
echo "--- Phase 2a: LoRA ---"
for H in 96 192; do
  for S in $SEEDS; do
    echo "LoRA h=$H seed=$S"
    $SCRIPT --data-path data/forecasting/ETTh2.csv --condition E \
      --horizon $H --epochs 20 --seed $S --device $DEVICE \
      --results-dir results/v5_mitigation/lora/h${H}
  done
done

# LoRA on ETTh1 — h=96, 3 seeds
for S in $SEEDS3; do
  echo "LoRA ETTh1 h=96 seed=$S"
  $SCRIPT --data-path data/forecasting/ETTh1.csv --condition E \
    --horizon 96 --epochs 20 --seed $S --device $DEVICE \
    --results-dir results/v5_mitigation/lora_etth1/h96
done

# ============================================================
# Phase 2b: L2-SP on ETTh2 — lambda=0.01, 0.1; h=96/192, 3 seeds
# ============================================================
echo "--- Phase 2b: L2-SP ---"
for L in 0.01 0.1; do
  for H in 96 192; do
    for S in $SEEDS3; do
      echo "L2-SP l=$L h=$H seed=$S"
      $SCRIPT --data-path data/forecasting/ETTh2.csv --condition F \
        --l2sp-weight $L --horizon $H --epochs 20 --seed $S --device $DEVICE \
        --results-dir results/v5_mitigation/l2sp_${L}/h${H}
    done
  done
done

# ============================================================
# Phase 2c: EWC on ETTh2 — lambda=100, 1000; h=96/192, 3 seeds
# ============================================================
echo "--- Phase 2c: EWC ---"
for L in 100 1000; do
  for H in 96 192; do
    for S in $SEEDS3; do
      echo "EWC l=$L h=$H seed=$S"
      $SCRIPT --data-path data/forecasting/ETTh2.csv --condition G \
        --ewc-lambda $L --horizon $H --epochs 20 --seed $S --device $DEVICE \
        --results-dir results/v5_mitigation/ewc_${L}/h${H}
    done
  done
done

echo "=== All V5 experiments complete ==="
echo "Finished: $(date)"
