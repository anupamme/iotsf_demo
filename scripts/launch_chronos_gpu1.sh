#!/bin/bash
# GPU Instance 1 (g5.xlarge): Condition B — full fine-tune
# Runs: 10 seeds × n=500 + 10 seeds × n=10k
# Estimated: ~6-8h total on A10G

set -e

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

SEEDS=(42 101 123 202 303 456 777 789 888 999)
SCRIPT="scripts/finetune_chronos_m4.py"

echo "=========================================="
echo "Chronos-T5-Small × M4-Monthly: Condition B"
echo "GPU Instance 1"
echo "=========================================="

# Phase 1: n=500, 10 seeds (~2-3h)
echo ""
echo "=== Phase 1: Condition B, n=500 ==="
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed $SEED, n=500 ---"
    python3 $SCRIPT \
        --model-id /home/ubuntu/models/chronos-t5-small \
        --condition B \
        --seed $SEED \
        --epochs 20 \
        --batch-size 32 \
        --lr 1e-5 \
        --max-train-samples 500 \
        --device cuda \
        --deterministic \
        --early-stopping \
        --save-best-encoder \
        --results-dir results/chronos_m4_n500
done

echo ""
echo "=== Phase 1 complete ==="
echo ""

# Phase 2: n=10k, 10 seeds (~4-5h)
echo "=== Phase 2: Condition B, n=10000 ==="
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed $SEED, n=10000 ---"
    python3 $SCRIPT \
        --model-id /home/ubuntu/models/chronos-t5-small \
        --condition B \
        --seed $SEED \
        --epochs 20 \
        --batch-size 32 \
        --lr 1e-5 \
        --max-train-samples 10000 \
        --device cuda \
        --deterministic \
        --early-stopping \
        --save-best-encoder \
        --results-dir results/chronos_m4_n10k
done

echo ""
echo "=========================================="
echo "GPU 1 COMPLETE"
echo "=========================================="
echo ""

# Summary
echo "=== Results Summary ==="
echo "n=500:"
for SEED in "${SEEDS[@]}"; do
    if [ -f "results/chronos_m4_n500/seed${SEED}/condition_B_s${SEED}.json" ]; then
        python3 -c "
import json
d = json.load(open('results/chronos_m4_n500/seed${SEED}/condition_B_s${SEED}.json'))
print(f'  seed ${SEED}: forg={d[\"forgetting_pct\"]:+.1f}% CKA={d[\"final_cka\"]:.3f} ΔR²={d[\"linear_probe\"][\"r2_delta\"]:+.3f}')
"
    fi
done

echo "n=10k:"
for SEED in "${SEEDS[@]}"; do
    if [ -f "results/chronos_m4_n10k/seed${SEED}/condition_B_s${SEED}.json" ]; then
        python3 -c "
import json
d = json.load(open('results/chronos_m4_n10k/seed${SEED}/condition_B_s${SEED}.json'))
print(f'  seed ${SEED}: forg={d[\"forgetting_pct\"]:+.1f}% CKA={d[\"final_cka\"]:.3f} ΔR²={d[\"linear_probe\"][\"r2_delta\"]:+.3f}')
"
    fi
done
