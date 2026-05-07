#!/bin/bash
# GPU Instance 2 (g5.xlarge): Controls — frozen encoder + random-init
# Runs: Condition D (10 seeds × n=10k) + Random-init (10 seeds × n=10k)
# Estimated: ~4-5h total on A10G

set -e

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

SEEDS=(42 101 123 202 303 456 777 789 888 999)
SCRIPT="scripts/finetune_chronos_m4.py"

echo "=========================================="
echo "Chronos-T5-Small × M4-Monthly: Controls"
echo "GPU Instance 2"
echo "=========================================="

# Phase 1: Frozen encoder (Condition D), n=10k, 10 seeds (~2h)
echo ""
echo "=== Phase 1: Condition D (frozen encoder), n=10000 ==="
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed $SEED, frozen encoder ---"
    python3 $SCRIPT \
        --model-id /home/ubuntu/models/chronos-t5-small \
        --condition D \
        --seed $SEED \
        --epochs 20 \
        --batch-size 32 \
        --lr 1e-5 \
        --max-train-samples 10000 \
        --device cuda \
        --deterministic \
        --early-stopping \
        --save-best-encoder \
        --results-dir results/chronos_m4_frozen
done

echo ""
echo "=== Phase 1 complete ==="
echo ""

# Phase 2: Random-init control, n=10k, 10 seeds (~2-3h)
echo "=== Phase 2: Random-init control, n=10000 ==="
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed $SEED, random-init ---"
    python3 $SCRIPT \
        --model-id /home/ubuntu/models/chronos-t5-small \
        --condition B \
        --random-init \
        --seed $SEED \
        --epochs 20 \
        --batch-size 32 \
        --lr 1e-5 \
        --max-train-samples 10000 \
        --device cuda \
        --deterministic \
        --early-stopping \
        --save-best-encoder \
        --results-dir results/chronos_m4_randinit
done

echo ""
echo "=========================================="
echo "GPU 2 COMPLETE"
echo "=========================================="
echo ""

# Summary
echo "=== Results Summary ==="
echo ""
echo "Frozen encoder (Condition D):"
for SEED in "${SEEDS[@]}"; do
    if [ -f "results/chronos_m4_frozen/seed${SEED}/condition_D_s${SEED}.json" ]; then
        python3 -c "
import json
d = json.load(open('results/chronos_m4_frozen/seed${SEED}/condition_D_s${SEED}.json'))
print(f'  seed ${SEED}: forg={d[\"forgetting_pct\"]:+.1f}% CKA={d[\"final_cka\"]:.3f} ΔR²={d[\"linear_probe\"][\"r2_delta\"]:+.3f}')
"
    fi
done

echo ""
echo "Random-init control:"
for SEED in "${SEEDS[@]}"; do
    if [ -f "results/chronos_m4_randinit/seed${SEED}/condition_B_s${SEED}.json" ]; then
        python3 -c "
import json
d = json.load(open('results/chronos_m4_randinit/seed${SEED}/condition_B_s${SEED}.json'))
print(f'  seed ${SEED}: forg={d[\"forgetting_pct\"]:+.1f}% CKA={d[\"final_cka\"]:.3f} ΔR²={d[\"linear_probe\"][\"r2_delta\"]:+.3f}')
"
    fi
done
