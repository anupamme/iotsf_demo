#!/bin/bash
# Run multi-seed catastrophic forgetting experiments for Path 2
# h=96 and h=192, seeds 42,123,456,789,1234, conditions B,C,D
#
# Usage: bash scripts/run_forgetting_sweep.sh [horizon]
# If no horizon specified, runs h=96 then h=192

set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/iotsf/bin/python3}"
SEEDS="42 123 456 789 1234"
CONDITIONS="B C D"
EPOCHS=20
TRAIN_SAMPLES=500
EVAL_EVERY=5
RESULTS_BASE="results/forecasting_finetune"

run_experiment() {
    local cond=$1
    local horizon=$2
    local seed=$3
    local results_dir="${RESULTS_BASE}_${EPOCHS}ep"
    local outfile="${results_dir}/condition_${cond}_h${horizon}_s${seed}.json"

    # Skip if already exists
    if [ -f "$outfile" ]; then
        echo "[SKIP] $outfile already exists"
        return 0
    fi

    echo "[START] Condition=$cond horizon=$horizon seed=$seed"
    $PYTHON scripts/finetune_forecasting.py \
        --data-path data/forecasting/ETTh2.csv \
        --condition "$cond" \
        --horizon "$horizon" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --max-train-samples "$TRAIN_SAMPLES" \
        --eval-every "$EVAL_EVERY" \
        --results-dir "$results_dir" \
        2>&1 | tail -5
    echo "[DONE] Condition=$cond horizon=$horizon seed=$seed"
    echo "---"
}

HORIZONS="${1:-96 192}"

for horizon in $HORIZONS; do
    echo "=============================================="
    echo "  HORIZON = $horizon"
    echo "=============================================="
    for seed in $SEEDS; do
        for cond in $CONDITIONS; do
            run_experiment "$cond" "$horizon" "$seed"
        done
    done
done

echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "=============================================="

# Print summary
echo ""
echo "Results summary (h=96):"
for cond in $CONDITIONS; do
    echo -n "  Condition $cond: "
    for seed in $SEEDS; do
        f="${RESULTS_BASE}_${EPOCHS}ep/condition_${cond}_h96_s${seed}.json"
        if [ -f "$f" ]; then
            $PYTHON -c "import json; d=json.load(open('$f')); print(f's{d[\"seed\"]}={d[\"forgetting_pct\"]:+.1f}%', end=' ')"
        fi
    done
    echo ""
done

echo ""
echo "Results summary (h=192):"
for cond in $CONDITIONS; do
    echo -n "  Condition $cond: "
    for seed in $SEEDS; do
        f="${RESULTS_BASE}_${EPOCHS}ep/condition_${cond}_h192_s${seed}.json"
        if [ -f "$f" ]; then
            $PYTHON -c "import json; d=json.load(open('$f')); print(f's{d[\"seed\"]}={d[\"forgetting_pct\"]:+.1f}%', end=' ')"
        fi
    done
    echo ""
done
