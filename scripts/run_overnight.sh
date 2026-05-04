#!/usr/bin/env bash
# Overnight orchestration: ETTh1 n=10k (all 10 seeds) + LoRA-Large k=5 + MLP probe + paper update
# Seeds 42 and 101 are already running (PIDs 47119, 47118). LoRA seed303 running (PID 48683).
# This script runs the remaining seeds sequentially and updates the paper when done.

set -euo pipefail

PYTHON=/opt/homebrew/Caskroom/miniconda/base/envs/iotsf/bin/python
REPO=/Users/mediratta/code/iotsf_demo
LOG_DIR=/tmp/overnight_runs
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/orchestration.log"; }

# ──────────────────────────────────────────────
# ETTh1 n=10k remaining seeds (42 and 101 already running)
# Run 2 at a time after the first pair finishes
# ──────────────────────────────────────────────
ETTH1_SEEDS=(123 202 303 456 777 789 888 999)

run_etth1_seed() {
    local seed=$1
    local outdir="$REPO/results/v21_etth1_n10k/seed${seed}"
    local logfile="$LOG_DIR/etth1_seed${seed}.log"
    if [ -f "$outdir/best_encoder.pt" ]; then
        log "ETTh1 seed${seed}: already done, skipping"
        return 0
    fi
    log "ETTh1 seed${seed}: starting"
    PYTORCH_ENABLE_MPS_FALLBACK=1 "$PYTHON" "$REPO/scripts/finetune_forecasting.py" \
        --data-path "$REPO/data/forecasting/ETTh1.csv" \
        --condition B --max-train-samples 10000 \
        --seed "$seed" --early-stopping --save-best-encoder --deterministic --device mps \
        --results-dir "$outdir" \
        > "$logfile" 2>&1
    if [ -f "$outdir/best_encoder.pt" ]; then
        log "ETTh1 seed${seed}: DONE"
    else
        log "ETTh1 seed${seed}: FAILED (no best_encoder.pt)"
    fi
}

wait_for_etth1_pair() {
    # Wait until both seed42 and seed101 are done before starting next batch
    log "Waiting for seeds 42 and 101 to finish..."
    while true; do
        done42=0; done101=0
        [ -f "$REPO/results/v21_etth1_n10k/seed42/best_encoder.pt" ] && done42=1
        [ -f "$REPO/results/v21_etth1_n10k/seed101/best_encoder.pt" ] && done101=1
        if [ $done42 -eq 1 ] && [ $done101 -eq 1 ]; then
            log "Seeds 42 and 101 done."
            break
        fi
        sleep 120
    done
}

# ──────────────────────────────────────────────
# LoRA-Large seed789 (after seed303 finishes)
# ──────────────────────────────────────────────
wait_for_lora303() {
    log "Waiting for LoRA-Large seed303 to finish..."
    while true; do
        if [ -f "$REPO/results/v21_lora_large_k5/seed303/condition_E_h96_s303.json" ]; then
            log "LoRA-Large seed303 done."
            break
        fi
        sleep 120
    done
}

run_lora_seed() {
    local seed=$1
    local outdir="$REPO/results/v21_lora_large_k5/seed${seed}"
    local logfile="$LOG_DIR/lora_seed${seed}.log"
    if [ -f "$outdir/condition_E_h96_s${seed}.json" ]; then
        log "LoRA seed${seed}: already done, skipping"
        return 0
    fi
    log "LoRA seed${seed}: starting"
    "$PYTHON" "$REPO/scripts/finetune_forecasting.py" \
        --data-path "$REPO/data/forecasting/ETTh2.csv" \
        --condition E --max-train-samples 500 \
        --model-size large --lr 1e-5 --seed "$seed" \
        --results-dir "$outdir" \
        > "$logfile" 2>&1
    log "LoRA seed${seed}: DONE"
}

# ──────────────────────────────────────────────
# MLP probe on all 10 ETTh1 encoders
# ──────────────────────────────────────────────
run_mlp_probe_all() {
    log "Running MLP probe on all ETTh1 encoders..."
    local probe_dir="$REPO/results/v21_etth1_mlp"
    mkdir -p "$probe_dir"

    # Zero-shot reference
    if [ ! -f "$probe_dir/zeroshot_mlp.json" ]; then
        log "MLP probe: zero-shot reference"
        "$PYTHON" "$REPO/scripts/reprobe_saved_encoders.py" \
            --zero-shot \
            --probe-types mlp \
            --head-types forecast96 \
            --mlp-layers 1,2,5 \
            --data-path "$REPO/data/forecasting/ETTh1.csv" \
            --device cpu \
            --out-path "$probe_dir/zeroshot_mlp.json" \
            >> "$LOG_DIR/mlp_probe.log" 2>&1
        log "MLP probe: zero-shot done"
    fi

    for seed in 42 101 123 202 303 456 777 789 888 999; do
        local enc_dir="$REPO/results/v21_etth1_n10k/seed${seed}"
        local out="$probe_dir/seed${seed}_mlp.json"
        if [ -f "$out" ]; then
            log "MLP probe seed${seed}: already done, skipping"
            continue
        fi
        if [ ! -f "$enc_dir/best_encoder.pt" ]; then
            log "MLP probe seed${seed}: encoder missing, skipping"
            continue
        fi
        log "MLP probe seed${seed}: running"
        "$PYTHON" "$REPO/scripts/reprobe_saved_encoders.py" \
            --encoder-dir "$enc_dir" \
            --probe-types mlp \
            --head-types forecast96 \
            --mlp-layers 1,2,5 \
            --data-path "$REPO/data/forecasting/ETTh1.csv" \
            --device cpu \
            --out-path "$out" \
            >> "$LOG_DIR/mlp_probe.log" 2>&1
        log "MLP probe seed${seed}: done"
    done
    log "All MLP probes complete."
}

# ──────────────────────────────────────────────
# Paper update script (called after all results in)
# ──────────────────────────────────────────────
update_paper() {
    log "Updating paper with results..."
    "$PYTHON" "$REPO/scripts/update_paper_results.py" \
        >> "$LOG_DIR/paper_update.log" 2>&1
    log "Paper update done."
}

# ──────────────────────────────────────────────
# Main execution
# ──────────────────────────────────────────────
log "=== Overnight orchestration started ==="

# Step 1: Wait for seeds 42+101, then run remaining ETTh1 seeds in pairs
wait_for_etth1_pair

# Run next 2 seeds in parallel
log "Starting ETTh1 seeds 123 and 202"
run_etth1_seed 123 &
run_etth1_seed 202 &
wait

log "Starting ETTh1 seeds 303 and 456"
run_etth1_seed 303 &
run_etth1_seed 456 &
wait

log "Starting ETTh1 seeds 777 and 789"
run_etth1_seed 777 &
run_etth1_seed 789 &
wait

log "Starting ETTh1 seeds 888 and 999"
run_etth1_seed 888 &
run_etth1_seed 999 &
wait

log "All 10 ETTh1 seeds complete."

# Step 2: MLP probe on all encoders
run_mlp_probe_all

# Step 3: Wait for LoRA seed303, run seed789
wait_for_lora303
run_lora_seed 789

# Step 4: Update paper with all results
"$PYTHON" "$REPO/scripts/compute_and_update_paper.py" \
    >> "$LOG_DIR/paper_update.log" 2>&1 && log "Paper updated." || log "Paper update script not found — manual update needed."

log "=== Overnight orchestration complete ==="
