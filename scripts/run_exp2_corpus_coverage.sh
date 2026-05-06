#!/usr/bin/env bash
# Experiment 2: Corpus-coverage sanity check — Chronos/MOMENT on M4
# Tests whether non-Moirai backbones gate-pass on a dataset documented in their corpora.
# M4-Monthly is documented in both Chronos (public M competitions) and MOMENT corpora.
# Hardware: g5.2xlarge (A10G)
# Expected time: ~2-4 GPU-hours

set -eu
cd ~/iotsf_demo

OUT_ROOT=results/exp2_corpus_coverage
LOG_ROOT=logs/exp2_corpus
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "=== Experiment 2: Corpus-Coverage Gate Check ==="
echo "Running Chronos-T5-Small and MOMENT-1-base on M4-Monthly"
echo ""

# Run Chronos gate check on M4
echo "[1/2] Chronos-T5-Small on M4-Monthly..."
python3 scripts/exp2_m4_gate_check.py \
  --backbone chronos \
  --model-name amazon/chronos-t5-small \
  --out-path "$OUT_ROOT/chronos_m4_monthly.json" \
  > "$LOG_ROOT/chronos_m4.log" 2>&1
echo "[done] Chronos. $(cat "$OUT_ROOT/chronos_m4_monthly.json" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"Gate: {d.get(\"gate_status\",\"?\")}, ZS_MSE={d.get(\"zs_mse_norm\",0):.4f}, Linear_MSE={d.get(\"linear_mse_norm\",0):.4f}")' 2>/dev/null || echo "check log")"

echo ""

# Run MOMENT gate check on M4
echo "[2/2] MOMENT-1-base on M4-Monthly..."
python3 scripts/exp2_m4_gate_check.py \
  --backbone moment \
  --model-name AutonLab/MOMENT-1-large \
  --out-path "$OUT_ROOT/moment_m4_monthly.json" \
  > "$LOG_ROOT/moment_m4.log" 2>&1
echo "[done] MOMENT. $(cat "$OUT_ROOT/moment_m4_monthly.json" 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"Gate: {d.get(\"gate_status\",\"?\")}, ZS_MSE={d.get(\"zs_mse_norm\",0):.4f}, Linear_MSE={d.get(\"linear_mse_norm\",0):.4f}")' 2>/dev/null || echo "check log")"

echo ""
echo "=== Experiment 2 Complete ==="
echo "Results:"
cat "$OUT_ROOT"/*.json 2>/dev/null
