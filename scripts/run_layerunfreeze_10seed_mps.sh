#!/usr/bin/env bash
# Re-run the layer-wise unfreeze comparison at 10 seeds per arm, one environment.
#
# WHY -------------------------------------------------------------------------
# tab:layerunfreeze's N=3 row claims 10 CUDA seeds at CKA 0.663+-0.044 and
# forgetting -6.5+-4.0%, but only three records exist (results/v26_layer_unfreeze,
# seeds 42/101/123) and those ran with early_stopping DISABLED, giving CKA 0.698
# and forgetting +1.33 -- the sign of the body's claim flips. The N=6 row's
# 0.387+-0.052 / -6.7+-6.3 matches no run set on disk either: the 10-seed n=10k
# condition-B sets give CKA 0.518 / -5.3+-6.2 (CUDA, v19_cuda_etth2_n10k) and
# CKA 0.568 / -5.7+-10.4 (MPS, v18_mps_deterministic_n10k).
#
# BOTH arms are re-run because the body claim is a comparison between them, and
# the environment that produced the existing N=6 sets cannot be reconstructed on
# this machine. A comparison is only traceable if both arms share one environment.
#
# ENVIRONMENT -----------------------------------------------------------------
# .venv12 (Python 3.12) with uni2ts 2.0.0. The uni2ts gradient patch to
# PackedStdScaler is applied at runtime by src/models/moirai_detector.py, so no
# source patching is needed -- but verify UNI2TS_AVAILABLE is True, because the
# code silently falls back to a MOCK Moirai when uni2ts is missing.
#
# Two documented deviations from the original runs, both forced:
#   1. torch 2.14 is used, above uni2ts's declared pin of <2.5. At torch 2.4.1
#      the NegativeBinomial component of Moirai's output mixture collapses to
#      total_count=0 on MPS and training dies in the first NLL step.
#   2. PYTORCH_ENABLE_MPS_FALLBACK=1 is required: aten::_cummax_helper has no MPS
#      kernel, so uni2ts's time-id generation falls back to CPU for that op.
# Record both in the table caption alongside the device.
#
# Resumable: a seed whose JSON already exists is skipped, so this can be
# re-launched after an interruption without losing or duplicating work.
# Cost: ~45 min/run measured at n=10k, 20 epochs => ~15 h for all 20 runs.

set -u

cd /Users/mediratta/code/paper_writing/iotsf_demo

export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv12/bin/python

SEEDS=(42 101 123 202 303 456 777 789 888 999)
OUT_ROOT=results/v49_layerunfreeze_10seed
LOG_ROOT=logs/v49_layerunfreeze
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

# Fail fast rather than silently producing mock-Moirai numbers.
if ! $PY -c "import src.models.moirai_detector as m; raise SystemExit(0 if m.UNI2TS_AVAILABLE else 1)" 2>/dev/null; then
  echo "FATAL: uni2ts unavailable -- would run a mock Moirai. Aborting."
  exit 1
fi
echo "uni2ts check passed"
$PY -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"

run_arm () {
  local tag="$1" cond="$2" nlayers="$3"
  echo ""
  echo "=== $tag: condition $cond, ETTh2 n=10k h=96, MPS, early-stopped, deterministic ==="
  for seed in "${SEEDS[@]}"; do
    local seed_dir="$OUT_ROOT/${tag}_seed${seed}"
    local log_file="$LOG_ROOT/${tag}_seed${seed}.log"
    if [ -f "$seed_dir/condition_${cond}_h96_s${seed}.json" ]; then
      echo "[skip]  $tag seed $seed already done"
      continue
    fi
    mkdir -p "$seed_dir"
    echo "[start] $(date '+%H:%M:%S') $tag seed=$seed"
    local extra=""
    [ "$nlayers" != "-" ] && extra="--unfreeze-top-n-layers $nlayers"
    $PY scripts/finetune_forecasting.py \
      --data-path data/forecasting/ETTh2.csv \
      --condition "$cond" --model-size small --horizon 96 \
      --epochs 20 --max-train-samples 10000 --eval-every 1 --device mps \
      --seed "$seed" --early-stopping --save-best-encoder --deterministic \
      $extra \
      --results-dir "$seed_dir" \
      > "$log_file" 2>&1
    echo "[done]  $(date '+%H:%M:%S') $tag seed=$seed exit=$?"
  done
}

echo "start $(date '+%Y-%m-%d %H:%M:%S')"

# Phase 1: the arm that the body claim actually lacks.
run_arm N3 D 3

# Phase 2: matched comparator, same environment.
run_arm N6 B -

echo ""
echo "finish $(date '+%Y-%m-%d %H:%M:%S')"
echo "N3 JSONs: $(ls "$OUT_ROOT"/N3_seed*/condition_D_h96_s*.json 2>/dev/null | wc -l | tr -d ' ')/10"
echo "N6 JSONs: $(ls "$OUT_ROOT"/N6_seed*/condition_B_h96_s*.json 2>/dev/null | wc -l | tr -d ' ')/10"
