#!/usr/bin/env bash
# Fill the thin rungs of the partial-unfreeze ladder: N=4 and N=5 at 3 seeds each.
#
# WHY -------------------------------------------------------------------------
# tab:layerunfreeze reports N in {0,3,4,5,6}, but only N=3 and N=6 are a matched
# comparison: 10 early-stopped MPS seeds each (run_layerunfreeze_10seed_mps.sh,
# results/v49_layerunfreeze_10seed). N=4 and N=5 are SINGLE seeds from
# results/v27_layer_unfreeze, read at the final epoch with early stopping
# disabled, which is exactly the reading the matched re-run exists to remove --
# the N=5 row's +31.8% forgetting is a final-epoch overshoot whose val MSE bottoms
# at epoch 7 essentially at zero-shot. The appendix therefore labels both rows
# context and draws no conclusion from them.
#
# These 6 runs put the two thin rungs on the pair's protocol, so the appendix can
# report a four-rung ladder in one environment. They do not create a new claim:
# the drawn claim stays the matched N=3 versus N=6 comparison.
#
# REGISTERED ------------------------------------------------------------------
# results/v49_layerunfreeze_10seed/preregistration_ladder.json, committed in
# c6f41dc BEFORE the first run here. It fixes the rungs, the seeds, this command,
# the statistic, and what each outcome means -- including that a non-monotone
# forgetting profile is reported as found and that a diverged rung is counted
# rather than rescued with a lower learning rate. Do not deviate from it; if a
# deviation is forced, record it there and in the appendix's provenance paragraph.
#
# ENVIRONMENT -----------------------------------------------------------------
# Identical to the matched pair, including both forced deviations:
#   1. torch 2.14, above uni2ts's declared <2.5 pin. Below it, the
#      NegativeBinomial component of Moirai's output mixture collapses to
#      total_count=0 on MPS and training dies in the first NLL step.
#   2. PYTORCH_ENABLE_MPS_FALLBACK=1, because aten::_cummax_helper has no MPS
#      kernel and uni2ts's time-id generation falls back to CPU for that op.
#
# The unfreeze DEPTH is not stored in the run JSON -- finetune_forecasting.py does
# not record it -- so it is carried by the directory name, as the N3 records are.
# That is why these go into the v49 root: check_paper_numbers.py excludes that
# root by name (_LU_ROOTS) from its condition_D sweep, and a NEW directory would
# silently reclassify partial-unfreeze runs as frozen-encoder runs.
#
# Resumable: a seed whose JSON already exists is skipped, so this can be
# relaunched after an interruption without losing or duplicating work.
# Cost: ~45 min/run measured at n=10k, 20 epochs => ~4.5 h for all 6.

set -u

# Repo root from this script's own location, never a hard-coded absolute path: the
# path that used to be written into these runners named a home directory, which is a
# deanonymisation vector in a supplementary bundle.
cd "$(dirname "$0")/.." || exit 1

export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv12/bin/python

SEEDS=(42 101 123)
RUNGS=(4 5)
OUT_ROOT=results/v49_layerunfreeze_10seed
LOG_ROOT=logs/v49_ladder
mkdir -p "$LOG_ROOT"

# Fail fast rather than silently producing mock-Moirai numbers: the detector falls
# back to a MOCK Moirai when uni2ts is missing, and a mock run writes a JSON that
# looks like every other one.
if ! $PY -c "import src.models.moirai_detector as m; raise SystemExit(0 if m.UNI2TS_AVAILABLE else 1)" 2>/dev/null; then
  echo "FATAL: uni2ts unavailable -- would run a mock Moirai. Aborting."
  exit 1
fi
echo "uni2ts check passed"
$PY -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"

# The registration is the gate on this whole arm: refuse to run if it is not committed,
# because "registered before the run" is checkable only from the history.
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "NOTE: no repository here; skipping the registration-is-committed check."
elif ! git log --oneline -1 -- "$OUT_ROOT/preregistration_ladder.json" | grep -q .; then
  echo "FATAL: $OUT_ROOT/preregistration_ladder.json is not committed. Commit it first."
  exit 1
else
  echo "registration committed in $(git log --format=%h -1 -- "$OUT_ROOT/preregistration_ladder.json")"
fi

echo "start $(date '+%Y-%m-%d %H:%M:%S')"

for N in "${RUNGS[@]}"; do
  echo ""
  echo "=== N=$N (top $N of 6 trainable): condition D + --unfreeze-top-n-layers $N,"
  echo "    ETTh2 n=10k h=96, MPS, early-stopped, deterministic ==="
  for seed in "${SEEDS[@]}"; do
    seed_dir="$OUT_ROOT/N${N}_seed${seed}"
    log_file="$LOG_ROOT/N${N}_seed${seed}.log"
    if [ -f "$seed_dir/condition_D_h96_s${seed}.json" ]; then
      echo "[skip]  N=$N seed $seed already done"
      continue
    fi
    mkdir -p "$seed_dir"
    echo "[start] $(date '+%H:%M:%S') N=$N seed=$seed"
    $PY scripts/finetune_forecasting.py \
      --data-path data/forecasting/ETTh2.csv \
      --condition D --model-size small --horizon 96 \
      --epochs 20 --max-train-samples 10000 --eval-every 1 --device mps \
      --seed "$seed" --early-stopping --save-best-encoder --deterministic \
      --unfreeze-top-n-layers "$N" \
      --results-dir "$seed_dir" \
      > "$log_file" 2>&1
    echo "[done]  $(date '+%H:%M:%S') N=$N seed=$seed exit=$?"
  done
done

echo ""
echo "finish $(date '+%Y-%m-%d %H:%M:%S')"
# Count RECORDS, not log lines: a runner's own success line has masked a crash in
# this project before, at a cost of 16 h.
for N in "${RUNGS[@]}"; do
  echo "N$N JSONs: $(ls "$OUT_ROOT"/N${N}_seed*/condition_D_h96_s*.json 2>/dev/null | wc -l | tr -d ' ')/3"
done
