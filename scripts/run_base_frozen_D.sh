#!/usr/bin/env bash
# Condition D (frozen encoder) for Moirai-Base / ETTh2, n=10k, h=96.
#
# WHY: the workshop paper's Moirai-Base cell has probe asymmetry (Axis A, 9/10 seeds)
# but no frozen-encoder control (Axis B), so it cannot be assigned a signature and the
# task-aligned result currently rests on the single Moirai/ILI cell. This run supplies
# the missing Axis B.
#
# Protocol matches the published condition-B run (sections/appendix.tex:296):
#   Moirai-Base, ETTh2, h=96, n=10k, lr 1e-4, early stopping.
#   Published B: forgetting -54.2 +/- 8.6 %, 10/10 seeds negative.
#
# RESUMABLE: seeds whose JSON already exists are skipped, so it is safe to
# re-run after an interruption. Roughly 2.1 h/seed on MPS.
#
# Usage:   bash scripts/run_base_frozen_D.sh [seed ...]
# Default: seeds 42 101 123

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="$REPO/.venv-probe/bin/python"
OUT="$REPO/results/v35_base_frozen"
LOG="$OUT/run.log"
DEVICE="${DEVICE:-mps}"
SEEDS=("${@:-}")
if [ -z "${SEEDS[0]:-}" ]; then SEEDS=(42 101 123); fi

# ----------------------------------------------------------------- preflight
echo "=== preflight ==="

if [ ! -x "$PY" ]; then
  echo "FAIL: python venv not found at $PY"
  echo "      Recreate it with:"
  echo "        uv venv --python python3.12 .venv-probe"
  echo "        uv pip install --python .venv-probe/bin/python \\"
  echo "          'numpy>=1.26,<2.0' 'pandas>=2.1' 'scipy>=1.11,<1.12' 'scikit-learn>=1.5' \\"
  echo "          'torch>=2.6' 'einops>=0.7,<0.8' 'transformers>=4.57' 'uni2ts>=2.0.0' \\"
  echo "          'loguru>=0.7.3' pyyaml tqdm"
  exit 1
fi

# torch >= 2.6 is required: on 2.4.1 MPS silently corrupts Moirai's
# NegativeBinomial parameters (total_count -> 0) and the run dies mid-epoch.
TORCH_OK=$("$PY" - <<'EOF'
try:
    import torch
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    print("ok" if (major, minor) >= (2, 6) else torch.__version__)
except Exception as e:
    print(f"import-failed: {e}")
EOF
)
if [ "$TORCH_OK" != "ok" ]; then
  echo "FAIL: need torch >= 2.6 on MPS, found: $TORCH_OK"
  echo "      (torch 2.4.x corrupts Moirai's NegativeBinomial on MPS)"
  echo "      Fix: uv pip install --python .venv-probe/bin/python 'torch>=2.6'"
  exit 1
fi

if [ ! -f "data/forecasting/ETTh2.csv" ]; then
  echo "FAIL: data/forecasting/ETTh2.csv not found"; exit 1
fi

# Each seed writes a small JSON, but HF caching and temp files need headroom.
FREE_MB=$(df -m . | awk 'NR==2 {print $4}')
if [ "$FREE_MB" -lt 2000 ]; then
  echo "FAIL: only ${FREE_MB} MB free; need ~2 GB."
  echo "      'uv cache clean' typically frees a lot."
  exit 1
fi

mkdir -p "$OUT"
echo "python : $PY"
echo "torch  : ok (>=2.6)"
echo "device : $DEVICE"
echo "disk   : ${FREE_MB} MB free"
echo "output : $OUT"
echo "seeds  : ${SEEDS[*]}"
echo

# ----------------------------------------------------------------- run
START_ALL=$(date +%s)
for S in "${SEEDS[@]}"; do
  RESULT="$OUT/condition_D_h96_s${S}.json"
  if [ -f "$RESULT" ]; then
    echo "seed $S: already done, skipping ($RESULT)"
    continue
  fi

  echo "seed $S: starting $(date '+%H:%M:%S')  (~2.1 h expected)"
  START=$(date +%s)

  PYTORCH_ENABLE_MPS_FALLBACK=1 "$PY" scripts/finetune_forecasting.py \
      --condition D \
      --model-size base \
      --data-path data/forecasting/ETTh2.csv \
      --horizon 96 \
      --max-train-samples 10000 \
      --lr 1e-4 \
      --early-stopping \
      --device "$DEVICE" \
      --seed "$S" \
      --results-dir "$OUT" >> "$LOG" 2>&1
  RC=$?

  ELAPSED=$(( ($(date +%s) - START) / 60 ))
  if [ $RC -ne 0 ] || [ ! -f "$RESULT" ]; then
    echo "seed $S: FAILED after ${ELAPSED} min (exit $RC). Last log lines:"
    tail -15 "$LOG" | sed 's/^/    /'
    echo "  Re-running this script will resume from here."
    exit 1
  fi
  echo "seed $S: done in ${ELAPSED} min"
done
echo
echo "total: $(( ($(date +%s) - START_ALL) / 60 )) min"

# ----------------------------------------------------------------- summary
echo
echo "=== condition D results (Moirai-Base / ETTh2, h=96, n=10k) ==="
"$PY" - "$OUT" <<'EOF'
import glob, json, statistics, sys, os
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, "condition_D_h96_s*.json"))):
    d = json.load(open(f))
    rows.append((d["seed"], d["forgetting_pct"], d["final_cka"],
                 d["zeroshot_mse"], d["final_val_mse"]))
if not rows:
    print("no results found"); sys.exit(0)

print(f"{'seed':>6}{'forgetting%':>14}{'final CKA':>12}{'ZS MSE':>10}{'val MSE':>10}")
for s, f_, c, z, v in rows:
    print(f"{s:>6}{f_:>14.2f}{c:>12.4f}{z:>10.4f}{v:>10.4f}")

vals = [r[1] for r in rows]
mean = statistics.mean(vals)
sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
neg = sum(v < 0 for v in vals)
print(f"\ncondition D : {mean:+.2f} +/- {sd:.2f} %   ({neg}/{len(vals)} seeds negative, n={len(vals)})")
print( "condition B : -54.20 +/- 8.60 %   (10/10 negative, published)")
print(f"B - D gap   : {-54.20 - mean:+.2f} pp")

print("\nInterpretation is pre-committed (see plan):")
if mean > -10:
    print("  D is near zero -> B >> D. Encoder adaptation contributes essentially")
    print("  all of the gain. Moirai-Base becomes a second full-axis cell alongside")
    print("  Moirai/ILI, and the 'rests on that cell' limitation can be removed.")
elif mean < -40:
    print("  D is close to B -> adaptation contributes little. Moirai-Base then has")
    print("  probe asymmetry WITHOUT adaptation contributing, populating the")
    print("  previously empty quadrant. Report this prominently; it weakens the")
    print("  task-aligned reading of the flagship cell. Do NOT re-roll seeds.")
else:
    print("  Intermediate. Report the gap with its spread and do not assign a quadrant.")
print("\nNOTE: this D uses fewer seeds than the published 10-seed B; state the")
print("      actual seed count wherever the comparison appears.")
EOF
