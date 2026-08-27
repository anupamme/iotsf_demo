#!/usr/bin/env bash
# Condition D (frozen encoder) for Moirai-Base / ETTh2, n=10k, h=96 -- EXTEND TO 10 SEEDS.
#
# WHY: the paper now reports Moirai-Base condition D at 3 seeds against a published
# condition B at 10 seeds. A reviewer objected to that asymmetry (and to the fact that the
# two ran on different hardware). This script closes the seed-count half of the objection by
# taking D to the same 10 seeds used for the Moirai-Small n=10k run
# (results/v19_cuda_etth2_n10k): 42 101 123 202 303 456 777 789 888 999.
#
# Seeds 42, 101, 123 are already done (results/v35_base_frozen), so the default run does the
# remaining seven. At roughly 2.1 h/seed on MPS that is about 15 h. It is RESUMABLE: any seed
# whose JSON already exists is skipped, so interrupt and re-run freely.
#
# The hardware difference remains -- the published B ran on CUDA, this D runs locally. State
# that wherever the comparison appears. The matched-hardware, matched-seed evidence is the
# n=1000 v5 pair (results/v5_etth2_base/h96), which is a different cell.
#
# Protocol matches the published condition-B run (paper_8/sections/appendix.tex):
#   Moirai-Base, ETTh2, h=96, n=10k, lr 1e-4, early stopping.
#   Published B: forgetting -54.2 +/- 8.6 %, 10/10 seeds negative, CKA .251 +/- .124.
#
# Usage:   bash scripts/run_base_frozen_D_10seed.sh [seed ...]
# Default: the seven seeds not yet run.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="$REPO/.venv-probe/bin/python"
OUT="$REPO/results/v35_base_frozen"
LOG="$OUT/run.log"
DEVICE="${DEVICE:-mps}"

ALL_SEEDS=(42 101 123 202 303 456 777 789 888 999)
if [ "$#" -gt 0 ]; then SEEDS=("$@"); else SEEDS=("${ALL_SEEDS[@]}"); fi

# ----------------------------------------------------------------- preflight
echo "=== preflight ==="

if [ ! -x "$PY" ]; then
  echo "FAIL: python venv not found at $PY"
  echo "      Recreate it with:"
  echo "        uv venv --python python3.12 .venv-probe"
  echo "        uv pip install --python .venv-probe/bin/python \\"
  echo "          'numpy>=1.26,<2.0' 'pandas>=2.1' 'scipy>=1.11,<1.12' 'scikit-learn>=1.5' \\"
  echo "          'torch>=2.4' 'einops>=0.7,<0.8' 'transformers>=4.40' 'uni2ts>=2.0.0' \\"
  echo "          'loguru>=0.7.3' pyyaml tqdm"
  echo "      then upgrade torch separately (uni2ts pins torch>=2.4, which is too old for MPS):"
  echo "        uv pip install --python .venv-probe/bin/python 'torch>=2.6'"
  exit 1
fi

# torch >= 2.6 is required on MPS: on 2.4.1 MPS silently corrupts Moirai's
# NegativeBinomial parameters (total_count -> 0) and the run dies mid-epoch.
TORCH_VER=$("$PY" - <<'EOF'
try:
    import torch
    print(torch.__version__)
except Exception as e:
    print(f"import-failed: {e}")
EOF
)
TORCH_OK=$("$PY" - <<'EOF'
try:
    import torch
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    print("ok" if (major, minor) >= (2, 6) else "old")
except Exception:
    print("old")
EOF
)
if [ "$TORCH_OK" != "ok" ] && [ "$DEVICE" = "mps" ]; then
  echo "FAIL: need torch >= 2.6 on MPS, found: $TORCH_VER"
  echo "      (torch 2.4.x corrupts Moirai's NegativeBinomial on MPS)"
  echo "      Fix: uv pip install --python .venv-probe/bin/python 'torch>=2.6'"
  echo "      Or run on CPU (much slower): DEVICE=cpu bash $0"
  exit 1
fi

if [ ! -f "data/forecasting/ETTh2.csv" ]; then
  echo "FAIL: data/forecasting/ETTh2.csv not found"; exit 1
fi

FREE_MB=$(df -m . | awk 'NR==2 {print $4}')
if [ "$FREE_MB" -lt 2000 ]; then
  echo "FAIL: only ${FREE_MB} MB free; need ~2 GB."
  echo "      'uv cache clean' typically frees a lot."
  exit 1
fi

mkdir -p "$OUT"
TODO=()
for S in "${SEEDS[@]}"; do
  [ -f "$OUT/condition_D_h96_s${S}.json" ] || TODO+=("$S")
done

echo "python  : $PY  (torch $TORCH_VER)"
echo "device  : $DEVICE"
echo "disk    : ${FREE_MB} MB free"
echo "output  : $OUT"
echo "requested: ${SEEDS[*]}"
echo "to run  : ${TODO[*]:-none, all already done}  (~2.1 h each, ~$(( ${#TODO[@]} * 2 )) h total)"
echo

# ----------------------------------------------------------------- run
START_ALL=$(date +%s)
for S in "${TODO[@]:-}"; do
  [ -n "$S" ] || continue
  RESULT="$OUT/condition_D_h96_s${S}.json"

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
sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
neg = sum(v < 0 for v in vals)
sem = sd / len(vals) ** 0.5 if len(vals) > 1 else 0.0
print(f"\ncondition D : {mean:+.2f} +/- {sd:.2f} %  (sd; SEM {sem:.2f})"
      f"   {neg}/{len(vals)} seeds negative, n={len(vals)}")
print( "condition B : -54.20 +/- 8.60 %   (10/10 negative, published, CUDA)")
print(f"B - D gap   : {-54.20 - mean:+.2f} pp")
print("\nCAVEAT to carry into the paper: B ran on CUDA, this D on local hardware.")
print("The matched-hardware matched-seed pair is the n=1000 v5 run, a different cell.")

print("\nInterpretation is pre-committed (see plan):")
if mean > -10:
    print("  D near zero -> B >> D. Encoder adaptation contributes essentially all of the")
    print("  gain; Moirai-Base becomes a second full-axis task-aligned cell alongside Moirai/ILI.")
elif mean < -40:
    print("  D close to B -> adaptation contributes little. Moirai-Base has probe asymmetry")
    print("  WITHOUT adaptation contributing: the 'aligned but redundant' quadrant. Report it")
    print("  prominently; it weakens the task-aligned reading. Do NOT re-roll seeds.")
else:
    print("  Intermediate. Report the gap with its spread and do not assign a quadrant.")
EOF
