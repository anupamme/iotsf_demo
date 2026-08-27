#!/usr/bin/env bash
# Chronos-T5-Small / ETTh2, MSE-loss fine-tuning, conditions B and D, 3 seeds.
#
# WHY: the probe suite has only ever been run on Moirai cells, so a reviewer cannot tell
# whether probe-sign asymmetry (Axis A) is a property of fine-tuning or a property of Moirai.
# This run applies both axes to a non-Moirai gate-pass cell. It also retains the encoder
# checkpoints, which the paper's existing Chronos probe numbers do not have
# (results/chronos_mse/ is empty) -- afterwards, scripts/probe_transparent.py can be pointed
# at these encoders so the SAME transparent protocol covers both backbones.
#
# PRE-COMMITMENT: prior Chronos/ETTh2 runs show drift WITHOUT probe asymmetry, i.e. the
# "drift matters, unaligned" quadrant. This run is expected to reproduce that, not to
# manufacture a second task-aligned cell. Report whatever it gives.
#
# Protocol note: the Chronos pipeline uses lookback 96 / horizon 24 (DATASETS in
# scripts/chronos_mse_finetune.py), NOT the h=96 used for Moirai. It is a different cell,
# not a replication of the Moirai numbers. n=8000, 30 epochs, patience 7, lr 1e-4.
#
# IMPORTANT: conditions B and D are written to SEPARATE --results-dir trees, because
# chronos_mse_finetune.py:551 always saves best_encoder.pt to <results-dir>/mse_<ds>/seed<S>/
# and would otherwise have D overwrite B's encoder.
#
# RESUMABLE: any (condition, seed) whose JSON exists is skipped. Roughly 20-40 min per cell
# on MPS, so about 2-4 h for all six.
#
# Usage:   bash scripts/run_chronos_etth2_probe.sh [seed ...]
# Default: seeds 42 43 44

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="$REPO/.venv-probe/bin/python"
ROOT="$REPO/results/v37_chronos_etth2"
LOG="$ROOT/run.log"
DEVICE="${DEVICE:-mps}"
DS="ETTh2"

if [ "$#" -gt 0 ]; then SEEDS=("$@"); else SEEDS=(42 43 44); fi

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
  echo "        uv pip install --python .venv-probe/bin/python 'torch>=2.6'"
  exit 1
fi

# transformers silently disables its PyTorch backend below torch 2.5 -- the T5 model then
# cannot be loaded at all. torch >= 2.6 is also what MPS needs elsewhere in this project.
TORCH_VER=$("$PY" -c "import torch; print(torch.__version__)" 2>&1 | tail -1)
TORCH_OK=$("$PY" - <<'EOF'
try:
    import torch
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    print("ok" if (major, minor) >= (2, 6) else "old")
except Exception:
    print("old")
EOF
)
if [ "$TORCH_OK" != "ok" ]; then
  echo "FAIL: need torch >= 2.6, found: $TORCH_VER"
  echo "      Below 2.5 transformers disables its PyTorch backend and T5 will not load."
  echo "      Fix: uv pip install --python .venv-probe/bin/python 'torch>=2.6'"
  exit 1
fi

T5_OK=$("$PY" - <<'EOF'
try:
    from transformers import T5ForConditionalGeneration  # noqa: F401
    print("ok")
except Exception as e:
    print(f"import-failed: {type(e).__name__}: {e}")
EOF
)
if [ "$T5_OK" != "ok" ]; then
  echo "FAIL: transformers cannot provide T5 with a torch backend: $T5_OK"
  exit 1
fi

FREE_MB=$(df -m . | awk 'NR==2 {print $4}')
if [ "$FREE_MB" -lt 3000 ]; then
  echo "FAIL: only ${FREE_MB} MB free; need ~3 GB (six encoder checkpoints + HF cache)."
  echo "      'uv cache clean' typically frees a lot."
  exit 1
fi

mkdir -p "$ROOT"
echo "python : $PY  (torch $TORCH_VER)"
echo "device : $DEVICE"
echo "disk   : ${FREE_MB} MB free"
echo "output : $ROOT/cond_{B,D}/mse_etth2/seed<S>/"
echo "seeds  : ${SEEDS[*]}   conditions: B D"
echo

# ----------------------------------------------------------------- run
START_ALL=$(date +%s)
for C in B D; do
  OUT="$ROOT/cond_$C"
  for S in "${SEEDS[@]}"; do
    RESULT="$OUT/mse_etth2/seed${S}/condition_${C}_s${S}.json"
    if [ -f "$RESULT" ]; then
      echo "cond $C seed $S: already done, skipping"
      continue
    fi

    echo "cond $C seed $S: starting $(date '+%H:%M:%S')  (~20-40 min expected)"
    START=$(date +%s)

    PYTORCH_ENABLE_MPS_FALLBACK=1 "$PY" scripts/chronos_mse_finetune.py \
        --dataset "$DS" \
        --condition "$C" \
        --seed "$S" \
        --device "$DEVICE" \
        --results-dir "$OUT" >> "$LOG" 2>&1
    RC=$?

    ELAPSED=$(( ($(date +%s) - START) / 60 ))
    if [ $RC -ne 0 ] || [ ! -f "$RESULT" ]; then
      echo "cond $C seed $S: FAILED after ${ELAPSED} min (exit $RC). Last log lines:"
      tail -15 "$LOG" | sed 's/^/    /'
      echo "  Re-running this script will resume from here."
      exit 1
    fi
    echo "cond $C seed $S: done in ${ELAPSED} min"
  done
done
echo
echo "total: $(( ($(date +%s) - START_ALL) / 60 )) min"

# ----------------------------------------------------------------- summary
echo
echo "=== Chronos-T5 / ETTh2, MSE loss, h=24 ==="
"$PY" - "$ROOT" <<'EOF'
import glob, json, statistics, sys, os
root = sys.argv[1]
by_cond = {}
for c in ("B", "D"):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, f"cond_{c}", "mse_etth2", "seed*",
                                           f"condition_{c}_s*.json"))):
        d = json.load(open(f))
        rows.append(d)
    by_cond[c] = rows

def agg(rows, key):
    v = [r[key] for r in rows if key in r and r[key] is not None]
    if not v:
        return None, None, 0
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    return statistics.mean(v), sd, len(v)

for c in ("B", "D"):
    rows = by_cond[c]
    if not rows:
        print(f"condition {c}: no results"); continue
    print(f"\ncondition {c}  (n={len(rows)} seeds)")
    print(f"{'seed':>6}{'forgetting%':>14}{'final CKA':>12}{'dR2 trained':>14}{'asym':>7}")
    for r in rows:
        print(f"{r['seed']:>6}{r.get('forgetting_pct', float('nan')):>14.2f}"
              f"{r.get('final_cka', float('nan')):>12.4f}"
              f"{r['linear_probe']['delta_r2']:>+14.4f}"
              f"{('yes' if r.get('probe_asymmetry') else 'no'):>7}")
    m, s, n = agg(rows, "forgetting_pct")
    mc, sc, _ = agg(rows, "final_cka")
    print(f"  forgetting {m:+.2f} +/- {s:.2f} %   CKA {mc:.4f} +/- {sc:.4f}")

mb, sb, nb = agg(by_cond["B"], "forgetting_pct")
md, sd_, nd = agg(by_cond["D"], "forgetting_pct")
if mb is not None and md is not None:
    print(f"\nAXIS B: B {mb:+.2f} (n={nb}) vs D {md:+.2f} (n={nd})  ->  B-D = {mb - md:+.2f} pp")
    print("  matched seeds, matched hardware, same script: this is a paired comparison.")

asym = [bool(r.get("probe_asymmetry")) for r in by_cond["B"]]
if asym:
    print(f"AXIS A: probe asymmetry in {sum(asym)}/{len(asym)} condition-B seeds")

print("\nAxis A here uses chronos_mse_finetune.py's built-in orthogonal probes")
print("(Ridge alpha=1.0 on context-window scalars, :132). For the transparent protocol,")
print("point scripts/probe_transparent.py at:")
print(f"  {root}/cond_B/mse_etth2/seed*/best_encoder.pt")
EOF
