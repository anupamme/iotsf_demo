#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Second-backbone GENERALITY experiment for the NeurIPS 2026 response phase.
#
# Chronos-T5-Small x M4-Monthly: a NON-Moirai backbone on a NON-ETT dataset
# that gate-passes (~84.5% ZS improvement over Linear). Runs the full
# diagnostic chain -- CKA, Ridge trained-head DeltaR^2, task-orthogonal probes
# (lag1/mean/var), forgetting%, frozen-encoder (D) + random-init controls.
#
# Directly addresses:
#   * Meta-review's #1 concern: dissociation/probe-asymmetry is Moirai-only.
#   * Reviewer Q1: "any non-Moirai backbone gate-pass on a NON-ETT dataset
#     with full probes -- even 5 seeds at n=500."
#
# Success criterion: CKA falls below ~0.95 AND trained-head DeltaR^2 > 0 in a
# majority of seeds WHILE task-orthogonal DeltaR^2 <= 0  == probe asymmetry
# replicated on a second architecture + dataset family.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root

SEEDS="42 123 202 303 456"       # 5 seeds; extend to "42 123 202 303 456 777 789 888 999 101" for 10
OUTDIR="results/chronos_m4_n500"
EPOCHS=20
NTRAIN=500
DEVICE="cuda"                    # set to "cpu" only for a smoke test

for s in $SEEDS; do
  echo "=== seed $s : condition B (full fine-tune) ==="
  python scripts/finetune_chronos_m4.py \
    --condition B --seed "$s" --epochs "$EPOCHS" \
    --max-train-samples "$NTRAIN" --device "$DEVICE" --deterministic \
    --results-dir "$OUTDIR"

  echo "=== seed $s : condition D (frozen-encoder control) ==="
  python scripts/finetune_chronos_m4.py \
    --condition D --seed "$s" --epochs "$EPOCHS" \
    --max-train-samples "$NTRAIN" --device "$DEVICE" --deterministic \
    --results-dir "$OUTDIR"
done

echo "=== random-init negative control (1 seed) ==="
python scripts/finetune_chronos_m4.py \
  --condition B --random-init --seed 42 --epochs "$EPOCHS" \
  --max-train-samples "$NTRAIN" --device "$DEVICE" --deterministic \
  --results-dir "${OUTDIR}_randinit"

# ---------------------------------------------------------------------------
# Aggregate the asymmetry signal across seeds (trained DeltaR^2 vs orthogonal).
# ---------------------------------------------------------------------------
echo
echo "=== aggregate (condition B) ==="
python - "$OUTDIR" <<'PY'
import json, sys, glob, statistics as st
root = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"{root}/seed*/condition_B_s*.json")):
    d = json.load(open(f))
    rows.append((d["seed"], d["gate_improvement_pct"], d["final_cka"],
                 d["forgetting_pct"], d["linear_probe"]["r2_delta"],
                 d["orthogonal_probes"]["delta"]))
if not rows:
    print("no condition_B results yet"); sys.exit()
tr = [r[4] for r in rows]
print(f"seeds={len(rows)}  gate%={rows[0][1]:.1f}")
print(f"CKA         mean={st.mean(r[2] for r in rows):.3f}")
print(f"forgetting% mean={st.mean(r[3] for r in rows):+.1f}")
print(f"trained DeltaR^2 mean={st.mean(tr):+.3f}  positive={sum(x>0 for x in tr)}/{len(tr)}")
for name in ("lag1","mean","var"):
    vals=[r[5][name] for r in rows]
    print(f"orthogonal DeltaR^2[{name}] mean={st.mean(vals):+.3f}  <=0: {sum(v<=0 for v in vals)}/{len(vals)}")
PY

echo
echo "Done. Per-seed JSON: ${OUTDIR}/seed*/condition_{B,D}_s*.json"
echo "Report in rebuttal: gate%, final_cka, forgetting_pct, linear_probe.r2_delta (trained),"
echo "orthogonal_probes.delta (should be <=0), early_stopping.final_epoch_forgetting_pct."
