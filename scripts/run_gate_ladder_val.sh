#!/usr/bin/env bash
# The eight-rung ladder on the SELECTION windows, which is the gate the round-3 review asks for:
# disjoint from the windows the B-D outcome is scored on. CPU-only -- every numerator is already
# stored, so no arm loads a model. The boosted-tree rung is the slow one.
#
# No pipes anywhere in the run line: a `| tee` or `| grep` would report the exit status of the
# right-hand side and a crashed sweep would log as a success.
set -u
cd "$(dirname "$0")/.."
LOG=logs/gate_ladder_val.log
: > "$LOG"

# Preflight: the ladder reads the benchmark CSVs, so verify them before spending the compute.
if ! .venv12/bin/python scripts/data_manifest.py --check >> "$LOG" 2>&1; then
  echo "PREFLIGHT FAILED: benchmark CSVs do not match the manifest" >> "$LOG"
  exit 1
fi

.venv12/bin/python scripts/gate_baseline_sensitivity.py --split val >> "$LOG" 2>&1
rc=$?
echo "=== EXIT $rc ===" >> "$LOG"
if [ $rc -eq 0 ]; then
  .venv12/bin/python - >> "$LOG" 2>&1 <<'PY'
import json
d = json.load(open("results/gate_baselines_val.json"))
g = d["graded"]
print(f"BANNER cells={len(d['cells'])} baselines={len(d['baselines'])} "
      f"selfcheck={d['selfcheck_passed']} "
      f"clears_all={g['n_clearing_all_admissible']} clears_any={g['n_clearing_any_admissible']} "
      f"of {g['n_scored']}")
PY
fi
exit $rc
