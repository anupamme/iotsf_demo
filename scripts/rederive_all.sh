#!/bin/bash
# Re-derive every table and figure the paper \input{}s, from the committed run records, and report
# any that changed.
#
# WHAT THIS IS FOR. The paper's reproducibility claim has to be exactly as strong as what this script
# can do and no stronger. So the script is organised by what each artifact NEEDS, and it says so:
#
#   TIER A  re-derivable from a clean clone alone. Reads only results/*.json, which are committed.
#           This is every table the body and appendices \input{}, and it is the claim the
#           Reproducibility Statement makes.
#   TIER B  needs the benchmark CSVs, which are gitignored (*.csv) and are not ours to redistribute.
#           scripts/data_manifest.py --check verifies that the CSVs you fetched are byte-identical to
#           the ones we used before anything in this tier runs; a hash mismatch is a hard stop,
#           because a differently-parsed ETTh1 changes every window and hence every MSE while
#           producing plausible-looking output.
#   TIER C  needs fine-tuned CHECKPOINTS, which are gitignored (*.pt, *.safetensors, reps_cache.npz)
#           and are too large to commit. NOT re-derivable here at any effort, and the script says so
#           rather than skipping it silently. The affected artifact is the drift-metric battery's
#           --compute pass; its stored output feeds the TIER A --report pass, so the TABLE is
#           re-derivable even though the geometry behind it is not.
#
# WHAT A NONZERO EXIT MEANS. Every emitter writes a tracked file, so `git diff` after the sweep is a
# complete staleness check: any diff is a number that was wrong in the PDF. It is NOT necessarily a
# reproduction failure -- it is equally often a run record added since the last regeneration. The
# script prints the diff and leaves the working tree dirty on purpose, so the difference can be read.
#
# WHAT IT DOES NOT CLAIM. This is re-derivation plus, separately, a two-cell from-scratch check
# (scripts/rerun_two_cells.sh). It is not independent third-party reproduction: no third party has
# run this. The Reproducibility Statement says that in those words.
#
# Usage:  bash scripts/rederive_all.sh              # TIER A only (clean-clone reproducible)
#         bash scripts/rederive_all.sh --with-data  # TIER A + TIER B (needs the CSVs)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

PY=${PY:-.venv12/bin/python}          # analysis: numpy/scipy/sklearn, no torch, no matplotlib
PYFIG=${PYFIG:-/opt/homebrew/bin/python3}   # figures: matplotlib (neither venv has it)
WITH_DATA=0
[ "${1:-}" = "--with-data" ] && WITH_DATA=1

fail=0
run() {  # run <label> <command...>
  local label=$1; shift
  printf '  %-34s' "$label"
  if out=$("$@" 2>&1); then
    echo "ok"
  else
    echo "FAILED"
    echo "$out" | tail -15 | sed 's/^/      /'
    fail=$((fail + 1))
  fi
}

echo "=============================================================================="
echo "RE-DERIVING EVERY \\input-ed TABLE AND FIGURE FROM THE COMMITTED RUN RECORDS"
echo "=============================================================================="
[ -x "$PY" ] || { echo "no $PY -- create it: python3.12 -m venv .venv12 && "\
"$PWD/.venv12/bin/pip install 'numpy<2' scipy scikit-learn pandas"; exit 2; }
echo "  analysis interpreter  $PY"
echo "  figure interpreter    $PYFIG"
git -C "$ROOT" rev-parse --short HEAD | sed 's/^/  HEAD                  /'

echo
echo "-- TIER A: tables, from results/*.json only ----------------------------------"
run "heldout_all + strictfreeze"   "$PY" scripts/cell_matrix.py --latex
run "r2task"                       "$PY" scripts/emit_r2task.py
run "degradation_sensitivity"      "$PY" scripts/degradation_sensitivity.py
run "clustered_* (4 tables)"       "$PY" scripts/clustered_inference.py --latex
run "paired_inference"             "$PY" scripts/paired_inference.py
run "heldout_decomposition"        "$PY" scripts/heldout_decomposition.py
run "crossbackbone"                "$PY" scripts/emit_crossbackbone.py
run "mitigation_spectrum"          "$PY" scripts/emit_mitigation_spectrum.py
run "gate_sensitivity"             "$PY" scripts/gate_threshold_sensitivity.py
run "prospective"                  "$PY" scripts/score_prospective.py --latex
run "drift_metrics (--report)"     "$PY" scripts/drift_metric_battery.py --report
run "gate_baseline_family"         "$PY" scripts/gate_baseline_sensitivity.py --from-json --latex
run "value_axis (+ interaction)"   "$PY" scripts/value_axis.py --json --latex
run "sample_sweep"                 "$PY" scripts/emit_sample_sweep.py

echo
echo "-- TIER A: figures (need matplotlib) ----------------------------------------"
# Matplotlib stamps a CreationDate into every PDF it writes, so without this the four figures show up
# as "changed" on every single sweep and the staleness check below becomes noise -- which trains you
# to ignore the one output that is supposed to be load-bearing. SOURCE_DATE_EPOCH pins that timestamp
# and makes the figures byte-reproducible; the value is arbitrary but must not change.
export SOURCE_DATE_EPOCH=1600000000
if [ -x "$PYFIG" ] && "$PYFIG" -c 'import matplotlib' 2>/dev/null; then
  # Both figure scripts assert their own counts at draw time, so they double as integrity checks:
  # a figure that disagrees with the tables fails here instead of shipping.
  run "fig1_diagnostic_flow"       "$PYFIG" paper_8/fig1_diagnostic_flow.py
  run "fig_value_axis"             "$PYFIG" paper_8/fig_value_axis.py
  # The other two \includegraphics'd figures. Both were missing from this sweep, and
  # analyse_n5k_trajectories.py had been dead since July -- it carried an absolute path to a
  # repository location that no longer exists, so nobody would have noticed the figure going stale.
  run "dissociation_trajectory"    "$PYFIG" scripts/plot_dissociation_trajectory.py
  run "n5k_trajectories"           "$PYFIG" scripts/analyse_n5k_trajectories.py
else
  echo "  SKIPPED: $PYFIG has no matplotlib. The figures are NOT re-derived."
  fail=$((fail + 1))
fi

if [ "$WITH_DATA" = 1 ]; then
  echo
  echo "-- TIER B: needs the benchmark CSVs ----------------------------------------"
  if "$PY" scripts/data_manifest.py --check > /tmp/rederive_manifest.txt 2>&1; then
    echo "  benchmark CSVs match results/data_manifest.json"
    # The full ladder recomputes all nine denominators from the raw windows. Hours, not minutes:
    # the MLP and GBM rungs are fitted per cell per feature. --from-json above is the cheap path and
    # reproduces the TABLE; this is the path that reproduces the NUMBERS in it.
    run "gate ladder (recompute)"  "$PY" scripts/gate_baseline_sensitivity.py --latex
  else
    echo "  STOPPED: the benchmark CSVs do not match the manifest."
    sed 's/^/    /' /tmp/rederive_manifest.txt
    echo "    A differently-parsed CSV changes every window and every MSE, so TIER B is not run."
    fail=$((fail + 1))
  fi
else
  echo
  echo "-- TIER B: skipped (pass --with-data to run it) -----------------------------"
fi

echo
echo "-- TIER C: NOT re-derivable here --------------------------------------------"
echo "  drift_metric_battery --compute   needs the 59 fine-tuned encoders (*.pt, gitignored)."
echo "  any condition B/D/E/H run        needs a GPU/MPS run; see scripts/rerun_two_cells.sh for"
echo "                                   the bounded two-cell from-scratch check instead."

echo
echo "-- TIER A: the PROSE's numbers ------------------------------------------------"
# The staleness check below covers every \input-ed table, and until now nothing covered the body
# text. That is where the drift has actually lived: the prose has asserted "twelve of the fifteen"
# after the records moved to sixteen, and quoted a pre-correction count for a whole arm. This
# re-derives each registered claim from results/*.json and fails on any disagreement, so the
# "any diff is a number that was stale in the PDF" contract now covers sentences as well as tables.
# Captured rather than piped: in a pipeline the exit status is the last command's, so
# `checker | sed` would report sed's success and swallow every failure.
if prose=$("$PY" "$ROOT/scripts/check_paper_numbers.py" 2>&1); then
  echo "$prose" | tail -n 2 | sed 's/^/  /'
else
  echo "$prose" | sed 's/^/  /'
  echo "  PROSE NUMBERS DISAGREE WITH THE RUN RECORDS (see above)."
  fail=$((fail + 1))
fi

echo
echo "-- STALENESS: what moved ----------------------------------------------------"
changed=$(git -C "$ROOT" status --porcelain paper_8/tables/ paper_8/fig1_diagnostic_flow.pdf \
            paper_8/fig_value_axis.pdf paper_8/figures/dissociation_trajectory.pdf \
            paper_8/figures/n5k_trajectories.pdf | sed 's/^/    /')
if [ -z "$changed" ]; then
  echo "  nothing changed: every \\input-ed table and figure in the PDF matches the run records."
else
  echo "  THESE ARTIFACTS CHANGED -- each one is a number that was stale in the PDF, or a run"
  echo "  record added since the last regeneration. Read the diff; do not just commit it."
  echo "$changed"
  git -C "$ROOT" --no-pager diff --stat paper_8/tables/ | sed 's/^/    /'
  fail=$((fail + 1))
fi

echo
if [ "$fail" = 0 ]; then
  echo "RE-DERIVATION CLEAN (tier A$([ "$WITH_DATA" = 1 ] && echo '+B'))."
else
  echo "RE-DERIVATION INCOMPLETE OR DIRTY: $fail item(s) need attention (see above)."
fi
exit "$fail"
