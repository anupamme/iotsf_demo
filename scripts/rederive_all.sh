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
#   TIER D  needs NETWORK, so it is opt-in and never part of the clean-clone claim: the bibliography
#           round-trip (scripts/verify_bib.py) resolves every cited entry by DOI or arXiv ID and
#           re-fetches it by that identifier. It is a pre-submission gate, not a re-derivation --
#           an offline clone cannot run it, and wiring it into the default sweep would make a
#           flaky network read as a dirty paper, which is how a checker stops being read.
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
#         bash scripts/rederive_all.sh --with-bib   # TIER A + TIER D (needs the network)
#         the two flags compose, in any order.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD

PY=${PY:-.venv12/bin/python}          # analysis: numpy/scipy/sklearn, no torch, no matplotlib
PYFIG=${PYFIG:-/opt/homebrew/bin/python3}   # figures: matplotlib (neither venv has it)
WITH_DATA=0
WITH_BIB=0
# A loop, not `[ "$1" = ... ]`: with a positional test the second flag is silently ignored, so
# `--with-data --with-bib` would run TIER B and report clean without ever touching the bibliography.
for arg in "$@"; do
  case "$arg" in
    --with-data) WITH_DATA=1 ;;
    --with-bib)  WITH_BIB=1 ;;
    *) echo "unknown flag: $arg (expected --with-data and/or --with-bib)"; exit 2 ;;
  esac
done

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
# --latex was missing here until 21 Sep 2026, so this line ran the ladder, printed "ok", and never
# rewrote tables/degradation_sensitivity.tex. The table therefore predated the baseline correction by
# ten rounds: it said 6/5 gate-passing cells against the corrected 7/7, and 1 admitted cell at
# gate 0.20 where the corrected ladder admits 2. An emitter invoked without its emit flag is a
# staleness hole the sweep reports as clean.
run "degradation_sensitivity"      "$PY" scripts/degradation_sensitivity.py --latex
run "clustered_* (4 tables)"       "$PY" scripts/clustered_inference.py --latex
run "paired_inference (+power_mde)"  "$PY" scripts/paired_inference.py
# Phase F3's table, added 25 Sep 2026. Deliberately invoked WITHOUT --allow-incomplete: that flag
# reports the top-up's state and does NOT write tables/power_topup.tex, so passing it here would
# reintroduce exactly the defect the two comments in this file already record -- an emitter that
# prints "ok" while its table rots. Without the flag the script exits non-zero if any of the 26
# registered runs is missing, which makes this line the mechanical proof that the pre-registered
# top-up finished rather than an author's assurance that it did. It is therefore EXPECTED to fail
# while the top-up is still running; that failure is the signal, not a bug in this file.
run "power_topup"                  "$PY" scripts/emit_power_topup.py
# The gate's sampling-noise floor (app:zsnoise), added 25 Sep 2026. Writes results/gate_zs_noise.json
# and no table, so the staleness block below cannot catch it drifting -- what catches it instead are
# its own two asserts (its grouping must reproduce gate_all_cells._zs_val_refs() exactly, and no
# cell's 0.20 call may flip under any replicate), which make this line fail rather than print a
# quietly wrong noise floor. It must run BEFORE check_paper_numbers.py at the foot of this file,
# which reads that JSON for nine of its registered claims.
run "gate_zs_noise"                "$PY" scripts/gate_zs_noise.py --quiet
run "heldout_decomposition"        "$PY" scripts/heldout_decomposition.py
run "crossbackbone"                "$PY" scripts/emit_crossbackbone.py
run "mitigation_spectrum"          "$PY" scripts/emit_mitigation_spectrum.py
# --latex was MISSING here until 21 Sep 2026, so tables/gate_sensitivity.tex was never
# regenerated by this sweep and the staleness check below could not have caught it if it had
# drifted. Second occurrence of this defect in this file; check the flags, not just the script.
run "gate_sensitivity"             "$PY" scripts/gate_threshold_sensitivity.py --latex
run "prospective"                  "$PY" scripts/score_prospective.py --latex
run "drift_metrics (--report)"     "$PY" scripts/drift_metric_battery.py --report
# Both splits of the ladder. The SELECTION-split one is the paper's primary -- it is the split every
# admission decision is made on, and the value axis below reads it -- and the held-out one is the
# retrospective variant the prospective predictor was frozen against. Emitting only one would leave
# the other's table drifting from its own JSON with nothing to catch it.
run "gate_baseline_family"         "$PY" scripts/gate_baseline_sensitivity.py --from-json --latex
run "gate_baseline_family (val)"   "$PY" scripts/gate_baseline_sensitivity.py --split val --from-json --latex
# The matched-lookback column, added 21 Sep 2026: the same fitted ridge given 96+h inputs instead of
# 96, so the baseline and Moirai see identical history. --from-json for the same reason the ladder uses
# it -- the denominators need the CSVs (TIER B), the TABLE must be re-derivable from a clean clone. Both
# splits, because the direction of the effect differs between them and reporting only one would pick
# whichever is convenient.
run "matched_lookback (val)"       "$PY" scripts/matched_lookback_gate.py --split val --from-json --latex
run "matched_lookback (test)"      "$PY" scripts/matched_lookback_gate.py --split test --from-json --latex
# The three-window-set comparison. Reads all three ladder JSONs and counts them over the cells they
# share, which is the only comparison that is about windows rather than about coverage.
run "gate_splitcompare (3 splits)"  "$PY" scripts/emit_traintail_ladder.py --latex
run "value_axis (+ interaction)"   "$PY" scripts/value_axis.py --json --latex
# The pre-registered nested ladder under leave-one-cluster-out. --latex pins TEX_BOOT/TEX_SEED, so the
# bootstrap numbers in the caption are byte-reproducible from this one line; the design it fits is
# imported from scripts/preregister_loco.py, which was committed before the models were run.
run "cka_loco"                     "$PY" scripts/cka_fixed_effects.py --loco --json --latex
run "sample_sweep"                 "$PY" scripts/emit_sample_sweep.py

# The pre-registered positive control. TIER A even though the arm itself is TIER C: the checkpoints it
# was scored from are gitignored, but the per-run JSONs and the retention JSONs are committed, and this
# script reads only those. It takes no flags at all -- deliberately, because the failure this repo has
# already had is an emitter invoked WITHOUT its emit flag, printing "ok" for ten rounds while its table
# rotted. A script with one behaviour cannot be invoked in the wrong mode. It prints PROVISIONAL and
# writes a visible in-PDF marker whenever a registered cell is still missing.
run "positive_control"             "$PY" scripts/emit_positive_control.py

# Chronos-T5-Small on M4-Monthly: the gate ladder and the six B/D runs. TIER A -- it reads only
# results/chronos_m4/*.json, all committed. This arm exists because the appendix section it feeds used
# to be 140 lines of hand-typed numbers with no run record anywhere under results/, which is the exact
# failure this whole file is built to prevent; the emitter takes no flags for the same reason
# emit_positive_control.py takes none, and it asserts the two facts the appendix prose asserts (that
# the superseded trend denominator is inadmissible on the selection split, and that condition D's
# CKA of exactly 1 is a consequence of the freeze rather than a measurement).
run "chronos_m4 (gate + runs)"     "$PY" scripts/emit_chronos_m4.py

# results/MANIFEST.md: which of the 139 directories under results/ the paper actually stands on.
# --check, not a bare run, and the distinction matters: a bare run RE-TRACES every emitter in this file
# under an audit hook, which takes the best part of an hour because two of the lines below refit the
# whole baseline ladder from the CSVs. --check re-labels from the stored trace
# (results/manifest_trace.json) and fails if MANIFEST.md no longer matches the directory tree, which is
# the staleness this sweep is for. Regenerate the trace itself by running the script with no flags,
# which is required whenever a `run` line here is added, removed or given different flags.
run "results_manifest (--check)"   "$PY" scripts/emit_results_manifest.py --check

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
  # The two appendix figures added 20 Sep 2026. figA_gate_scatter is the axis Figure 1's panel (b)
  # gave up when it went back to CKA; figA_freeze_boundary transcribes finetune_forecasting.py's
  # requires_grad blocks and asserts condition H's CKA is exactly 1 on every cell it ran on.
  run "figA_gate_scatter"          "$PYFIG" paper_8/figA_gate_scatter.py
  run "figA_freeze_boundary"       "$PYFIG" paper_8/figA_freeze_boundary.py
  # Added 21 Sep 2026. Reads the two window builders out of finetune_forecasting.py by AST rather than
  # by import -- importing that module needs torch, loguru and the dataset loaders, none of which a
  # clean clone has -- and asserts target-start = input-end + 1 before drawing. A rename, a changed
  # slice, or pushing either builder back inside main() fails this line instead of shipping a diagram
  # that no longer describes the code.
  run "figA_window_layout"         "$PYFIG" paper_8/figA_window_layout.py
  # The other two \includegraphics'd figures. Both were missing from this sweep, and
  # analyse_n5k_trajectories.py had been dead since July -- it carried an absolute path to a
  # repository location that no longer exists, so nobody would have noticed the figure going stale.
  run "dissociation_trajectory"    "$PYFIG" scripts/plot_dissociation_trajectory.py
  run "n5k_trajectories"           "$PYFIG" scripts/analyse_n5k_trajectories.py
  # The WORKSHOP version's only figure (workshop/main.tex:293 \includegraphics's it). Not a main-paper
  # artifact, but it was \includegraphics'd by a tracked document with no emitter line, which is the
  # same staleness hole as any other -- the workshop build would have kept shipping an old figure
  # silently. paper_8/fig2_drift_utility.py writes a figure NO document includes, so it is deliberately
  # not here: an emitter line for an unused output is noise that trains you to skim this list.
  run "fig2_dissociation_sweep (wkshp)" "$PYFIG" paper_8/fig2_dissociation_sweep.py
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
    run "gate ladder val (recompute)" "$PY" scripts/gate_baseline_sensitivity.py --split val --latex
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

if [ "$WITH_BIB" = 1 ]; then
  echo
  echo "-- TIER D: the BIBLIOGRAPHY, by identifier (needs the network) ---------------"
  # Every cited entry resolved to a DOI or arXiv ID and then re-fetched BY that identifier, with the
  # title required to match exactly after TeX normalisation. Never on search rank: asked for
  # "Overcoming catastrophic forgetting in neural networks", Crossref returns a different paper first
  # and says nothing about it, which is how one entry here once shipped with a venue it never had.
  if bib=$("$PY" "$ROOT/scripts/verify_bib.py" 2>&1); then
    echo "$bib" | tail -n 1 | sed 's/^/  /'
  else
    echo "$bib" | tail -n 30 | sed 's/^/  /'
    echo "  A CITED ENTRY DID NOT ROUND-TRIP (see above). A network failure looks the same as a"
    echo "  bad entry here -- read the per-entry block before changing the .bib."
    fail=$((fail + 1))
  fi
else
  echo
  echo "-- TIER D: skipped (pass --with-bib to round-trip the references) ------------"
fi

echo
echo "-- TIER C: NOT re-derivable here --------------------------------------------"
echo "  drift_metric_battery --compute   needs the 59 fine-tuned encoders (*.pt, gitignored)."
# Declared here rather than run above, and the distinction is the same one drift_metric_battery makes:
# building a MoiraiModule to count its parameters needs torch, uni2ts and the HF cache, none of which
# a clean clone has. Its committed OUTPUT (results/model_sizes.json) is what check_paper_numbers.py
# reads, so the three capacities and the 6.6x ratio stay TIER A. Listed rather than omitted because an
# emitter absent from this file is indistinguishable from an emitter nobody wrote.
echo "  emit_model_sizes.py             needs torch + uni2ts + the HF cache; its output"
echo "                                  (results/model_sizes.json) is committed and IS checked."
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
            paper_8/fig_value_axis.pdf paper_8/figA_gate_scatter.pdf \
            paper_8/figA_freeze_boundary.pdf paper_8/figA_window_layout.pdf \
            paper_8/figures/dissociation_trajectory.pdf \
            paper_8/figures/n5k_trajectories.pdf \
            paper_8/fig2_dissociation_sweep.pdf | sed 's/^/    /')
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
  echo "RE-DERIVATION CLEAN (tier A$([ "$WITH_DATA" = 1 ] && echo '+B')$([ "$WITH_BIB" = 1 ] && echo '+D'))."
else
  echo "RE-DERIVATION INCOMPLETE OR DIRTY: $fail item(s) need attention (see above)."
fi
exit "$fail"
