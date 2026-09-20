#!/bin/bash
# Condition E (LoRA, r=8, alpha=16) on the eight Moirai value-cells that do not have it.
#
# WHY THIS RUN EXISTS: the mitigation spectrum -- B/C/D/E/F/G -- has only ever run on
# Moirai-Small/ETTh2, so "the best of six conditions" is a statement about one cell. Every other
# value-cell rests on B and D alone, which means the paper can say whether allowing the encoder to
# adapt helped, but not whether a cheap constrained adaptation would have been better than either
# extreme. LoRA is the one mitigation worth extending first: it is the condition practitioners
# actually reach for, and it is the only one whose reading in this project has ever flipped on a
# hyperparameter (Appendix app:lora_rank: on Moirai-Large the LEARNING RATE, not the rank, decides
# the sign at near-identical CKA). This run takes LoRA coverage of the ten Moirai value-cells from
# two to ten.
#
# WHAT IT CANNOT SETTLE, stated before the runs: E is not a control on encoder adaptation the way D
# is. LoRA adds trainable low-rank updates to the encoder's attention projections, so its CKA is
# strictly between B's and D's, and a good E result is evidence about a MIDDLE point on the
# adaptation axis, not about Delta_encoder. Delta_encoder stays B - D. E enters the paper as a
# practical option on the value-cells, nothing more.
#
# PROTOCOL MATCHING -- read off the stored B/D runs, per cell, before writing this file. The n column
# is the reason this table exists: THREE OF THE EIGHT RAN AT n=500, and a global
# --max-train-samples 1000 would have silently unmatched them against their own B/D arms.
#
#   cell                B/D dir                          n     epochs  B/D seeds
#   small ETTh1 h96     results/v5_etth1                  1000  20      42/123/456/789/999
#   small ETTm2 h96     results/v5_ettm2                  1000  20      42/123/456/789/999
#   small ETTm2 h192    results/v5_ettm2                  1000  20      42/123/456/789/999
#   base  ETTh2 h96     results/v5_etth2_base             1000  20      42/123/456
#   base  ETTh2 h192    results/v5_etth2_base             1000  20      42/123/456
#   base  ETTm2 h96     results/v47_prospective/...        1000  20      42/123/456
#   base  ETTm2 h192    results/v47_prospective/...        1000  20      42/123/456
#   large ETTh2 h96     results/v8_etth2_large             500   20      42/123/456
#
# small/ETTh2 h96 and h192 are NOT here: they already have condition E, in results/v5_mitigation/lora
# at n=1000 against B/D at n=500. That mismatch is pre-existing, is disclosed in the caption of
# tables/mitigation_spectrum.tex, and is not made better by adding a ninth arm at a third n.
#
# LEARNING RATE. Left at the script default, 1e-4, which is the value every condition-B run that
# records the field used. The B/D JSONs for six of these eight cells do not store lr at all, so it
# CANNOT be verified from the run records -- the same limitation run_strict_freeze_ettm2.sh records.
# The rank sweep in app:lora_rank is the reason not to sweep it here: with the LR fixed to B's, a
# difference between E and B is attributable to the low-rank constraint rather than to optimisation.
#
# SEEDS 42/123/456: the intersection of B and D across all eight cells. cell_matrix pairs on seed, so
# three E seeds against five B/D seeds is fine and the comparison is re-averaged on the shared seeds.
#
# Serialized -- concurrent MPS jobs contend badly. Resumes by output-file existence, so this script
# is safe to re-run after an interrupted night; it picks up at the first missing seed.
#
# ORDER: cheapest first, so a closed window loses whole cells at the tail rather than halves of
# every cell. Estimates scaled from the v51/v53 strict-freeze logs, which ran the same model sizes at
# the same n on this machine: Small n=1000 h96 ~20 min/seed, h192 ~27; Base ~46 and ~63; Large n=500
# h96 ~35. LoRA trains fewer parameters than B but the forward/backward cost dominates on MPS, so
# these are treated as upper bounds rather than adjusted downward.
#
# Whatever is missing at write-up time is stated explicitly rather than quietly dropped.
set -uo pipefail
cd /Users/mediratta/code/paper_writing/iotsf_demo
PY=.venv-probe/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=1
OUT=results/v54_lora_valuecells

# PREFLIGHT. The first attempt at this arm died on `ImportError: peft is required for LoRA` one second
# into the first seed, ran all 24 seeds into the same error in two minutes, and then printed "ALL LORA
# VALUE-CELL RUNS DONE" -- so the log's last line said the night had succeeded while nothing had run.
# Condition E is the ONLY condition in this project with a dependency outside the base environment, so
# it is the only one that can fail this way; the check is two lines and it costs nothing.
if ! $PY -c 'import peft' 2>/dev/null; then
  echo "STOPPED: peft is not importable in $PY, and condition E cannot run without it."
  echo "         Install with: uv pip install --python $PY peft"
  exit 2
fi
$PY -c 'import peft; print("  peft " + peft.__version__ + " -- the version every cell in this arm ran")'
# The pre-existing LoRA records (v5_mitigation/lora, v8/v9/v11/v24/v32) were produced under an
# unrecorded earlier peft, so E-vs-E across those arms and this one is not a like-for-like comparison.
# Nothing in the paper makes one: E is read against B and D within its own cell, and those are the
# runs this script matches on protocol.

fails=0
cell() {  # size dataset horizon n_train
  for S in 42 123 456; do
    D_="$OUT/$1_$2_h$3/condition_E"
    [ -f "$D_/condition_E_h$3_s$S.json" ] && continue
    echo "=== $1/$2 h$3 n=$4 cond E r=8 a=16 seed $S  $(date +%H:%M:%S)"
    # PIPESTATUS, not $?: the grep filter is the last command in the pipe, so `$?` reports whether
    # grep matched and a crashed run reads as a success. This is how 24 failed runs got reported as
    # a finished night the first time.
    $PY -u scripts/finetune_forecasting.py --data-path "data/forecasting/$2.csv" \
      --condition E --lora-rank 8 --lora-alpha 16 \
      --model-size "$1" --horizon "$3" --epochs 20 --seed "$S" \
      --max-train-samples "$4" --device mps --results-dir "$D_" 2>&1 \
      | grep -aE 'Zero-shot|forgetting|CKA|LoRA|Traceback|Error' | tail -5
    rc=${PIPESTATUS[0]}
    if [ "$rc" != 0 ] || [ ! -f "$D_/condition_E_h$3_s$S.json" ]; then
      echo "    FAILED (exit $rc, no record written): $1/$2 h$3 seed $S"
      fails=$((fails + 1))
    fi
  done
}

cell small ETTh1  96 1000    # ~1.0h  (20 min/seed)
cell small ETTm2  96 1000    # ~1.0h  (20 min/seed)
cell small ETTm2 192 1000    # ~1.4h  (27 min/seed)
cell large ETTh2  96  500    # ~1.8h  (35 min/seed)   <- n=500, matching v8_etth2_large
cell base  ETTh2  96 1000    # ~2.3h  (46 min/seed)
cell base  ETTm2  96 1000    # ~2.3h  (46 min/seed)
cell base  ETTh2 192 1000    # ~3.2h  (63 min/seed)
cell base  ETTm2 192 1000    # ~3.2h  (63 min/seed)   <- first thing lost if the window closes
written=$(find "$OUT" -name 'condition_E_h*_s*.json' 2>/dev/null | wc -l | tr -d ' ')
echo "LORA VALUE-CELL ARM FINISHED $(date +%H:%M:%S): $written/24 records on disk, $fails failure(s)."
[ "$fails" = 0 ] && [ "$written" = 24 ] || echo "  INCOMPLETE -- say which cells are missing in the paper; do not ship mid-flight seeds."
exit "$fails"
