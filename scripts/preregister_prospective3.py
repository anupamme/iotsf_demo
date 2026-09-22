#!/usr/bin/env python3
"""BATCH 3 of the prospective arm: freeze the predictions BEFORE any condition B or D run exists.

WHY A THIRD BATCH.  Batch 1 (results/v47_prospective) tested a predictor the paper has since
CORRECTED: its frozen gate column is the unfitted per-window trend denominator, which a constant
predictor beats, and scripts/score_prospective.py says so at its top.  That batch is reported
unchanged -- rewriting a registration after the fact would turn a prospective test into a
retrospective one -- but it does not test the gate the paper now publishes.  Batch 3 does: the
predictor frozen here is the corrected ridge gate, computed on the SELECTION split, at the operating
point 0.20 the paper defines.

THE TWO BATCHES ARE NOT POOLED, and the reason is not caution but arithmetic: they have a different
predictor (trend vs ridge denominator) AND a different outcome rule (three-clause unanimity vs the
interval rule below).  A pooled precision over the union would be a number about no single
hypothesis.  score_prospective.py reports them side by side with both columns labelled.

THE OUTCOME RULE, AND THE CIRCULARITY IT REMOVES.  The published degradation definition makes
clearing the screen a CLAUSE of degradation, so a gate-failing cell cannot degrade by construction.
Scored that way, every gate-failing cell in this batch would be a true negative whatever its runs
did, and the specificity reported would be an identity rather than a measurement.  The outcome
registered here is therefore paired_inference.degrades_ci_ungated: clauses (ii) and (iii) with the
screen clause REMOVED --

    CI(B - ZS) lies entirely above 0   AND   CI(D - ZS) lies entirely below 0

at 95%, from the paired per-seed differences, with the zero-shot reference's own SEM propagated into
both intervals where the arm is unpaired (it is unpaired on the Moirai cells).  A gate-failing cell
that satisfies both clauses is a FALSE NEGATIVE for the gate, and we expect some: on the published
31 cells the same ungated rule fires on Moirai-Small/ETTh1 at h=96 and h=192, and BOTH of those cells
FAIL the screen.  That is the retrospective pattern this batch is a prospective test of, and it is
the reason the competitor dataset rule is registered again below.

WHAT THIS BATCH CAN AND CANNOT ESTABLISH.  Eight of the ten cells are gate-failing, so if none of
them degrades the arm estimates SPECIFICITY for the first time -- batch 1 had two gate-failing cells
and the published matrix scores the screen only where the screen already selected the cell.
SENSITIVITY remains unestimable unless a gate-PASSING cell degrades, and at an expected prevalence
near zero no number of batches fixes that; we say it in advance rather than reporting a recall of
0/0 as though it were informative.

POWER, STATED IN ADVANCE.  Each cell runs 3 paired seeds.  On the published cells at n=3 the MDE on
the paired encoder contrast is 4.7-14.7 pp, so INCONCLUSIVE is the expected modal call and the
interval clauses will often straddle zero.  The primary confusion matrix counts an inconclusive cell
as NOT degrading -- a cell whose interval straddles zero has not demonstrated degradation -- and a
second matrix restricted to cells where both intervals are decisive is reported beside it with its
own n.  Neither is a substitute for the other and the batch is not a power calculation.

THE CELLS, AND EVERY CELL NOT IN THE BATCH.  The pool is results/v56_pool3/pool_gate_val.json, 19
cells screened before this file was written by scripts/pool_screen_prospective3.py, none of which has
ever been fine-tuned in any arm.  Three rules pick the batch, in this order:

  1. Every Moirai pool cell that CLEARS 0.20.  Exactly two do: large/ETTh2 h192 (+0.924, strong) and
     large/ETTm2 h192 (+0.241, a quarter of a threshold above the line).  This is not a choice.
  2. Every TimesFM cell at h=48, all five datasets.  TimesFM is the second backbone and the cheap
     one (~30 min per cell at 3 paired seeds), and unlike the Chronos arm its B and D score through
     the model's OWN head, so its per-condition forgetting terms are genuine -- which the outcome
     rule above requires.
  3. Three gate-FAILING Moirai cells, so the negative class is not made entirely of one backbone:
     both base/Electricity7 cells (h96 and h192, the two cheapest gate-failing cells in the pool at
     ~50 and ~68 min per run) and large/ETTm2 h96.  The third is chosen for the comparison it makes
     rather than for its price: it differs from the gate-PASSING large/ETTm2 h192 in the horizon and
     in nothing else, so if the gate is tracking the outcome at all, those two cells are where it
     has to show it.

Excluded, with the reason, because a batch that hides what it dropped reports a specificity it did
not earn:

  - All five Chronos cells at h=48 (gates -0.292 to -0.023).  cell_matrix flags the Chronos arm
    forg_confounded: an MSE head is attached for B and D while the zero-shot reference comes from
    the Chronos pipeline, so clauses (ii) and (iii) are not measuring what they name there.  That
    flag predates this batch.  Those cells could still be scored on the encoder contrast; they are
    left out rather than mixed into a confusion matrix they cannot enter.
    The TimesFM h=48 sweep, by contrast, is complete: all five of its datasets are in the batch,
    including Electricity, whose gate (+0.041) is the closest of any pool cell to the threshold from
    below and so the least convenient one to have dropped.
  - Four gate-failing Moirai-Large cells: ETTh1 h192 (-0.289), Weather h192 (-0.521),
    Electricity7 h96 (-0.663) and h192 (-0.548).  Each is 10-26 h of fine-tuning at 3 paired seeds
    against a ~50 h budget for the whole batch (Moirai-Large medians, measured: 101 min per run at
    h=96 on ETTh1, 197 min on Weather).  They are gate-failing like the three Moirai cells that are
    in, so dropping them removes compute and not a class.

Run once, then commit BEFORE launching scripts/run_prospective3.sh.  The ordering is the claim:
`git log` must place this file's commit strictly before the first batch-3 result file.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/v57_prospective3/preregistration_v3.json"
POOL = ROOT / "results/v56_pool3/pool_gate_val.json"
GATE_THRESHOLD = 0.20
DEGRADATION_DATASETS = {"ETTh1", "Weather"}

# Seeds follow each ARM's existing convention so a batch-3 cell is comparable with that arm's
# published cells: the Moirai and TimesFM arms ran {42, 123, 456} in v47/v46, and nothing here
# changes an estimator, so nothing here should change a seed set either.
SEEDS = (42, 123, 456)

# (pool key, arm, size, dataset, horizon, n_train, epochs), in the order they will be run: the two
# gate-passing cells first, because they are the only cells where the gate predicts degradation and
# therefore the only cells that can produce a true positive.
CELLS = [
    ("large_ETTh2_h192", "moirai", "large", "ETTh2", 192, 1000, 20),
    ("large_ETTm2_h192", "moirai", "large", "ETTm2", 192, 1000, 20),
    ("timesfm_etth1_h48", "timesfm", None, "ETTh1", 48, 1000, 20),
    ("timesfm_etth2_h48", "timesfm", None, "ETTh2", 48, 1000, 20),
    ("timesfm_ettm2_h48", "timesfm", None, "ETTm2", 48, 1000, 20),
    ("timesfm_weather_h48", "timesfm", None, "Weather", 48, 1000, 20),
    ("timesfm_electricity_h48", "timesfm", None, "Electricity", 48, 1000, 20),
    ("base_Electricity7_h96", "moirai", "base", "Electricity7", 96, 1000, 20),
    ("base_Electricity7_h192", "moirai", "base", "Electricity7", 192, 1000, 20),
    ("large_ETTm2_h96", "moirai", "large", "ETTm2", 96, 1000, 20),
]


def main():
    if OUT.exists():
        sys.exit(f"{OUT} already exists -- refusing to overwrite a registration")
    if not POOL.exists():
        sys.exit("run scripts/pool_screen_prospective3.py (all three arms) first")

    # The registration is only meaningful if no outcome run exists yet. Both the batch-3 directory
    # and the pool directory are checked: a B/D run under the pool directory would mean a cell was
    # trained while the pool was being screened.
    for d in ("results/v57_prospective3", "results/v56_pool3"):
        existing = sorted(p.name for p in (ROOT / d).glob("*/condition_[BD]*/*.json"))
        if existing:
            sys.exit(f"refusing to register: outcome runs already exist under {d}: {existing[:3]}")

    pool = json.load(open(POOL))
    entries = []
    for key, arm, size, ds, h, n_train, epochs in CELLS:
        if key not in pool:
            sys.exit(f"{key} is not in the screened pool; the predictor must be frozen from the pool")
        p = pool[key]
        if p["baseline"] != "fitted" or p["split"] != "val":
            sys.exit(f"{key}: pool entry is {p['baseline']}/{p['split']}, not the registered "
                     f"fitted/selection-split gate")
        g = p["r2_task"]
        entries.append(dict(
            cell=key, arm=arm, size=size, dataset=ds, horizon=h,
            n_train=n_train, epochs=epochs, seeds=list(SEEDS),
            zeroshot_val_mse=p["zs_test"],      # the selection-split zero-shot; see note below
            gate_val=g,
            predict_gate_rule=bool(g >= GATE_THRESHOLD),
            predict_dataset_rule=ds in DEGRADATION_DATASETS,
        ))

    n_pass = sum(e["predict_gate_rule"] for e in entries)
    payload = dict(
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip(),
        batch=3,
        gate_threshold=GATE_THRESHOLD,
        degradation_datasets=sorted(DEGRADATION_DATASETS),
        seeds=list(SEEDS),
        predictor=dict(
            statistic="R2_task = 1 - MSE_zeroshot / MSE_fitted",
            estimator=("one ridge map from the lookback-96 context to the h-step target, lam=1e-4, "
                       "least squares on this cell's TRAIN windows, applied unchanged"),
            split="selection (validation), disjoint from the windows the outcome is scored on",
            source=str(POOL.relative_to(ROOT)),
            computed_by="scripts/pool_screen_prospective3.py",
            note=("This is the CORRECTED gate. Batch 1 froze the superseded trend denominator, "
                  "which a constant predictor beats; the two batches therefore test different "
                  "predictors and are never pooled."),
        ),
        rules=dict(
            gate_rule="gate_val >= 0.20 -> at risk of degradation (the paper's criterion, corrected)",
            dataset_rule=("dataset in {ETTh1, Weather} -> degradation (POST HOC pattern from the "
                          "original 13 Moirai cells, registered again as a COMPETITOR because both "
                          "cells that satisfy the ungated interval rule on the published 31 are "
                          "ETTh1 cells that FAIL the gate)"),
        ),
        outcome_definition=dict(
            primary=("paired_inference.degrades_ci_ungated: CI(B-ZS) entirely > 0 AND CI(D-ZS) "
                     "entirely < 0 at 95%, from the paired per-seed differences with the zero-shot "
                     "reference's SEM propagated where the arm is unpaired. The screen clause (i) "
                     "of the published definition is REMOVED so that a gate-failing cell can "
                     "falsify the gate instead of being definitionally unable to."),
            secondary=("the three-way call on the paired encoder contrast B-D -- freeze-decisive / "
                       "adapt-decisive / inconclusive at BH q<0.05 computed across the batch-3 "
                       "cells only, never pooled with the published 31."),
            published_definition_also_reported=("cell_matrix.degradation_cells (three-clause "
                                                "unanimity, screen clause included) is computed for "
                                                "these cells too and reported beside the primary."),
            inconclusive=("counted as NOT degrading in the primary confusion matrix; a second "
                          "matrix restricted to cells where both intervals are decisive is reported "
                          "with its own n."),
        ),
        reports=dict(
            confusion_matrix=("rows = predicted at risk (gate_val >= 0.20), columns = primary "
                              "outcome; TP/FP/FN/TN with precision and specificity. Specificity is "
                              "estimable from the 8 gate-failing cells; sensitivity is NOT "
                              "estimable unless a gate-passing cell degrades."),
            competitor_matrix="the same matrix for the dataset rule, side by side.",
            scorer="scripts/score_prospective.py",
        ),
        expectations=dict(
            retrospective_prior=("on the published 31 cells the ungated interval rule fires on "
                                 "Moirai-Small/ETTh1 h96 and h192, both gate-FAILING, so the "
                                 "registered expectation is that the gate's errors are false "
                                 "negatives on ETTh1 rather than false positives."),
            power=("3 paired seeds per cell; at n=3 the published cells' MDE on the encoder "
                   "contrast is 4.7-14.7 pp, so inconclusive is the expected modal call."),
            prevalence=("expected near zero, as in batches 1 and 2 and in the published matrix; if "
                        "it is zero the arm reports specificity and states that sensitivity is "
                        "unestimable."),
        ),
        disclosures=dict(
            zeroshot_predates_registration=(
                "Six of the nine Moirai pool cells take their zero-shot numerator from "
                "results/v48_prospective2, an abandoned batch 2 run on 2026-08-27, and the Moirai "
                "cells' zeroshot_val_mse below therefore predates this file. They are zero-shot "
                "measurements of a released checkpoint and not outcomes: no B or D run exists for "
                "any of them, and a zero-shot MSE cannot move with a fine-tuning result. The "
                "remaining three Moirai pool cells were measured by scripts/run_pool3_zs.sh on "
                "2026-09-22, also before this file."),
            timesfm_zeroshot=("the TimesFM cells' zeroshot_val_mse was computed by the pool screen "
                              "itself on the selection windows, because no run of an h=48 TimesFM "
                              "cell exists; it is the same measurement the h=24 cells store."),
            excluded_cells=(
                "five Chronos h=48 cells (forg_confounded predates this batch, so the outcome rule "
                "is not measurable there) and four gate-failing Moirai-Large cells (10-26 h each "
                "against a ~50 h batch budget). All nine are gate-FAILING in the pool, so the "
                "exclusions remove compute and not a class; their gate values are in the pool file."),
            cost_estimate_h=50,
        ),
        note=("Every prediction is a function of the training and selection splits and the dataset "
              "name only. The outcome requires condition B and D runs, which do not exist for any "
              "cell in this file at the time it is committed."),
        cells=entries,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(entries)} cells, {n_pass} predicted at risk by the "
          f"gate rule, {sum(e['predict_dataset_rule'] for e in entries)} by the dataset rule")
    for e in entries:
        print(f"  {e['cell']:26s} gate {e['gate_val']:+.3f}  "
              f"{'AT RISK' if e['predict_gate_rule'] else 'not at risk':11s}  "
              f"dataset-rule {'degrades' if e['predict_dataset_rule'] else 'no'}")


if __name__ == "__main__":
    main()
