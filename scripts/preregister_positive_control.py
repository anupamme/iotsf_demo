#!/usr/bin/env python3
"""
POSITIVE CONTROL, STEP 1 of 3. Freeze the design, the thresholds and all three outcomes BEFORE any run.

WHY THIS ARM EXISTS
-------------------
Every result in this paper is negative: across 31 screened cells no cell meets a degradation
definition fixed in advance, and CKA does not order the intervention contrast. A reviewer's objection
to that is not that it is wrong but that it is UNINTERPRETABLE without a positive control -- if
nothing in the matrix was damaged, then "CKA fails to detect damage" and "there was no damage to
detect" make identical predictions, and the paper cannot tell them apart. The only way to separate
them is to construct damage on purpose and ask whether CKA sees it.

So: take the cell with the largest measured pre-trained margin, fine-tune it on a task whose correct
answer contradicts the pre-trained prior, and check (a) whether the capability is destroyed and (b)
whether CKA tracks the destruction.

WHAT THIS ARM DOES *NOT* CLAIM, stated here so it cannot be quietly upgraded later
----------------------------------------------------------------------------------
Task A is the best cell available, not a cell with a large margin over a strong baseline. Read from
the records: its V_ridge is +0.932, but against the STRONGEST admissible rung of the eight-baseline
ladder (seasonal-naive) the same cell is only +0.081, and it does not clear the ladder. That is the
paper's own headline and this arm does not escape it. Consequently this is a test of whether CKA can
DETECT a large retention loss, measured against the checkpoint's own zero-shot performance -- a
self-referential quantity that needs no baseline. It is not a demonstration that the destroyed
capability was economically valuable. The registered outcome sentences below are written in those
terms.

WHY THE DESTRUCTION IS ENGINEERED RATHER THAN FOUND
---------------------------------------------------
Fine-tuning Moirai on another ETT series does not destroy anything -- that is this paper's main
result, where condition B frequently HELPS. Task B is therefore synthetic and adversarial by
construction: scripts/make_conflicting_series.py builds a series with x(t+192) = -x(t), so the
seasonal-naive prior (the strongest admissible rung on task A's own cell) returns exactly the negation
of the truth and is 200.9x worse than that negation. Learning task B REQUIRES overwriting the prior
rather than extending it. It is pinned here by SHA-256, not by filename.

THE HONEST READING OF AN ENGINEERED CONTROL. A conflict this sharp is not a realistic deployment;
nobody fine-tunes a forecaster on a sign-inverting series. That is the point. A positive control is
supposed to be the easiest possible case for the detector: if CKA cannot order damage HERE, the
negative result on the benign benchmark cells is not a power problem, and if it CAN, then Claim A is
bounded to the benign regime and we say so. Either way the arm is informative, which is why all three
outcomes are written below before any run exists.

Run once, then commit, THEN run the grid. `git log` must show this file's commit strictly before the
first positive-control result commit or the arm may not be called pre-registered.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/positive_control/preregistration.json"

GATE_THRESHOLD = 0.20
TASK_B_JSON = ROOT / "results/synthetic_task_b.json"
GATE_VAL = ROOT / "results/gate_val_side.json"
LADDER_VAL = ROOT / "results/gate_baselines_val.json"

# ---------------------------------------------------------------------------
# The grid, fixed here.
# ---------------------------------------------------------------------------
# Three learning rates spanning two orders of magnitude, which is the destruction ladder: the graded
# dose. 1e-4 is the main matrix's default, so the bottom rung is the paper's own protocol and the top
# rung is 100x it.
LADDER_LRS = [1e-4, 1e-3, 1e-2]
# ONE declared-in-advance extension, and only one. If destruction is not achieved at 1e-2 we may add
# 1e-1 and then we stop, whatever it shows. Escalating until the result appears is p-hacking by
# another route, so the stopping rule is registered rather than decided later.
DECLARED_EXTENSION_LR = 1e-1
CONDITIONS = ["B", "D"]          # full fine-tune vs frozen encoder: the paper's own contrast
SEEDS = [42, 123, 456]
EPOCHS = 20                      # the runner default, unchanged
MAX_TRAIN_SAMPLES = 1000         # the runner default, unchanged
BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Thresholds. Numeric, and fixed now.
# ---------------------------------------------------------------------------
# "Destruction achieved": task A's MSE rises by at least this much over the same checkpoint's
# zero-shot MSE, in a MAJORITY of seeds at some rung under condition B. 50% is far outside anything
# the main matrix produces (where Delta_enc spans -51.8% to +46.1%), so it cannot be met by noise.
DESTRUCTION_THRESHOLD_PCT = 50.0
# "Task B was actually learned": its own held-out MSE must fall by at least this much. Destruction
# without learning is weight thrashing, not forgetting, and is reported as such.
TASKB_LEARNED_THRESHOLD_PCT = 20.0
# "CKA orders the destruction": Spearman rho between CKA(theta_FT, theta_ZS) measured on TASK A's
# inputs and retention_A across all runs must be NEGATIVE with magnitude at least this. Negative is
# the only direction that means anything: more damage must come with less similarity.
CKA_ORDERING_RHO = -0.50


def read_task_a():
    """Task A = argmax V_ridge over the gate-passing cells, read from the records, not from prose."""
    gates = json.load(open(GATE_VAL))
    survivors = {k: v for k, v in gates.items() if v["r2_task"] >= GATE_THRESHOLD}
    if len(survivors) != 7:
        sys.exit(f"expected 7 gate-passing cells on the selection split, found {len(survivors)}; "
                 f"the registration's selection rule assumes the published survivor set")
    key = max(survivors, key=lambda k: survivors[k]["r2_task"])
    rec = survivors[key]
    size, dataset, htag = key.split("_")
    horizon = int(htag.lstrip("h"))

    # The honest second margin: the same cell against the STRONGEST admissible rung of the ladder.
    sys.path.insert(0, str(ROOT / "scripts"))
    import gate_baseline_sensitivity as gbs
    lad = json.load(open(LADDER_VAL))
    graded = gbs.graded_value(lad["cells"], key, lad["baselines"])
    return dict(
        cell=key, model_size=size, dataset=dataset, horizon=horizon,
        v_ridge_selection=rec["r2_task"],
        zeroshot_mse_selection=rec["zs_test"],
        ridge_baseline_mse_selection=rec["linear_test"],
        n_windows=rec["n_windows"],
        v_best_admissible_rung=graded["r2_best"],
        strongest_admissible_rung=graded["best_baseline"],
        clears_all_admissible_rungs=graded["clears_all_admissible"],
        selection_rule="argmax V_ridge over the 7 cells clearing 0.20 on the selection split",
        data_path=f"data/forecasting/{dataset}.csv",
        margin_caveat=("V_ridge is +%.3f but the margin over the strongest admissible rung (%s) is "
                       "only +%.3f and the cell does not clear the ladder; this arm tests DETECTION "
                       "of a retention loss, not the value of what is lost"
                       % (rec["r2_task"], graded["best_baseline"], graded["r2_best"])),
    )


def main():
    if OUT.exists():
        sys.exit(f"{OUT.relative_to(ROOT)} already exists -- refusing to overwrite a registration")

    # Guard: the registration is only meaningful if no outcome run exists yet. Checked structurally,
    # not trusted: the outcome variable must not exist anywhere under the arm's results directory.
    existing = sorted(str(p.relative_to(ROOT)) for p in
                      (ROOT / "results/positive_control").glob("**/*.json"))
    if existing:
        sys.exit(f"refusing to register: positive-control runs already exist: {existing[:3]}")

    if not TASK_B_JSON.exists():
        sys.exit("run scripts/make_conflicting_series.py first -- task B is pinned by hash")
    taskb = json.load(open(TASK_B_JSON))

    task_a = read_task_a()

    runs = [dict(condition=c, lr=lr, seed=s) for lr in LADDER_LRS for c in CONDITIONS for s in SEEDS]

    reg = dict(
        registered_by="scripts/preregister_positive_control.py",
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip(),
        purpose=("Construct a regime where fine-tuning demonstrably destroys a pre-trained "
                 "capability, and test whether CKA detects the damage."),

        task_a=task_a,
        task_b=dict(path=taskb["path"], sha256=taskb["sha256"],
                    generator=taskb["generator"], seed=taskb["seed"],
                    construction=taskb["construction"],
                    seasonal_naive_over_its_negation=taskb["seasonal_naive_over_its_negation"],
                    why=("the correct forecast is the NEGATION of the last period, so the "
                         "seasonal-naive prior -- the strongest admissible rung on task A's own "
                         "cell -- is maximally wrong; learning B requires overwriting the prior")),

        grid=dict(learning_rates=LADDER_LRS, conditions=CONDITIONS, seeds=SEEDS,
                  epochs=EPOCHS, max_train_samples=MAX_TRAIN_SAMPLES, batch_size=BATCH_SIZE,
                  n_runs=len(runs), runs=runs,
                  declared_extension_lr=DECLARED_EXTENSION_LR,
                  stopping_rule=("if destruction is not achieved at lr=1e-2 we may add lr=1e-1 ONCE "
                                 "and then stop and report whatever it shows")),

        primary_readout=dict(
            name="retention_A",
            formula="100 * (MSE_A(theta_FT) - MSE_A(theta_ZS)) / MSE_A(theta_ZS)",
            windows="task A's HELD-OUT windows, built by make_eval_sequences at lookback 96, h=192",
            sign="POSITIVE means the capability got worse, i.e. damage",
            contrast=("retention_A(B) - retention_A(D), on the same 100/L_ZS scaling as Eq. 1, so it "
                      "is directly comparable to Delta_enc in the main matrix")),

        primary_checkpoint=dict(
            which="the early-stopped checkpoint",
            why=("comparability with the main matrix, which early-stops on task B's validation MSE"),
            also_reported=("the FINAL-epoch state, as the maximal-destruction bound: early stopping "
                           "on task B can halt before destruction of task A completes, so the final "
                           "state is the fairer test of whether damage is constructible at all")),

        cka=dict(
            measured_on="TASK A's inputs",
            why=("this differs from the main matrix, where task A and task B coincide and the "
                 "distinction is empty. Here it is the whole question: CKA must be asked whether the "
                 "representation of the capability being lost has changed. Measuring it on task B's "
                 "inputs would answer a different question -- whether the model adapted to B -- and "
                 "would be trivially yes."),
            ordering_test=("Spearman rho between CKA and retention_A across all runs"),
            ordering_threshold=CKA_ORDERING_RHO),

        thresholds=dict(destruction_pct=DESTRUCTION_THRESHOLD_PCT,
                        destruction_rule=("retention_A >= %.0f%% in a MAJORITY of seeds at some rung "
                                          "under condition B" % DESTRUCTION_THRESHOLD_PCT),
                        taskb_learned_pct=TASKB_LEARNED_THRESHOLD_PCT,
                        taskb_learned_rule=("MSE on task B's own held-out windows falls by at least "
                                            "%.0f%% vs zero-shot" % TASKB_LEARNED_THRESHOLD_PCT),
                        cka_ordering_rho=CKA_ORDERING_RHO),

        validity_conditions=[
            dict(name="task B is actually learned",
                 rule="MSE_B(theta_FT) < MSE_B(theta_ZS) by >= %.0f%%" % TASKB_LEARNED_THRESHOLD_PCT,
                 if_violated=("any collapse on task A is weight thrashing rather than forgetting, "
                              "and must be reported as such rather than as a positive control")),
            dict(name="condition D preserves task A",
                 rule="retention_A(D) is small relative to retention_A(B) at the same rung",
                 falsifiable=True,
                 if_violated=("the damage is not located in the encoder, which is what already "
                              "happens on Chronos/ETTh1 in Sec. 6; the frozen-encoder control would "
                              "then be uninformative here too and we say so")),
        ],

        outcomes_fixed_in_advance={
            "1_destruction_and_cka_orders_it": (
                "CKA has a domain of validity: it detects damage when the effect is large and "
                "engineered, and fails to order the small-effect benign regime these benchmarks "
                "produce. Claim A narrows explicitly to the benign regime, and the paper gains its "
                "positive control -- the negative result is then about effect size, not about CKA "
                "being blind."),
            "2_destruction_but_cka_does_not_order_it": (
                "The strongest form of the negative claim: CKA fails to order damage even when the "
                "damage is large, engineered and known to be present, so its failure on the "
                "benchmark cells is not a power problem."),
            "3_destruction_not_achieved": (
                "The positive control failed to construct. We report the full ladder and the "
                "failure, and state that deliberately adversarial fine-tuning did not destroy the "
                "capability -- itself a finding about TSFM robustness, and a reason the main "
                "matrix's null is unsurprising. One declared extension to lr=1e-1, then stop."),
        },

        what_this_cannot_show=[
            "that the destroyed capability was valuable against a strong baseline (it was +0.081 "
            "over seasonal-naive)",
            "that anything generalises beyond one backbone, one series and one synthetic task B",
            "that a sign-inverting target task resembles a deployment scenario",
        ],
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reg, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"\ntask A: {task_a['cell']}  V_ridge={task_a['v_ridge_selection']:+.4f}  "
          f"(vs strongest admissible rung {task_a['strongest_admissible_rung']}: "
          f"{task_a['v_best_admissible_rung']:+.4f})")
    print(f"task B: {taskb['path']}  sha256={taskb['sha256'][:16]}...  "
          f"seasonal-naive {taskb['seasonal_naive_over_its_negation']:.1f}x worse than its negation")
    print(f"grid:   {len(runs)} runs = {len(LADDER_LRS)} lrs x {len(CONDITIONS)} conditions "
          f"x {len(SEEDS)} seeds")
    print(f"\nCOMMIT THIS FILE BEFORE LAUNCHING ANY RUN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
