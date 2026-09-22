#!/usr/bin/env python3
"""Register the Chronos/M4-Monthly re-run BEFORE its gate is computed or any B/D run is launched.

WHAT THIS ARM IS, AND WHAT IT IS NOT. It is NOT prospective, and the paper must not call it that: an
earlier version of this cell was run at ten seeds and its outcome -- forgetting near zero, CKA near
one -- is known to us. What is legitimately fixable in advance is everything else, and that is what
this file fixes: the data, the split, the gate estimator, the seeds, the training protocol, the
decision rule, and WHAT EACH GATE OUTCOME WILL MEAN. The last of those is the one that matters. A
reviewer asked for a gate-positive non-Moirai cell; if we computed the gate first and wrote the
interpretation afterwards, either answer could be narrated as a success, and the sentence below is
what stops that.

WHY THE CELL IS BEING RE-RUN AT ALL. Appendix "Chronos-T5-Small on M4-Monthly" reported ten seeds,
two sample sizes, a random-init control and an 84.5% gate, and NO file under results/ can produce any
of it: the series were never retained, the 200-series selection was never written down, and the gate
was measured with the superseded per-window trend denominator on the TEST split. Under this project's
own rule -- every number in the paper traces to an executed run with an emitter and a check -- that
appendix was the largest integrity exposure in the paper. This re-run either supplies the records or
the numbers come out of the paper.

THREE CORRECTIONS TO THE OLD PROTOCOL, all of them fixed here before any number is seen.

  1. The gate is scored on the SELECTION split with the FITTED ridge, which is how
     Section "Protocol" defines it, against the old script's test-split score with a per-series
     LinearRegression fitted per window.
  2. The splits are chronological and disjoint in their targets (scripts/m4_monthly_data.py). The
     old script cut a randomly subsampled window pool at the 80th percentile of its own index, which
     puts 95%-overlapping neighbours on both sides of the "split".
  3. Early stopping selects on the selection windows, never on the held-out ones, and the best epoch
     must be at least 1 -- a "fine-tune" that early-stops back onto the pre-trained weights is not a
     fine-tune, and six of the old ten seeds did exactly that (best_epoch=0, CKA=1.000 by
     definition), which is most of why the old appendix read as "no drift".

THE TWO OUTCOME SENTENCES, WRITTEN NOW.

  If the corrected gate clears 0.20:  the paper gains the gate-positive non-Moirai cell the review
    asks for, and the CKA-vs-Delta_enc dissociation gets a second backbone at a second horizon. The
    cell then carries a paired intervention contrast that can be read beside Moirai's, though still
    not pooled with it (different objective, different horizon, different normalisation).

  If it does not clear 0.20:  the 84.5% advantage was an artefact of the unfitted trend denominator,
    exactly as the 17-of-21 Moirai correction was, and the paper reports that NO non-Moirai cell in
    reach clears its own screen. That is a finding about the benchmark x backbone pairing -- these
    backbones have no demonstrated pre-trained advantage over a ridge fitted on the target data --
    and not a gap in the design. The screen's disqualifier reading is unaffected; what goes is any
    hope of a non-Moirai cell with something to preserve.

Either way the appendix is rewritten from the records this arm produces, and every number the re-run
does not reproduce is deleted rather than restated.

Run once, then commit BEFORE scripts/gate_chronos_m4.py and scripts/chronos_m4_corrected.py.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/chronos_m4/preregistration.json"

GATE_THRESHOLD = 0.20
SEEDS = (42, 43, 44)                 # the seeds the other five Chronos cells use
N_TRAIN = 500
CONDITIONS = ("B", "D")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT.exists():
        sys.exit(f"{OUT} already exists -- refusing to overwrite a registration")
    # The registration is only meaningful if no outcome and no GATE exists yet: the gate estimator is
    # one of the things being fixed, so seeing the gate first would let the estimator be chosen to
    # clear the threshold.
    for stray in sorted((ROOT / "results/chronos_m4").glob("**/*.json")):
        sys.exit(f"refusing to register: {stray.relative_to(ROOT)} already exists")

    from m4_monthly_data import (HORIZON, LOOKBACK, MIN_LEN, NUM_SERIES, TEST_CSV, TRAIN_CSV,
                                 build_splits, load_series)

    series = load_series()
    _, _, _, info = build_splits(series, max_train=N_TRAIN, seed=SEEDS[0])

    payload = dict(
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip(),
        cell="chronos_m4monthly",
        prospective=False,
        why_not_prospective=("an earlier ten-seed run of this cell exists and its outcome is known "
                             "to us; only the protocol, the estimator and the interpretation are "
                             "fixed in advance here"),
        data=dict(
            train_csv=TRAIN_CSV, test_csv=TEST_CSV,
            train_sha256=sha256(ROOT / TRAIN_CSV), test_sha256=sha256(ROOT / TEST_CSV),
            selection_rule=(f"first {NUM_SERIES} series in file order with history length "
                            f">= {MIN_LEN} and 18 test points; no RNG in the selection"),
            first_series=info["series_ids"][:5], last_series=info["series_ids"][-1],
            lookback=LOOKBACK, horizon=HORIZON,
            split=dict(
                train="sliding windows over history[:-114]",
                selection="context history[-114:-18], target history[-18:] (one window per series)",
                heldout="context history[-96:], target the official M4-Monthly test values",
                disclosed_overlap=("the held-out CONTEXT contains the 18 values used as the "
                                   "selection target, because M4's test set is defined as the 18 "
                                   "steps following the history; no held-out TARGET is ever a "
                                   "training or selection target"),
            ),
            train_windows_available=info["train_windows_available"],
            degenerate_train_contexts_dropped=info["degenerate_train_contexts_dropped"],
            n_selection_windows=info["n_selection_windows"],
            n_heldout_windows=info["n_heldout_windows"],
        ),
        gate=dict(
            estimator=("fitted: one ridge-OLS map R^{96} -> R^{18}, lam=1e-4, least squares on the "
                       "TRAIN windows, applied unchanged; identical to gate_baselines' `fitted` rung "
                       "and to _window_linear_mse(baseline='fitted') in gate_all_cells.py"),
            normalisation="per-window z-score by the context's own mean/sd (the Chronos arm's scale)",
            split="selection",
            statistic="R2_task = 1 - MSE_zeroshot / MSE_fitted",
            threshold=GATE_THRESHOLD,
            ladder=("the full eight-rung ladder plus the `constant` admissibility floor is computed "
                    "and reported for this cell, as for every other cell; the primary number is the "
                    "`fitted` rung"),
            season_steps=12,
        ),
        runs=dict(
            script="scripts/chronos_m4_corrected.py",
            model="amazon/chronos-t5-small",
            objective=("Chronos's native tokenised cross-entropy, as the superseded appendix "
                       "documented for this cell -- NOT the direct-MSE head the other five Chronos "
                       "cells use, so the two are reported separately and never pooled"),
            conditions=list(CONDITIONS),
            seeds=list(SEEDS),
            n_train_windows=N_TRAIN,
            lr=1e-5, batch_size=32, epochs=20, patience=5, weight_decay=0.01, grad_clip=1.0,
            early_stopping=("on the selection windows; best epoch >= 1, so a run may not select the "
                            "pre-trained checkpoint"),
            evaluation="batched median of 20 Chronos samples, on the 200 held-out windows",
            diagnostics=("linear CKA of the final encoder output, mean-pooled over tokens, on the "
                         "200 selection contexts; l2 weight drift of the encoder state dict"),
        ),
        outcome=dict(
            primary=("Delta_enc = forgetting_B - forgetting_D on the HELD-OUT windows, per seed, "
                     "with a paired two-sided 95% t-interval over the three seeds"),
            decision_rule=("freezing decisively better if the interval lies entirely above 0; "
                           "adaptation decisively better if entirely below 0; inconclusive "
                           "otherwise -- the same three-way rule the 31 published cells use"),
            multiplicity=("NOT in the Benjamini-Hochberg family of the 31 published cells: "
                          "different horizon, different objective, different dataset family. "
                          "Reported unadjusted and as its own cell, never pooled into the 2/4/25 "
                          "split"),
            degradation=("the three-clause definition applies unchanged: gate >= 0.20 AND "
                         "forgetting_B > 0 AND forgetting_D < 0"),
            power=("three seeds resolve only large effects; the minimum detectable effect will be "
                   "reported for this cell as it is for the others, and an inconclusive call will "
                   "be reported as inconclusive rather than as absence"),
        ),
        interpretation=dict(
            gate_clears=("the paper gains the gate-positive non-Moirai cell the review asks for, and "
                         "the CKA-vs-Delta_enc dissociation gains a second backbone at a second "
                         "horizon; still read beside Moirai, not pooled with it"),
            gate_fails=("the 84.5% figure was an artefact of the unfitted trend denominator, and the "
                        "paper reports that no non-Moirai cell in reach clears its own screen -- a "
                        "finding about the benchmark x backbone pairing, not a gap in the design"),
        ),
        appendix_commitment=("app:chronos_detail is rewritten from the records this arm produces; "
                            "every number the re-run does not reproduce is deleted, not restated"),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(series)} series, {info['train_windows_available']} train windows available")
    print(f"  seeds {SEEDS}, conditions {CONDITIONS}, n_train={N_TRAIN}")
    print("  COMMIT THIS BEFORE running gate_chronos_m4.py or chronos_m4_corrected.py")


if __name__ == "__main__":
    main()
