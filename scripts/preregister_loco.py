#!/usr/bin/env python3
"""
Freeze the LOCO predictive-model specification BEFORE the models are fitted.

WHAT GUARANTEE THIS DOES AND DOES NOT GIVE
------------------------------------------
This is NOT the same kind of pre-registration as results/v48_prospective2/preregistration_v2.json.
There, the outcome variable did not exist: condition B and D runs had not been launched, so the
prediction could not have been informed by the answer. Here the outcome (bd_test, held-out B-D) is
already computed and already published in the paper. What is frozen is therefore the ANALYSIS, not a
prediction about unseen data:

  * which nested models are compared, and in which order,
  * how the covariates are built from a cell label,
  * which statistic decides the question,
  * and -- the part that actually matters -- WHAT SENTENCE THE PAPER WILL PRINT under each of the
    three possible outcomes, written here before any of them is known.

That is a real but weaker guarantee, and the paper must describe it as the weaker thing: a
pre-specified analysis, not a prospective test. Its value is that the specification cannot be tuned
until it produces the congenial answer, which is the failure mode a reader should worry about when a
paper reports a null.

THE QUESTION, in the reviewer's words: does CKA add predictive information about the value of encoder
adaptation beyond backbone/dataset identity? The paper currently answers with a rank correlation and a
wide clustered interval, which establishes a FAILURE TO ESTABLISH a relationship rather than evidence
of its absence. An out-of-sample comparison of nested models answers the decision-relevant version:
if you already know which backbone and series you are on, does measuring CKA help you predict whether
letting the encoder move was worth it?

THE NESTED LADDER. Outcome is bd_test (held-out B-D, percent of zero-shot loss; positive = freezing
better), on all 31 cells that have both bd_test and cka.

    M0  intercept only
    M1  backbone FE
    M2  backbone FE + horizon + log n
    M3  M2 + CKA                       <- the model under test
    M4  M2 + CKA x backbone

DECIDING STATISTIC:  dR2 = R2_LOCO(M3) - R2_LOCO(M2).
PREDICTION REGISTERED HERE:  dR2 <= 0.

WHY DATASET FE IS NOT IN THE LADDER, and this is the trap that would have made the whole analysis
meaningless. The reviewer asks for dataset in the model AND leave-one-cluster-out validation. Those
two requests are jointly unsatisfiable: a cluster IS (backbone, dataset), so holding out a cluster
removes the only cells that identify that dataset's fixed effect, and the held-out fold's prediction
silently collapses onto the intercept. The fold then scores a model nobody specified. Worse, three of
the seven series (electricity, electricity7, ili) are covered by ONE backbone each, so their dataset
effect is not identified by any other fold even in principle.
So we run two schemes and report both, labelled for what they are:

  LOCO          cluster = (backbone, dataset) from cluster_keys.cluster_of; dataset FE EXCLUDED.
                Genuinely out-of-sample. This is what dR2 is computed on, and it is the well-posed
                form of "does CKA beat knowing the backbone".
  by-dataset    leave out all cells of one series across backbones, WITH dataset FE in the fit.
                The held-out series' own FE is unidentified by construction, so this is reported as
                an in-sample fit plus a per-series residual table, and it is NOT called
                out-of-sample. It exists to show the dataset effects, not to validate.

TWO MEASUREMENT TRAPS IN THE COVARIATES, fixed here so the fit cannot inherit them:

  (1) Horizon cannot be parsed with h(\\d+): the dataset names contain h1, h2 and m2, so that pattern
      reads "Moirai-small/ETTh2 h192 n500" as horizon 2. Nine of the 31 cells parse WRONG that way
      and the error is invisible -- 2 is a plausible-looking number. Horizon is therefore taken from
      the horizon TOKEN only, mirroring cell_matrix._display's three label shapes.
  (2) n comes from two complementary places and neither alone covers the matrix. rows carry
      n_train for 22 of 31 cells; the other 9 carry it as an "n500"/"n1000" token in the label and
      have n_train = None. Using only one source would drop or mislabel a third of the cells. n spans
      452 (ILI, which trains on the whole split) to 8,000 (Chronos), so it enters as log n.

R2 CONVENTION: out-of-sample R2 is computed against the TRAINING-fold mean, never the held-out fold's
own mean. Against its own mean, a fold with little internal spread looks catastrophic and one with a
lot looks good, and the aggregate then reports fold composition rather than model quality. Per-fold
MAE is reported alongside pooled-residual R2 because the folds are unbalanced (cluster sizes run 1 to
5 cells).

INFERENCE: the cluster bootstrap already in cka_fixed_effects.py, resampling clusters with
replacement, reported for b1 (the CKA coefficient in M3) and for dR2 itself.

Run once, then commit, then fit:
    .venv12/bin/python scripts/preregister_loco.py
"""
import contextlib
import io
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cell_matrix  # noqa: E402
from cluster_keys import backbone_of, cluster_of, dataset_of  # noqa: E402

OUT = ROOT / "results/preregister_loco.json"

# The three label shapes, and the horizon token in each. Mirrors cell_matrix._display rather than
# inventing a fourth parse: "Chronos/etth1 h24", "Moirai-base_ETTh1_h192", "Moirai-small/ETTh2 h192
# n500". In every shape the horizon is a whitespace- or underscore-delimited "h<digits>" TOKEN, which
# is what distinguishes it from the h2 inside ETTh2.
H_TOKEN = re.compile(r"(?:^|[\s_])h(\d+)(?:$|[\s_])")
N_TOKEN = re.compile(r"(?:^|[\s_])n(\d+)(?:$|[\s_])")


def covariates_of(cell, n_train):
    """(horizon, n) for a cell, from the label token and the run records, with both traps closed."""
    m = H_TOKEN.search(cell)
    if not m:
        raise SystemExit(f"no horizon token in cell label: {cell!r}")
    h = int(m.group(1))
    tok = N_TOKEN.search(cell)
    if tok is not None:
        n = int(tok.group(1))
        # If both sources exist they must agree, or one of them is describing a different run.
        if n_train is not None and int(n_train) != n:
            raise SystemExit(f"{cell}: label says n={n} but records say n_train={n_train}")
    elif n_train is not None:
        n = int(n_train)
    else:
        raise SystemExit(f"no training-set size for {cell!r} in either the label or the records")
    return h, n


def main():
    if OUT.exists():
        sys.exit(f"{OUT} already exists -- refusing to overwrite a registration")

    with contextlib.redirect_stdout(io.StringIO()):
        rows = cell_matrix.build_rows()
    rows = [r for r in rows if r.get("bd_test") is not None and r.get("cka") is not None]

    cells = []
    for r in rows:
        h, n = covariates_of(r["cell"], r.get("n_train"))
        # PREDICTORS ONLY. bd_test is deliberately not written here: the point of the file is to fix
        # the design and the decision rule, and a registration that also transcribes the outcome
        # invites the reading that the two were chosen together.
        cells.append(dict(cell=r["cell"], backbone=backbone_of(r["cell"]),
                          dataset=dataset_of(r["cell"]), cluster=cluster_of(r["cell"]),
                          horizon=h, n_train=n, cka=r["cka"]))

    clusters = sorted({c["cluster"] for c in cells})
    payload = dict(
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip(),
        guarantee=("PRE-SPECIFIED ANALYSIS, NOT A PROSPECTIVE TEST. The outcome (bd_test) already "
                   "exists and is already published; what is frozen is the model ladder, the "
                   "covariate construction, the deciding statistic and the sentence the paper "
                   "prints under each outcome. Weaker than results/v48_prospective2/"
                   "preregistration_v2.json, where the outcome runs did not yet exist, and the "
                   "paper must describe it as the weaker thing."),
        outcome="bd_test (held-out B-D as percent of zero-shot loss; positive = freezing better)",
        models=dict(
            M0="intercept",
            M1="backbone FE",
            M2="backbone FE + horizon + log n",
            M3="M2 + CKA",
            M4="M2 + CKA x backbone",
        ),
        deciding_statistic="dR2 = R2_LOCO(M3) - R2_LOCO(M2)",
        prediction="dR2 <= 0 (CKA adds no out-of-sample information beyond backbone identity)",
        validation=dict(
            loco=("leave-one-cluster-out, cluster = (backbone, dataset) via cluster_keys.cluster_of; "
                  "dataset FE EXCLUDED because holding out a cluster removes the cells that identify "
                  "it. Genuinely out-of-sample; dR2 is computed here."),
            by_dataset=("leave-one-series-out WITH dataset FE, reported as in-sample fit plus "
                        "per-series residuals and explicitly NOT called out-of-sample, because the "
                        "held-out series' own FE is unidentified by construction."),
        ),
        r2_convention=("out-of-sample R2 against the TRAINING-fold mean, never the held-out fold's "
                       "own mean; per-fold MAE reported alongside because cluster sizes are 1-5"),
        inference="cluster bootstrap (resample clusters with replacement) on b1 of M3 and on dR2",
        # Written before the answer is known, so that whichever fires, the wording was not chosen to
        # suit it. These are the exact claims the body will carry.
        outcome_branches={
            "dR2 <= 0": (
                "STRONGEST OUTCOME FOR THE PAPER. Claim A promotes from 'no reliable ordering' to "
                "'no incremental out-of-sample predictive value at this n': knowing CKA does not "
                "improve prediction of the value of encoder adaptation over knowing the backbone, "
                "horizon and training-set size. Title and abstract unchanged."),
            "dR2 > 0, bootstrap CI excludes 0": (
                "THE CLAIM NARROWS. Section 6's framing changes to 'CKA carries some out-of-sample "
                "information about the value of encoder adaptation, insufficient to license a "
                "decision rule: no threshold separates the cells adaptation helped from those it "
                "hurt, and the predictive gain is dR2.' The abstract's ordering sentence is replaced "
                "by that sentence. The title survives unchanged: a predictor carrying signal is "
                "still not a substitute for the intervention, which is what the title claims."),
            "dR2 > 0, bootstrap CI covers 0": (
                "INDETERMINATE AT THIS N. Reported as a point estimate with its interval and the "
                "explicit statement that 15 clusters cannot separate a modest predictive gain from "
                "none. No existing claim changes; the appendix carries the numbers."),
        },
        n_cells=len(cells),
        n_clusters=len(clusters),
        clusters=clusters,
        cells=cells,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(cells)} cells, {len(clusters)} clusters")
    print("  horizons:", sorted({c['horizon'] for c in cells}))
    print("  n_train :", sorted({c['n_train'] for c in cells}))
    print("\nCommit this file BEFORE fitting the models.")


if __name__ == "__main__":
    main()
