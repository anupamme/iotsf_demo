#!/usr/bin/env python3
"""Register the seed top-up BEFORE running it, because otherwise it is optional stopping.

WHY THIS FILE EXISTS.  Five of the seven gate-passing cells come out INCONCLUSIVE on the paired
encoder contrast, and the obvious response -- add seeds until something resolves -- is the textbook
way to manufacture a significant result.  What makes a top-up legitimate is fixing, in advance: which
cells get seeds, how many, which seeds, and what will be reported whatever happens.  All four are
written here, together with the two cells we commit NOT to top up however their p-values move.

THE TARGET, IN TWO TIERS, AND WHY IT IS NOT ONE NUMBER.  A cell has two required-n budgets and they
differ by an order of magnitude: the UNCORRECTED one, the smallest n at which a paired t-test at
alpha=0.05 resolves the effect the cell has already shown, and the BH-ADJUSTED one, which is what the
paper's own call requires because that call is adjusted across 31 cells.  On these cells the adjusted
budgets are 4, 5, 29, 36, 182 and 212 against uncorrected budgets of 2, 3, 3, 12, 14, 71 and 83.  The
registered rule takes the adjusted budget WHERE IT IS FEASIBLE, because that is the only budget that
can change a verdict, and falls back to the uncorrected one where it is not:

    inconclusive at BH  and  adjusted n <= 20   ->  run to the adjusted n
    inconclusive at BH  and  uncorrected n <= 20 ->  run to the uncorrected n
    otherwise                                    ->  unresolvable; not topped up

with an MDE target of 8 pp quoted alongside as the scale a reader should hold the cells to.  The
consequence is stated in advance rather than discovered afterwards: TWO OF THE THREE TOPPED-UP CELLS
FALL IN THE SECOND TIER AND ARE EXPECTED TO REMAIN INCONCLUSIVE UNDER THE PAPER'S BH CALL.  What the
top-up buys there is a tighter interval and a smaller MDE on cells the paper leans on, not a change of
verdict.  If a verdict does change, the augmented n is reported beside the original n and the sentence
says at which n it changed.

THE SD IS OBSERVED, NOT BLIND, AND THAT IS A REAL LIMITATION.  Every required n below is computed
from the cell's own already-measured sd, so this is not a power calculation made before seeing data;
it is a stopping rule fixed before seeing MORE data.  A cell whose sd is underestimated by its first
few seeds will get a budget that is too small, and the augmented interval will then be wider than
this file predicts.  That is a property of any retrospective power exercise and the reason the MDE
table is published rather than only the calls.

EXCHANGEABILITY OF THE ADDED SEEDS.  The new seeds are run today, into the same directories, with the
same size, horizon, epoch budget and sample cap as the cell's existing seeds; but the cell's original
seeds were run months earlier, and the paper's own two-cell reproduction check found that a Moirai
cell reproduces its MSEs to within a few percent and its forgetting percentage to within 63.5%.  So
the added seeds are not guaranteed to be draws from the same distribution as the old ones.  This is
why F3 reports original-n and augmented-n side by side in one table instead of quietly replacing the
published number, and why no claim in the paper will rest on the augmented n alone.

WHAT IS DECLARED UNRESOLVABLE IN ADVANCE.  Any gate-passing inconclusive cell whose uncorrected
required n exceeds the feasibility cap of 20.  Those cells are NOT topped up, whatever their p-values
do, and the paper reports the required n as the finding: a cell needing 71 or 83 paired seeds to
resolve a 2.8 pp or 1.3 pp effect is telling the reader the effect is small relative to the noise,
which is more informative than an interval that straddles zero with no scale attached.

Run once, then commit BEFORE any top-up run.  `git log` must place this file's commit strictly before
the first added result file.  The script refuses to write if any of the runs it registers already
exists.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/power_topup/preregistration_power.json"
ANALYSIS = ROOT / "results/paired_inference.json"

GATE_THRESHOLD = 0.20      # unchanged; the cells in scope are the gate-passing ones
MDE_TARGET_PP = 8.0        # the scale the MDE table is read against
FEASIBILITY_CAP = 20       # paired seeds per cell; beyond this the cell is declared unresolvable
ALPHA = 0.05

# The next values in the repeated-digit ladder the existing seeds already follow (42, 123, 456, 789,
# 999, 1011, 1234, 2022, 3033, 4044, 5055). Fixed here so the added seeds are a property of this file
# and not of whoever launches the runner; any seed a cell already ran is skipped.
EXTRA_SEEDS = (6066, 7077, 8088, 9099, 10100, 11111, 12122, 13133)

# ref -> how to run one more seed of this cell, matching where its existing seeds live. The layouts
# differ because these cells were run at different times: the oldest writes both conditions into one
# flat directory, the two v5 cells use h{H}/condition_{C}. cell_matrix keys a Moirai cell by
# (results-subdirectory, size, horizon, n_train), so an added seed MUST land in the same directory as
# the cell's existing seeds or it becomes a separate cell instead of a new seed of this one.
RUN_SPEC = {
    "small_ETTh2_h192": dict(size="small", dataset="ETTh2", horizon=192, n_train=500,
                             results_dir="results/forecasting_finetune_20ep",
                             filename="condition_{C}_h{H}_s{S}.json"),
    "small_ETTm2_h192": dict(size="small", dataset="ETTm2", horizon=192, n_train=1000,
                             results_dir="results/v5_ettm2/h{H}/condition_{C}",
                             filename="condition_{C}_h{H}_s{S}.json"),
    "base_ETTh2_h96": dict(size="base", dataset="ETTh2", horizon=96, n_train=1000,
                           results_dir="results/v5_etth2_base/h{H}/condition_{C}",
                           filename="condition_{C}_h{H}_s{S}.json"),
}


def existing_seeds(ref):
    """The seeds this cell has already run, read from the records rather than listed by hand."""
    import cell_matrix as cm
    for (rel, size, h, n), seeds in cm.moirai_cells().items():
        ds = cm.DATASET_OF.get(rel)
        if ds is None:
            continue
        if f"{size}_{ds}_h{h}" == ref:
            return sorted(s for s, v in seeds.items() if {"B", "D"} <= set(v))
    return []


def main():
    if OUT.exists():
        sys.exit(f"{OUT} already exists -- refusing to overwrite a registration")
    if not ANALYSIS.exists():
        sys.exit("run scripts/paired_inference.py first: the budgets are read from its output")

    cells = json.loads(ANALYSIS.read_text())["cells"]
    scope = [c for c in cells
             if c.get("gate") is not None and c["gate"] >= GATE_THRESHOLD and c.get("d_enc")]
    entries, planned_runs = [], []
    for c in sorted(scope, key=lambda c: c["cell"]):
        e = c["d_enc"]
        decisive = e["q"] is not None and e["q"] == e["q"] and e["q"] < ALPHA
        need, need_adj = e["n_needed"], e["n_needed_adj"]
        feasible = [n for n in (need_adj, need) if n is not None and n <= FEASIBILITY_CAP]
        if decisive:
            action, target, tier = "none (already decisive at BH)", e["n"], None
        elif not feasible:
            action, target, tier = ("none (UNRESOLVABLE: required n exceeds the cap)", e["n"], None)
        else:
            # The adjusted budget first where it fits -- it is the one that can change the paper's
            # call -- and the uncorrected budget otherwise, which buys a narrower interval on a cell
            # expected to stay inconclusive. The tier is recorded per cell so the expectation is
            # attached to the cell it applies to and not only to the prose above.
            tier = "adjusted" if (need_adj is not None and need_adj <= FEASIBILITY_CAP) else "uncorrected"
            action, target = "top up", max(feasible[0], e["n"])
        have = existing_seeds(c["ref"])
        add = [s for s in EXTRA_SEEDS if s not in have][:max(0, target - e["n"])]
        if action == "top up" and len(add) < target - e["n"]:
            sys.exit(f"{c['ref']}: EXTRA_SEEDS has too few unused values for {target - e['n']} more")
        entries.append(dict(
            cell=c["cell"], ref=c["ref"], gate=c["gate"],
            n=e["n"], mean=e["mean"], sd=e["sd"], q=e["q"],
            mde=e["mde"], mde_adj=e["mde_adj"],
            n_needed=e["n_needed"], n_needed_adj=e["n_needed_adj"],
            call=c["call"], action=action, target_n=target, tier=tier,
            expected_to_stay_inconclusive=bool(tier == "uncorrected"),
            seeds_have=have, seeds_to_add=add,
            meets_mde_target=bool(e["mde"] <= MDE_TARGET_PP),
        ))
        if action != "top up":
            continue
        spec = RUN_SPEC.get(c["ref"])
        if spec is None:
            sys.exit(f"{c['ref']} is to be topped up but RUN_SPEC has no entry saying where its "
                     f"existing seeds live")
        for s in add:
            for cond in ("B", "D"):
                d = spec["results_dir"].format(H=spec["horizon"], C=cond)
                f = spec["filename"].format(C=cond, H=spec["horizon"], S=s)
                planned_runs.append(dict(
                    ref=c["ref"], seed=s, condition=cond, path=f"{d}/{f}",
                    cmd=(f"python scripts/finetune_forecasting.py "
                         f"--data-path data/forecasting/{spec['dataset']}.csv "
                         f"--model-size {spec['size']} --horizon {spec['horizon']} "
                         f"--condition {cond} --seed {s} --epochs 20 "
                         f"--max-train-samples {spec['n_train']} --device mps "
                         f"--results-dir {d}")))

    # A registration is only a registration if none of the runs it predicts has happened.
    already = [r["path"] for r in planned_runs if (ROOT / r["path"]).exists()]
    if already:
        sys.exit(f"refusing to register: {len(already)} of the planned runs already exist, "
                 f"e.g. {already[:2]}")

    topped = [e for e in entries if e["action"] == "top up"]
    unres = [e for e in entries if e["action"].startswith("none (UNRESOLVABLE")]
    payload = dict(
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip(),
        analysis_source="results/paired_inference.json",
        alpha=ALPHA, gate_threshold=GATE_THRESHOLD,
        mde_target_pp=MDE_TARGET_PP, feasibility_cap=FEASIBILITY_CAP,
        rule=dict(
            scope="every gate-passing cell with a paired encoder contrast",
            target=("the BH-ADJUSTED required n where that is at most the cap of 20 paired seeds, "
                    "otherwise the UNCORRECTED required n where THAT is at most the cap; both are "
                    "computed from the cell's own observed effect and sd"),
            top_up=("a cell is topped up iff its BH call is inconclusive AND at least one of the two "
                    "budgets is at most the cap; it is then run to exactly that n and no further. The "
                    "per-cell `tier` field says which budget was used, and cells on the uncorrected "
                    "tier carry expected_to_stay_inconclusive=true"),
            unresolvable=("a cell whose required n exceeds the cap is NOT topped up, whatever its "
                          "p-value does, and its required n is published as the finding"),
            stopping=("the seeds to add are listed per cell in this file; the runner adds those and "
                      "stops, so there is no decision left to make once a result is seen"),
        ),
        statements_fixed_in_advance=dict(
            bh_expectation=("the cells on the uncorrected tier are being run to a budget smaller than "
                            "their BH-adjusted requirement, so they are EXPECTED to remain "
                            "inconclusive under the paper's own call; the top-up buys a tighter "
                            "interval there, not a verdict"),
            reporting=("original-n and augmented-n are reported side by side in one table; no "
                       "published number is replaced, and any cell whose call changes gets a "
                       "sentence naming the n at which it changed"),
            sd_not_blind=("every required n is computed from the cell's already-observed sd, so this "
                          "is a stopping rule fixed before more data, not a prospective power "
                          "calculation"),
            exchangeability=("the added seeds run today into the cell's original directory; the "
                             "paper's two-cell reproduction check bounds the drift a re-run "
                             "introduces (MSEs within a few percent, forgetting percentage 63.5%), "
                             "so the augmented n is reported as an augmentation and not as if the "
                             "seeds had all been drawn at one time"),
        ),
        n_cells_in_scope=len(entries), n_topped_up=len(topped), n_unresolvable=len(unres),
        n_runs=len(planned_runs),
        cells=entries, planned_runs=planned_runs,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(entries)} gate-passing cells in scope; {len(topped)} topped up "
          f"({len(planned_runs)} fine-tunes), {len(unres)} declared unresolvable")
    for e in entries:
        print(f"  {e['ref']:20s} gate {e['gate']:+.3f} n={e['n']:2d} mean={e['mean']:+7.2f} "
              f"sd={e['sd']:5.2f} q={e['q']:.3f} need={e['n_needed']} (adj {e['n_needed_adj']})  "
              f"-> {e['action']}" + (f" to n={e['target_n']} adding {e['seeds_to_add']}"
                                     if e["action"] == "top up" else ""))


if __name__ == "__main__":
    main()
