#!/usr/bin/env python
"""STEP 2 of 2 for the third-split confirmation arm: the eight-rung ladder on the train tail.

WHAT THIS IS FOR.  The paper's primary value gate is scored on the SELECTION split.  That is already
disjoint from the windows the B-D outcome is scored on, which is what the round-3 review asked for.
What it does not remove is that the selection split also drove early stopping and checkpoint choice.
This arm removes even that: the gate is scored on the last `--frac` of the TRAIN region, every rung
of the ladder is fitted on the head alone, and the normalisation is head-only, so the windows the
gate is scored on fitted no baseline, selected no checkpoint and carried no outcome.

WHAT IT IS NOT.  It is a confirmation row, not a third headline.  It covers the 21 Moirai cells only
-- the Chronos and TimesFM numerators come from separate evaluators, and this arm is reported as
Moirai-only rather than presented as matrix-wide.  And it cannot make the OUTCOME independent of the
selection split: fine-tuned checkpoints are not retained, so the outcome cannot be re-scored on a
region that drove nothing.  Only the gate side moves here, which is where the circularity was.

WINDOW AGREEMENT IS CHECKED, NOT ASSUMED.  The numerator is produced by a different script
(finetune_forecasting.py --zs-windows traintail), so gate_all_cells.moirai_gates() recomputes the
cut index, the window count and the extended lookback and refuses to report a cell whose geometry
disagrees with what the numerator recorded.  A ratio of two numbers measured on different windows is
the defect this whole appendix exists to document; it is not going to be reintroduced here.

Usage:
    .venv12/bin/python scripts/gate_thirdsplit.py            # after run_thirdsplit_zs.sh
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import gate_all_cells as gac                                   # noqa: E402
import gate_baselines                                          # noqa: E402
import gate_baseline_sensitivity as gbs                         # noqa: E402

BASELINES = ("fitted",) + gate_baselines.NAMES
OUT = ROOT / "results" / "gate_baselines_traintail.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2,
                    help="Fraction of the train region held back. Must match the numerator runs.")
    ap.add_argument("--max-eval", type=int, default=300)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    refs, meta = gac._zs_traintail_refs()
    if not refs:
        sys.exit("no third-split numerators found in results/v49_thirdsplit/ -- "
                 "run scripts/run_thirdsplit_zs.sh first")
    fracs = {m["traintail_frac"] for m in meta.values()}
    if fracs != {a.frac}:
        sys.exit(f"--frac {a.frac} does not match the numerators' {sorted(fracs)}: the denominator "
                 f"would be fitted on a different head than the numerator was scored against")
    print(f"third-split gate: {len(refs)} Moirai cells, tail = last {a.frac:.0%} of train, "
          f"numerator from results/v49_thirdsplit/")

    results = {}
    for b in BASELINES:
        print(f"\n{'=' * 96}\nBASELINE: {b}\n{'=' * 96}")
        gates = gac.moirai_gates(refs, split="traintail", baseline=b, max_eval=a.max_eval,
                                 traintail_meta=meta, traintail_frac=a.frac)
        for k, v in gates.items():
            results.setdefault(k, {})[b] = dict(r2_task=v["r2_task"], zs_test=v["zs_test"],
                                                linear_test=v["linear_test"],
                                                n_windows=v["n_windows"], info=v.get("baseline_info", {}))

    graded = gbs.graded_table(results, list(BASELINES))
    per_rung = {b: sum(1 for k in results if gbs.passes(results, k, b)) for b in BASELINES}

    print(f"\n{'=' * 96}\nTHIRD-SPLIT LADDER\n{'=' * 96}")
    for b in BASELINES:
        print(f"  {b:16s} clears {gac.GATE_THRESHOLD}: {per_rung[b]} of {len(results)}")
    print(f"\n  clears {gac.GATE_THRESHOLD} against ALL admissible rungs: "
          f"{graded['n_clearing_all_admissible']} of {graded['n_scored']}")
    print(f"  clears {gac.GATE_THRESHOLD} against AT LEAST ONE:         "
          f"{graded['n_clearing_any_admissible']} of {graded['n_scored']}")
    union = sorted(k for k, v in graded["cells"].items() if v["clears_any_admissible"])
    print("  union: " + (", ".join(union) if union else "(empty)"))

    payload = dict(split="traintail", traintail_frac=a.frac, arm="moirai",
                   threshold=gac.GATE_THRESHOLD, baselines=list(BASELINES),
                   cells=results, graded=graded, per_rung=per_rung, union=union,
                   window_meta=meta)
    Path(a.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {Path(a.out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
