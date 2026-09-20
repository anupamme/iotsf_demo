#!/usr/bin/env python3
"""Compare a from-scratch re-run against the stored run record it is meant to reproduce.

WHAT THIS CHECKS, AND WHAT IT CANNOT. The committed results/*.json are the sole source for every
number in the paper, so the question a reader is entitled to ask is: do those files still follow from
the code in this repository, or have the two drifted? A re-run answers that. It does NOT answer
whether the numbers are right -- only whether they are what this code produces from this data.

WHY IT IS NOT A PASS/FAIL. A bit-exact match is not achievable and claiming one would be false:

  * DEVICE. The Moirai and TimesFM arms were run on MPS and the Chronos arm on CUDA. Reduction order
    differs between backends, so the same seed gives a slightly different loss trajectory.
  * NON-DETERMINISM. MPS has no deterministic mode equivalent to torch.use_deterministic_algorithms,
    so even same-device re-runs differ in the low-order bits, and early stopping can amplify that
    into a whole-epoch difference in which checkpoint is selected.
  * LIBRARY VERSIONS. torch, uni2ts and transformers have all moved since the earliest cells.

So this prints the SIGNED RELATIVE DEVIATION per field and lets the reader judge. The one thing it
does adjudicate is the qualitative reading: the paper's claims are about the SIGN of forgetting and
of B-D, and a re-run that flips a sign is a reproduction failure in the only sense the paper's
conclusions depend on. That is reported separately from the magnitudes, and it is the line the
Reproducibility Statement should quote.

Run:  .venv12/bin/python scripts/compare_rerun.py --stored A.json --fresh B.json [--label name]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The fields every downstream number is built from. Anything not in this list (timings, parameter
# counts, paths) is deliberately not compared: it does not enter a table.
FIELDS = ["zeroshot_mse", "zeroshot_test_mse", "final_val_mse", "test_mse",
          "final_cka", "final_weight_drift", "forgetting_pct", "forgetting_pct_test",
          "forgetting_pct_val", "best_epoch"]
# Fields whose SIGN carries a claim. A magnitude change of 30% on test_mse is an artefact of device
# and library drift; a sign change on forgetting means the cell now says the opposite thing.
SIGN_FIELDS = ["forgetting_pct", "forgetting_pct_test", "forgetting_pct_val"]
# Protocol fields that must match exactly. If these differ the two runs are not the same experiment
# and comparing their losses is meaningless -- the most common way a "reproduction" quietly lies.
PROTOCOL = ["condition", "seed", "horizon", "epochs", "max_train_samples", "model_size",
            "dataset", "lr", "batch_size", "lookback", "n_test_windows", "test_seed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stored", required=True)
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    if not Path(a.fresh).exists():
        print(f"  {a.label or a.fresh}: fresh run not found -- the re-run did not complete")
        return 2
    s, f = json.load(open(a.stored)), json.load(open(a.fresh))

    print(f"\n  {a.label or Path(a.stored).name}")
    print(f"    stored {Path(a.stored).relative_to(ROOT) if Path(a.stored).is_absolute() else a.stored}")
    print(f"    fresh  {Path(a.fresh).relative_to(ROOT) if Path(a.fresh).is_absolute() else a.fresh}")

    mismatched = [k for k in PROTOCOL if k in s and k in f and s[k] != f[k]]
    if mismatched:
        print("    PROTOCOL MISMATCH -- these are not the same experiment, so the comparison below")
        print("    is not interpretable. Fix the re-run command before reading any deviation.")
        for k in mismatched:
            print(f"      {k:22s} stored {s[k]!r}   fresh {f[k]!r}")
        return 2

    print(f"    {'field':22s}{'stored':>14s}{'fresh':>14s}{'rel dev':>11s}")
    flips = []
    for k in FIELDS:
        if k not in s or k not in f or not isinstance(s[k], (int, float)):
            continue
        rel = (f[k] - s[k]) / abs(s[k]) * 100 if s[k] else float("nan")
        flip = k in SIGN_FIELDS and (s[k] > 0) != (f[k] > 0)
        if flip:
            flips.append(k)
        print(f"    {k:22s}{s[k]:>14.6f}{f[k]:>14.6f}{rel:>10.1f}%"
              + ("   SIGN FLIP" if flip else ""))

    if flips:
        print(f"    REPRODUCTION FAILURE on sign: {', '.join(flips)}. The re-run reverses a claim,")
        print("    not just a magnitude. This must be stated in the paper, not averaged away.")
        return 1
    print("    Every sign-bearing field keeps its sign; deviations are magnitude only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
