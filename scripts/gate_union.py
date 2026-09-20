#!/usr/bin/env python
"""The ladder-union admission set, per split, with its backbone composition.

WHAT THE UNION IS.  A cell is in the union when the pre-trained model clears the 0.20 gate against
AT LEAST ONE admissible rung, where admissible means the rung is not itself beaten by the constant
floor.  That is the MOST PERMISSIVE admissible screen, not a stronger one: `seasonal_naive` can be
admissible and still far weaker than the fitted ridge, so a union cell may LOSE to the ridge
outright (small_ETTh1_h96 on the test split is exactly that -- ridge -0.243, seasonal_naive +0.216).
The union is therefore the widest defensible answer to "which cells might have something to
preserve", and the companion number -- how many clear 0.20 against EVERY admissible rung -- is the
conservative bound that stays the headline.  Reporting one without the other would misread either way.

WHY IT IS RECOMPUTED HERE.  gate_baseline_sensitivity.py already writes a `graded` block, and this
script does not trust it: it rebuilds admissibility and both counts from the per-rung `linear_test`
and `r2_task` values and then asserts agreement.  The ladder's own grading code and the paper's
checker apply the same rule in two places, which is how a rule silently drifts between them.

Usage:
    .venv12/bin/python scripts/gate_union.py                              # test side
    .venv12/bin/python scripts/gate_union.py --in results/gate_baselines_val.json
    .venv12/bin/python scripts/gate_union.py --compare results/gate_baselines_val.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def arm_of(key):
    if key.startswith(("small_", "base_", "large_")):
        return "moirai"
    if key.startswith("chronos_"):
        return "chronos"
    if key.startswith("timesfm_"):
        return "timesfm"
    return key                                  # `ili` is its own external case study


def grade(path):
    """Rebuild the admission sets from the per-rung numbers, then check the stored grading."""
    d = json.load(open(path))
    cells, floor, T = d["cells"], d["floor"], d["gate_threshold"]
    out = {}
    for k, per in cells.items():
        if floor not in per or per[floor].get("linear_test") is None:
            print(f"  {k:24s} SKIPPED -- no {floor} floor to judge admissibility against")
            continue
        fl = per[floor]["linear_test"]
        adm = {b: e for b, e in per.items()
               if b != floor and e.get("r2_task") is not None and e.get("linear_test") is not None
               and e["linear_test"] <= fl}
        if not adm:
            print(f"  {k:24s} SKIPPED -- no admissible rung")
            continue
        # The strongest admissible rung is the one with the LOWEST loss, and it gives the lowest
        # r2_task: a weaker rival inflates the pre-trained advantage, so "best" here means the
        # hardest baseline to beat, not the most flattering number.
        best = min(adm, key=lambda b: adm[b]["linear_test"])
        weakest = max(adm, key=lambda b: adm[b]["linear_test"])
        out[k] = dict(arm=arm_of(k), n_admissible=len(adm), admissible=sorted(adm),
                      r2_best=adm[best]["r2_task"], best_baseline=best,
                      r2_any=adm[weakest]["r2_task"], weakest_baseline=weakest,
                      r2_fitted=per.get("fitted", {}).get("r2_task"),
                      clears_all=all(e["r2_task"] >= T for e in adm.values()),
                      clears_any=any(e["r2_task"] >= T for e in adm.values()))

    g = d.get("graded")
    if g:
        mine_any = sum(1 for v in out.values() if v["clears_any"])
        mine_all = sum(1 for v in out.values() if v["clears_all"])
        bad = [(n, a, b) for n, a, b in
               (("clears_any", mine_any, g["n_clearing_any_admissible"]),
                ("clears_all", mine_all, g["n_clearing_all_admissible"]),
                ("n_scored", len(out), g["n_scored"])) if a != b]
        if bad:
            sys.exit("recomputed grading disagrees with the stored `graded` block: " +
                     "; ".join(f"{n}: mine {a} vs stored {b}" for n, a, b in bad))
    return d, out, T


def report(label, d, out, T):
    union = sorted(k for k, v in out.items() if v["clears_any"])
    strict = sorted(k for k, v in out.items() if v["clears_all"])
    arms = {}
    for k in union:
        arms[out[k]["arm"]] = arms.get(out[k]["arm"], 0) + 1
    print(f"\n{'=' * 96}\n{label}  (split={d.get('split')}, threshold={T})\n{'=' * 96}")
    print(f"  cells scored:                        {len(out)}")
    print(f"  clears {T} vs AT LEAST ONE admissible: {len(union)}  "
          f"({', '.join(f'{a} {n}' for a, n in sorted(arms.items())) or 'none'})")
    print(f"  clears {T} vs EVERY admissible rung:   {len(strict)}"
          f"{'  (' + ', '.join(strict) + ')' if strict else ''}")
    # The cells the union admits but the paper's primary ridge does not: these are exactly the ones
    # a reader will challenge, so they are named rather than folded into a count.
    loose = [k for k in union if (out[k]["r2_fitted"] or -9) < T]
    print(f"  in the union but FAILING the fitted ridge: {len(loose)}")
    for k in loose:
        v = out[k]
        print(f"      {k:24s} ridge {v['r2_fitted']:+.3f}   "
              f"admitted by {v['weakest_baseline']} {v['r2_any']:+.3f}")
    return dict(union=union, strict=strict, arms=arms, loose=loose, cells=out,
                n_scored=len(out), threshold=T, split=d.get("split"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "results/gate_baselines.json"))
    ap.add_argument("--compare", default=None,
                    help="A second ladder file to diff the admission set against.")
    ap.add_argument("--out", default=None, help="Write the admission sets as JSON.")
    a = ap.parse_args()

    d, out, T = grade(a.inp)
    A = report(Path(a.inp).name, d, out, T)
    payload = {Path(a.inp).name: A}

    if a.compare:
        d2, out2, T2 = grade(a.compare)
        B = report(Path(a.compare).name, d2, out2, T2)
        payload[Path(a.compare).name] = B
        sa, sb = set(A["union"]), set(B["union"])
        print(f"\n{'=' * 96}\nADMISSION SET DIFF\n{'=' * 96}")
        print(f"  both:           {len(sa & sb)}")
        print(f"  only {Path(a.inp).name}: {sorted(sa - sb) or '(none)'}")
        print(f"  only {Path(a.compare).name}: {sorted(sb - sa) or '(none)'}")
        payload["diff"] = dict(both=sorted(sa & sb), only_a=sorted(sa - sb), only_b=sorted(sb - sa))

    if a.out:
        Path(a.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
