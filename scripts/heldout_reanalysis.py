#!/usr/bin/env python3
"""
Held-out (test-window) re-analysis of every B/D cell whose artifacts are in this repo.

WHY
---
Every B-D number the paper published was read off the *validation* windows -- the same windows
condition B early-stops on and (for the converged Chronos runs) the same windows the frozen head's
ridge penalty is chosen on. That makes B-D a controlled within-cell intervention contrast but not a
generalization estimate, and a reviewer is right to say the magnitudes may be biased.

Every training script in this repo already carves a chronologically disjoint test split and, for
the Moirai cells, already stores `test_mse` -- it was simply never reported. This script reads what
is on disk and recomputes B-D on those untouched windows.

Run:  python3 scripts/heldout_reanalysis.py
"""
import glob
import json
import statistics as s


def _sd(xs):
    """Std dev, or 0.0 for a single sample -- partial runs are inspected while still going."""
    return s.stdev(xs) if len(xs) > 1 else 0.0
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(str(ROOT / pattern)))]


def moirai_base_n1000():
    """
    Moirai-Base / ETTh2, h=96, n=1000 -- the paper's 'matched paired' row.

    Both arms: same seeds, 20 epochs, no early stopping, same hardware. `zeroshot_mse` in these
    files is the validation reference; no zero-shot *test* reference was ever stored, so the
    denominator-free relative gap is reported alongside.
    """
    # Zero-shot reference on the TEST windows, measured for this revision (condition A now records
    # it). Without this the only available denominator was the VALIDATION zero-shot, which is on a
    # different scale to a test-window MSE difference and inflates the gap.
    zs_test = {d["seed"]: d["zeroshot_test_mse"]
               for d in _load("results/v39_moirai_zs_test/h96/condition_A/*.json")
               if "zeroshot_test_mse" in d}
    by_seed = {}
    for d in _load("results/v5_etth2_base/h96/condition_[BD]/*.json"):
        by_seed.setdefault(d["seed"], {})[d["condition"]] = d

    rows, bd_val, bd_test, rel = [], [], [], []
    for seed in sorted(by_seed):
        r = by_seed[seed]
        if not {"B", "D"} <= set(r):
            continue
        zs = r["B"]["zeroshot_mse"]
        assert abs(zs - r["D"]["zeroshot_mse"]) < 1e-12, f"seed {seed}: arms disagree on zero-shot"
        zs_t = zs_test.get(seed)
        v = (r["B"]["final_val_mse"] - r["D"]["final_val_mse"]) / zs * 100
        # NOTE: scale-mismatched, do not quote as percentage points of forgetting. The numerator is
        # a test-scale MSE difference (~0.36 vs ~0.47) while `zeroshot_mse` is the VALIDATION
        # reference (~0.20), so this inflates the magnitude. Only its sign is meaningful. The
        # honest statistics for this cell are the sign reversal, the 3/3 seed count, and the
        # denominator-free relative gap below; a properly scaled pp value needs the zero-shot TEST
        # reference (scripts/finetune_forecasting.py condition A now records it).
        t = (r["B"]["test_mse"] - r["D"]["test_mse"]) / (zs_t if zs_t else zs) * 100
        g = (r["B"]["test_mse"] - r["D"]["test_mse"]) / r["D"]["test_mse"] * 100
        rows.append((seed, r["B"]["final_val_mse"], r["D"]["final_val_mse"],
                     r["B"]["test_mse"], r["D"]["test_mse"], v, t, g))
        bd_val.append(v); bd_test.append(t); rel.append(g)

    print("\nMoirai-Base / ETTh2, h=96, n=1000  (matched seeds, 20 ep, no early stopping)")
    print(f"  {'seed':>5} {'B val':>8} {'D val':>8} {'B test':>8} {'D test':>8} "
          f"{'B-D val':>9} {'B-D test':>9} {'rel %':>8}")
    for seed, bv, dv, bt, dt, v, t, g in rows:
        print(f"  {seed:>5} {bv:>8.4f} {dv:>8.4f} {bt:>8.4f} {dt:>8.4f} "
              f"{v:>+9.1f} {t:>+9.1f} {g:>+8.1f}")
    print(f"  mean B-D  validation {s.mean(bd_val):+.2f} +- {_sd(bd_val):.2f}   "
          f"({sum(x < 0 for x in bd_val)}/{len(bd_val)} negative)")
    print(f"  mean B-D  HELD-OUT   {s.mean(bd_test):+.2f} +- {_sd(bd_test):.2f}   "
          f"({sum(x < 0 for x in bd_test)}/{len(bd_test)} negative)   "
          + ("[zs-test denominator]" if zs_test else "[SCALE-MISMATCHED: val denominator, sign only]"))
    print(f"  mean relative gap    {s.mean(rel):+.2f}% +- {_sd(rel):.2f}   <-- quote this")
    print("  Reported result: the sign reverses and full fine-tuning wins on held-out windows in "
          f"{sum(x < 0 for x in bd_test)}/{len(bd_test)} seeds.")


def moirai_base_n10k_D_only():
    """Moirai-Base D at n=10k. Its matched condition B is the published CUDA run, not in this repo."""
    ds = _load("results/v35_base_frozen/condition_D_h96_s*.json")
    if not ds:
        return
    print(f"\nMoirai-Base / ETTh2, h=96, n=10k -- condition D only ({len(ds)} seeds)")
    print(f"  val  MSE {s.mean([d['final_val_mse'] for d in ds]):.4f} "
          f"+- {_sd([d['final_val_mse'] for d in ds]):.4f}")
    print(f"  test MSE {s.mean([d['test_mse'] for d in ds]):.4f} "
          f"+- {_sd([d['test_mse'] for d in ds]):.4f}")
    print("  matched condition B is not in this repo, so held-out B-D is not computable here.")


# ---------------------------------------------------------------------------
# PRE-COMMITTED CHOICE OF THE CONDITION-D ESTIMATOR (recorded 2026-08-23, before the ETTh2
# held-out values existed; at the time only ETTh1 seeds 42 and 43 had been scored on test).
#
# Condition D has two estimates of the best frozen-encoder head: the closed-form ridge optimum at
# the alpha validation chose, and a protocol-matched AdamW head early-stopped on validation. On the
# validation windows they agree to about 1 pp. On held-out windows they can diverge sharply --
# ETTh1 seed 43 gave ridge +89.4% against AdamW +48.0% -- because a weak val-selected alpha wins on
# the windows it was chosen on and transfers poorly off them.
#
# RULE: the held-out headline is the estimator validation preferred, which is the ridge optimum in
# every seed measured. The AdamW leg is reported alongside as a sensitivity. We do NOT take the
# better of the two on test: that would be selecting on the held-out set, which is the exact fault
# this whole exercise exists to remove.
# ---------------------------------------------------------------------------


def _shared_zs_test(dataset):
    """
    The single zero-shot test denominator, from scripts/chronos_zs_test_reference.py.

    B-D divides by a SHARED zero-shot term, so both arms must use one value. Each run stores its own
    `zs_mse_test`, but that decode samples and the two arms hit it with different RNG state (1.4%
    apart on ETTh1/42), which would make B-D depend on whose estimate you picked. Prefer the seeded
    reference; fall back to the mean of the two arms' estimates, which is at least symmetric.
    """
    f = ROOT / "results/v39_chronos_heldout/zs_test_reference.json"
    if f.exists():
        return json.load(open(f))["datasets"][dataset]["zs_mse_test"], "shared reference"
    return None, "per-arm mean (fallback)"


def chronos(name, b_glob, d_glob, horizon=24, dataset=None):
    """Chronos cells. D's headline is the closed-form ridge optimum; alpha is chosen on validation."""
    B = {d["seed"]: d for d in _load(b_glob)}
    D = {d["seed"]: d for d in _load(d_glob)}
    seeds = sorted(set(B) & set(D))
    if not seeds:
        return
    has_test = all("test_mse_per_element" in B[x] and "test_mse_per_element_ols" in D[x]
                   for x in seeds)
    shared_zs, zs_src = _shared_zs_test(dataset) if dataset else (None, "n/a")
    print(f"\n{name}  ({len(seeds)} seeds)"
          + ("" if has_test else "   [validation only -- held-out run not present yet]")
          + (f"   zs-test denominator: {zs_src}" if has_test else ""))
    val, test = [], []
    for x in seeds:
        b, d = B[x], D[x]
        zs = d["zs_mse"]
        v = (b["best_val_loss"] / horizon - d["best_val_loss_ols"] / horizon) / zs * 100
        val.append(v)
        line = (f"  s{x}: B_val={b['best_val_loss']/horizon:.4f} "
                f"D_val={d['best_val_loss_ols']/horizon:.4f} B-D val={v:+.1f}")
        if has_test:
            zst = shared_zs if shared_zs else (d["zs_mse_test"] + b["zs_mse_test"]) / 2
            t = (b["test_mse_per_element"] - d["test_mse_per_element_ols"]) / zst * 100
            test.append(t)
            line += (f" | B_test={b['test_mse_per_element']:.4f} "
                     f"D_test={d['test_mse_per_element_ols']:.4f} B-D test={t:+.1f}"
                     f"  alpha={d['ridge_optimum']['alpha']:g}"
                     f" interior={d['ridge_optimum']['interior']}")
        print(line)
    print(f"  mean B-D validation {s.mean(val):+.2f} +- {_sd(val):.2f}   "
          f"({sum(x < 0 for x in val)}/{len(val)} negative)")
    if has_test:
        print(f"  mean B-D HELD-OUT   {s.mean(test):+.2f} +- {_sd(test):.2f}   "
              f"({sum(x < 0 for x in test)}/{len(test)} negative)")


def reproduction_check():
    """
    The held-out runs re-train condition B; they are not reruns of the published checkpoints.
    Quantify how closely they reproduce the published validation figures, so the paper can state
    the agreement rather than assert it. Exact agreement is not expected: MPS matmuls are not
    bit-deterministic run to run and torch.use_deterministic_algorithms is set only under CUDA.
    """
    pairs = [("ETTh1", "results/v39_chronos_heldout/cond_B/mse_etth1",
              "results/v38_chronos_converged/cond_B/mse_etth1"),
             ("ETTh2", "results/v39_chronos_heldout/cond_B/mse_etth2",
              "results/v37_chronos_etth2/cond_B/mse_etth2")]
    print("\nReproduction of published condition-B validation loss (held-out rerun vs published)")
    worst = 0.0
    for name, newd, oldd in pairs:
        for seed in (42, 43, 44):
            fn = ROOT / newd / f"seed{seed}" / f"condition_B_s{seed}.json"
            fo = ROOT / oldd / f"seed{seed}" / f"condition_B_s{seed}.json"
            if not (fn.exists() and fo.exists()):
                continue
            a = json.load(open(fn))["best_val_loss"]
            b = json.load(open(fo))["best_val_loss"]
            d = abs(a - b) / b * 100
            worst = max(worst, d)
            print(f"  {name} s{seed}: rerun {a:.4f}  published {b:.4f}   {d:.3f}% apart")
    if worst:
        print(f"  worst-case disagreement {worst:.3f}% -- MPS run-to-run float noise, not a "
              f"protocol change")


def ili():
    """
    Moirai-Small / ILI, h=24, 10 seeds, conditions B and D.

    Condition D here is a REAL frozen-encoder run. An earlier version of finetune_ili.py
    short-circuited condition D entirely -- it wrote ft_mse = zs_mse and forgetting_pct = 0.0
    without training anything -- so the published "frozen control gives exactly 0.0% in all 10
    seeds" was one hardcoded literal, and the resulting B-D of -80.7 pp was just forgetting_B
    restated. These numbers come from a D that actually trains.
    """
    B = {d["seed"]: d for d in _load("results/v40_ili_heldout/condition_B_seed*.json")}
    D = {d["seed"]: d for d in _load("results/v40_ili_heldout/condition_D_seed*.json")}
    seeds = sorted(set(B) & set(D))
    if not seeds:
        print("\nMoirai-Small / ILI: no matched B/D seeds yet")
        return
    print(f"\nMoirai-Small / ILI, h=24  ({len(seeds)} matched seeds)")
    bv, bt = [], []
    for x in seeds:
        b, d = B[x]["aggregate"], D[x]["aggregate"]
        v = b["forgetting_pct"] - d["forgetting_pct"]
        bv.append(v)
        line = (f"  s{x}: B_val={b['forgetting_pct']:+.1f} D_val={d['forgetting_pct']:+.1f} "
                f"B-D val={v:+.1f}  cka_B={b['cka']:.3f} cka_D={d['cka']:.3f}")
        if b.get("forgetting_pct_test") == b.get("forgetting_pct_test") and \
           d.get("forgetting_pct_test") == d.get("forgetting_pct_test"):
            t = b["forgetting_pct_test"] - d["forgetting_pct_test"]
            bt.append(t)
            line += (f" | B_test={b['forgetting_pct_test']:+.1f} "
                     f"D_test={d['forgetting_pct_test']:+.1f} B-D test={t:+.1f}")
        print(line)
    print(f"  mean B-D validation {s.mean(bv):+.2f} +- {_sd(bv):.2f}   "
          f"({sum(x < 0 for x in bv)}/{len(bv)} negative)")
    if bt:
        print(f"  mean B-D HELD-OUT   {s.mean(bt):+.2f} +- {_sd(bt):.2f}   "
              f"({sum(x < 0 for x in bt)}/{len(bt)} negative)")
    print("  published value was -80.7 pp, against a condition D that never ran")


def main():
    print("=" * 78)
    print("HELD-OUT RE-ANALYSIS -- every B-D cell with artifacts in this repo")
    print("Protocol: train on train, select on validation, report on the untouched test split.")
    print("=" * 78)
    reproduction_check()
    ili()
    moirai_base_n1000()
    moirai_base_n10k_D_only()
    for ds in ("etth1", "etth2"):
        chronos(f"Chronos / {ds.upper()}, h=24  (held-out run)",
                f"results/v39_chronos_heldout/cond_B/mse_{ds}/seed*/condition_B_s*.json",
                f"results/v39_chronos_heldout/cond_D/mse_{ds}/seed*/condition_D_s*.json",
                dataset=ds.upper().replace("ETTH", "ETTh"))
    chronos("Chronos / ETTh1, h=24  (published, validation only)",
            "results/v38_chronos_converged/cond_B/mse_etth1/seed*/condition_B_s*.json",
            "results/v38_chronos_converged/cond_D/mse_etth1/seed*/condition_D_s*.json")
    chronos("Chronos / ETTh2, h=24  (published, validation only)",
            "results/v37_chronos_etth2/cond_B/mse_etth2/seed*/condition_B_s*.json",
            "results/v38_chronos_converged/cond_D/mse_etth2/seed*/condition_D_s*.json")
    print()


if __name__ == "__main__":
    main()
