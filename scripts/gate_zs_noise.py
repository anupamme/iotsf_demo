#!/usr/bin/env python3
"""How much of R2_task is the evaluator's own sampling noise, and does any gate call turn on it?

WHY THIS EXISTS
---------------
The gate is R2_task = 1 - MSE_ZS / MSE_Linear, and its numerator is not a deterministic quantity.
finetune_forecasting.py scores Moirai as the median of 20 sampled paths (:445, num_samples=20) drawn
from the global RNG, which it seeds per run with --seed (:743). The WINDOWS are deterministic; the
forecast on them is not. So two runs of one cell record two different `zeroshot_mse` values, and
every gate value in the paper is one draw from a distribution whose width nobody had measured.

That mattered concretely. Until 2026-09-25 gate_all_cells._ZS_VAL_REGISTERED globbed `condition_*`
under results/v47_prospective, which averaged batch 1's 48 outcome records into denominators the
registration had fixed from one condition_A record each -- moving a pre-registered predictor after
its outcomes existed. The repair restricted those roots to condition_A. What made the defect hard to
see is exactly what this script quantifies: the shift was small (~1e-3 in R2_task), no cell changed
its 0.20 call, and so nothing downstream failed. "Small enough that nothing caught it" is a statement
about the noise floor, and a paper that reports a thresholded screen owes its reader that floor
rather than the assurance that it was checked.

WHAT IT MEASURES, per cell, three ways:
  * r2_registered   -- the paper's value: 1 - mean(registered condition_A refs)/linear, read back
                       from the cached gate and re-derived here as a cross-check.
  * r2_rep_range    -- the spread of R2_task across INDIVIDUAL replicate measurements of the same
                       zero-shot checkpoint. This is the direct read of evaluator noise: every
                       replicate scores one frozen checkpoint on one deterministic window set, so
                       the only thing that varies is the sampling.
  * delta_r2        -- |r2 using all replicates - r2_registered|, i.e. the magnitude of the v47
                       contamination described above, computed for every cell rather than only the
                       ones it happened to move. The pre-repair cache no longer exists in the tree,
                       so the count of cells it actually shifted is NOT derivable here and is not
                       claimed; what is derivable, and is reported, is how many admissions the
                       contaminated denominator would change (zero).
Then the question the screen actually depends on: is each cell's distance from the 0.20 operating
point large against the noise ON THAT CELL? Per cell, and that is not a pedantic distinction. The
largest sampling effect anywhere (2.4e-2, base_ETTh2_h96) is larger than the tightest margin anywhere
(0.0109, base_ETTm2_h192), so pairing the global maximum with the tightest margin says the gate sits
inside its own noise floor. It does not: those are two different cells, and base_ETTh2_h96 sits 0.58
from the operating point, where noise of any size cannot move a call. The decisive statistic is the
MINIMUM over cells of margin/own-noise, which is 9.2x. The global maximum is still reported, because
it bounds how large the effect ever gets and belongs in the appendix beside the ratio.

TIER A. Reads only results/**.json -- the cached `linear_test` denominators and the per-run
`zeroshot_mse` values, all committed. It never refits a baseline, so it needs neither the CSVs nor a
GPU, and it cannot drift from the caches it audits.

WHAT IT ASSERTS, and why these and not a tolerance:
  (1) restricting the replicate collection to the registered roots and condition_A reproduces
      gate_all_cells._zs_val_refs() exactly, for every cell. That is the proof that the key mapping
      below is the paper's own and not a lookalike -- without it this script could measure the noise
      of a differently-grouped set of files and agree with nothing.
  (2) no cell's 0.20 call flips under ANY single replicate substitution. This is the claim the
      appendix makes, so it fails here rather than shipping. Asserting a numeric tolerance instead
      would be asserting the conclusion: the point is that the calls are robust, and if one ever is
      not, the honest output is a failure naming that cell.

Run:  .venv12/bin/python scripts/gate_zs_noise.py
      .venv12/bin/python scripts/gate_zs_noise.py --quiet    # write the JSON, print the summary only
"""
import argparse
import glob
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gate_all_cells as G                                             # noqa: E402

OUT_PATH = ROOT / "results/gate_zs_noise.json"

# A Moirai cell directory: <size>_<dataset>_h<horizon>. Used to key a record to its cell by walking
# its parents, which handles both record layouts at once -- results/v41_zs_test/<cell>/<file>.json and
# results/v43_moirai_matrix/<cell>/condition_X/<file>.json -- and, more usefully, ignores directories
# that are not Moirai cells at all. results/v57_prospective3 carries Chronos-arm records under names
# like Electricity_h48/ alongside the Moirai ones; those have no size prefix and so never match.
CELL_DIR = re.compile(r"^(?:small|base|large)_[A-Za-z0-9]+_h\d+$")

# Which registration a cell's frozen denominator comes from. Read off the root its condition_A record
# lives under, not hardcoded per cell, so a cell added to a batch lands in the right group by
# construction. The retrospective matrix has no registration: its denominators pool every condition
# on purpose (see _ZS_VAL_RETROSPECTIVE), which is why "delta" is meaningless there and is reported
# as the replicate range only.
BATCH_OF_ROOT = {"v47_prospective": "prospective batch 1",
                 "v48_prospective2": "prospective batch 2",
                 "v56_pool3": "prospective batch 3",
                 "v41_zs_test": "retrospective",
                 "v43_moirai_matrix": "retrospective",
                 "v39_moirai_zs_test": "retrospective"}


def cell_of(path):
    """Cell key for a record path, or None if it does not belong to a Moirai cell."""
    for parent in Path(path).parents:
        if CELL_DIR.match(parent.name):
            return parent.name
    # The paper's one hardcoded exception, carried over verbatim from _zs_val_refs(): this root
    # predates the <cell>/ layout and stores base_ETTh2_h96 under h96/.
    if "v39_moirai_zs_test" in Path(path).parts:
        return "base_ETTh2_h96"
    return None


def replicates():
    """cell -> {"registered": [...], "extra": [...]} of zeroshot_mse values, with their file paths.

    "registered" is what the paper's denominator is the mean of. "extra" is every OTHER committed
    measurement of the same cell's zero-shot MSE: the outcome runs' re-measurements, and batch 3's
    post-registration condition_A top-up. Both are measurements of one frozen checkpoint on one
    deterministic window set, so their spread is sampling noise and nothing else.
    """
    reg, extra = defaultdict(list), defaultdict(list)
    registered_files = set()
    for _group, pats in (("retrospective", G._ZS_VAL_RETROSPECTIVE),
                         ("registered", G._ZS_VAL_REGISTERED)):
        for pat, _depth in pats:
            for f in sorted(glob.glob(str(ROOT / pat))):
                registered_files.add(f)
    for f in sorted(glob.glob(str(ROOT / "results/v39_moirai_zs_test/h96/condition_*/*.json"))):
        registered_files.add(f)

    for f in sorted(glob.glob(str(ROOT / "results/**/*.json"), recursive=True)):
        key = cell_of(f)
        if key is None:
            continue
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:
            continue
        zs = d.get("zeroshot_mse")
        if not isinstance(zs, float):
            continue
        (reg if f in registered_files else extra)[key].append((zs, str(Path(f).relative_to(ROOT))))
    return reg, extra


def batch_of(paths):
    """Which registration group a cell belongs to, from the roots its registered records live under."""
    roots = {p.split("/")[1] for p in paths}
    groups = {BATCH_OF_ROOT[r] for r in roots if r in BATCH_OF_ROOT}
    if len(groups) == 1:
        return groups.pop()
    # A cell whose registered refs straddle two roots. Batch 3 pooled some batch-2 cells, so its
    # denominator can legitimately come from v48_prospective2; report the LATEST registration, since
    # that is the one whose prediction the cell was scored against.
    order = ["retrospective", "prospective batch 1", "prospective batch 2", "prospective batch 3"]
    return sorted(groups, key=order.index)[-1] if groups else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="skip the per-cell table")
    a = ap.parse_args()

    caches = {}
    for f, tag in (("results/gate_val_side.json", None),
                   ("results/gate_val_side_prospective3.json", 3)):
        p = ROOT / f
        if not p.exists():
            sys.exit(f"missing {f}: run scripts/gate_all_cells.py --split val first")
        for k, v in json.loads(p.read_text()).items():
            if v.get("arm") != "moirai":
                continue
            # Batch 3 is cached separately and may re-score a cell the retrospective grid also holds.
            # Keep both under distinct keys rather than letting one overwrite the other silently.
            caches[k if tag is None else f"{k} (batch 3)"] = dict(v, _ref=k)

    reg, extra = replicates()

    # ASSERT (1): our grouping IS the paper's. _zs_val_refs() is the function that computes the
    # denominators the gate cache was written from, so agreeing with it to floating-point equality is
    # the only evidence that the CELL_DIR walk above reproduces its two depth conventions.
    paper_refs = G._zs_val_refs()
    ours = {k: st.mean(v for v, _ in vals) for k, vals in reg.items()}
    for k, v in paper_refs.items():
        assert k in ours, f"{k}: _zs_val_refs() has a denominator this script found no records for"
        assert abs(ours[k] - v) < 1e-12, \
            f"{k}: regrouped registered refs give {ours[k]!r}, _zs_val_refs() gives {v!r}"

    cells, flips = {}, []
    for key, c in sorted(caches.items()):
        ref, lin, r2_cached = c["_ref"], c["linear_test"], c["r2_task"]
        if ref not in reg:
            continue
        r2 = lambda zs: 1 - zs / lin                                   # noqa: E731
        reg_vals = [v for v, _ in reg[ref]]
        all_vals = reg_vals + [v for v, _ in extra.get(ref, [])]
        r2_reg = r2(st.mean(reg_vals))
        assert abs(r2_reg - r2_cached) < 1e-9, \
            f"{key}: re-derived R2_task {r2_reg!r} != cached {r2_cached!r}; the cache is stale"
        per_rep = sorted(r2(v) for v in all_vals)
        # ASSERT (2), collected rather than raised per cell so the failure names every affected cell
        # at once. A call "flips" if the registered value and some single replicate fall on opposite
        # sides of the operating point -- which is what it would mean for a gate decision to be a
        # statement about the evaluator's RNG rather than about the checkpoint.
        if any((p >= G.GATE_THRESHOLD) != (r2_reg >= G.GATE_THRESHOLD) for p in per_rep):
            flips.append(key)
        # `noise_own` and `ratio_own` are the statistic the robustness claim actually rests on, and
        # getting this pairing wrong is easy: comparing the TIGHTEST margin anywhere against the
        # LARGEST noise anywhere gives 0.0109 against 2.4e-2 and reads as a gate inside its own noise
        # floor. It is not, because those are different cells -- the 2.4e-2 belongs to base_ETTh2_h96,
        # which sits 0.58 from the operating point, where noise of any size cannot move a call.
        # Sampling noise on one cell can only threaten THAT cell's call, so the margin and the noise
        # have to be taken per cell and the minimum ratio reported. Both statistics are kept: the
        # global maximum bounds how large the effect ever gets, the per-cell minimum ratio decides
        # whether it ever matters.
        noise_own = max(abs(r2(st.mean(all_vals)) - r2_reg), per_rep[-1] - per_rep[0])
        cells[key] = dict(
            denominator_registered_in=batch_of([p for _, p in reg[ref]]),
            n_registered=len(reg_vals), n_extra=len(all_vals) - len(reg_vals),
            linear=lin, r2_registered=r2_reg,
            r2_all_replicates=r2(st.mean(all_vals)),
            delta_r2=abs(r2(st.mean(all_vals)) - r2_reg),
            r2_rep_lo=per_rep[0], r2_rep_hi=per_rep[-1], r2_rep_range=per_rep[-1] - per_rep[0],
            noise_own=noise_own,
            ratio_own=abs(r2_reg - G.GATE_THRESHOLD) / noise_own if noise_own else float("inf"),
            margin=abs(r2_reg - G.GATE_THRESHOLD), passes=r2_reg >= G.GATE_THRESHOLD)

    assert not flips, ("the 0.20 call is sensitive to the evaluator's sampling on: "
                       + ", ".join(flips) + ". The appendix's robustness claim is false as written.")

    # Separately: would the CONTAMINATED denominator -- the all-replicate mean, which is what the v47
    # pattern computed before the 2026-09-25 repair -- have changed any admission? The audit trail
    # states it would not, so the statement is derived here rather than asserted from the pre-repair
    # cache, which no longer exists in the tree. Reported as a count, not a bare boolean, so "none"
    # carries its denominator.
    contaminated = [k for k, v in cells.items()
                    if (v["r2_all_replicates"] >= G.GATE_THRESHOLD) != v["passes"]]

    by_batch = {}
    for b in sorted({c["denominator_registered_in"] for c in cells.values()}):
        grp = {k: v for k, v in cells.items() if v["denominator_registered_in"] == b}
        worst_d = max(grp, key=lambda k: grp[k]["delta_r2"])
        worst_r = max(grp, key=lambda k: grp[k]["r2_rep_range"])
        by_batch[b] = dict(n_cells=len(grp),
                           max_delta_r2=grp[worst_d]["delta_r2"], max_delta_cell=worst_d,
                           max_rep_range=grp[worst_r]["r2_rep_range"], max_range_cell=worst_r,
                           n_with_replicates=sum(v["n_extra"] > 0 for v in grp.values()))

    # The two tightest margins. Separate because they are different cells and answer different
    # questions: the overall one bounds the retrospective matrix's 7-of-31 count, the pre-registered
    # one bounds the prospective arm, and quoting only the larger of the two would be picking.
    tight = min(cells, key=lambda k: cells[k]["margin"])
    pro = {k: v for k, v in cells.items()
           if v["denominator_registered_in"].startswith("prospective")}
    tight_pro = min(pro, key=lambda k: pro[k]["margin"]) if pro else None
    max_noise = max(v["noise_own"] for v in cells.values())
    worst_ratio = min(cells, key=lambda k: cells[k]["ratio_own"])

    out = dict(
        generated_by="scripts/gate_zs_noise.py",
        gate_threshold=G.GATE_THRESHOLD,
        n_cells=len(cells), num_samples=20,
        max_noise_any_cell=max_noise,
        min_ratio_any_cell=cells[worst_ratio]["ratio_own"], min_ratio_cell=worst_ratio,
        n_calls_changed_by_contaminated_denominator=len(contaminated),
        cells_changed_by_contaminated_denominator=sorted(contaminated),
        n_zs_measurements=sum(v["n_registered"] + v["n_extra"] for v in cells.values()),
        tightest=dict(cell=tight, margin=cells[tight]["margin"],
                      r2=cells[tight]["r2_registered"],
                      registered_in=cells[tight]["denominator_registered_in"],
                      noise_own=cells[tight]["noise_own"], ratio_own=cells[tight]["ratio_own"],
                      ratio_over_max_noise=cells[tight]["margin"] / max_noise),
        tightest_prospective=(None if tight_pro is None else
                              dict(cell=tight_pro, margin=pro[tight_pro]["margin"],
                                   r2=pro[tight_pro]["r2_registered"],
                                   registered_in=pro[tight_pro]["denominator_registered_in"],
                                   noise_own=pro[tight_pro]["noise_own"],
                                   ratio_own=pro[tight_pro]["ratio_own"])),
        by_batch=by_batch, cells=cells)
    OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    if not a.quiet:
        print(f"{'cell':34s} {'denominator from':22s} {'reps':>5s} {'R2':>9s} {'delta':>9s}"
              f" {'range':>9s} {'margin':>8s} {'ratio':>7s}")
        for k, v in sorted(cells.items(), key=lambda kv: kv[1]["ratio_own"]):
            print(f"{k:34s} {v['denominator_registered_in']:22s}"
                  f" {v['n_registered']}+{v['n_extra']:<3d}"
                  f" {v['r2_registered']:>9.4f} {v['delta_r2']:>9.1e} {v['r2_rep_range']:>9.1e}"
                  f" {v['margin']:>8.4f} {v['ratio_own']:>7.1f}")
    print()
    for b, v in by_batch.items():
        print(f"  {b:22s} {v['n_cells']:2d} cells, {v['n_with_replicates']} with replicates: "
              f"max |delta R2_task| {v['max_delta_r2']:.1e} ({v['max_delta_cell']}), "
              f"max replicate range {v['max_rep_range']:.1e} ({v['max_range_cell']})")
    print(f"\n  largest sampling effect on any cell: {max_noise:.1e} "
          f"({max(cells, key=lambda k: cells[k]['noise_own'])})")
    print(f"  tightest margin to {G.GATE_THRESHOLD}: {out['tightest']['margin']:.4f} at "
          f"{tight} (R2 {cells[tight]['r2_registered']:+.4f}), against its OWN noise of "
          f"{cells[tight]['noise_own']:.1e} -- {cells[tight]['ratio_own']:.1f}x")
    print(f"  SMALLEST margin-to-own-noise ratio anywhere: {cells[worst_ratio]['ratio_own']:.1f}x "
          f"at {worst_ratio}")
    print(f"  no cell's {G.GATE_THRESHOLD} call flips under any single replicate "
          f"({out['n_zs_measurements']} zero-shot measurements over {len(cells)} cell-scorings).")
    print(f"  calls changed by the pre-repair (all-replicate) denominator: {len(contaminated)}"
          + (": " + ", ".join(contaminated) if contaminated else ""))
    print(f"  wrote {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
