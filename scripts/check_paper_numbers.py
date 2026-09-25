#!/usr/bin/env python3
"""Re-derive the body's load-bearing numbers from the run records and check them against the prose.

WHY THIS EXISTS. Every table the paper \\input's has an emitter, so `git status --porcelain
paper_8/tables/` after a regeneration sweep is a complete staleness check for tables. The prose has
had no such check, and that is where the staleness has actually lived: across ten rounds the body has
asserted "twelve of the fifteen" after the records moved to sixteen, quoted a cell-level interval the
analysis had replaced with a clustered one, and printed p-values the body elsewhere says it does not
report. Each of those was found by eye, late, and one of them only because a reviewer quoted it back.

So this script does for prose what the emitters do for tables: it recomputes each quantity from
`cell_matrix`, `cluster_keys` and the stored JSON, greps the sections for every site that states it,
and fails if any site disagrees. It is deliberately *not* a linter over all numbers -- it is a
registry of the claims that carry weight, and adding a claim to the body means adding it here.

Two design notes that matter if you extend it:

  * Comparison is decimal-aware. The same quantity is printed at different precisions in different
    places -- rho is +0.168 in the abstract and +0.17 in the Figure 1 caption, and both are correct.
    A check states the full-precision value once and each site is compared at the precision it
    prints, so rounding is not a failure but a wrong digit is.

  * Patterns run against whitespace-collapsed text, because LaTeX prose wraps mid-claim and a
    pattern anchored on a line would silently match nothing. A check that matches zero sites FAILS
    rather than passing quietly: a claim that has been reworded out of the pattern's reach is
    exactly the state this script exists to catch, and a silent skip would hide it.

Usage:  .venv12/bin/python scripts/check_paper_numbers.py [--verbose]
Exit 0 iff every registered claim is present and correct at every site that states it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

BODY = [ROOT / "paper_8/main.tex"] + sorted((ROOT / "paper_8/sections").glob("*.tex"))

WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "twenty-two": 22, "twenty-six": 26, "twenty-seven": 27,
    "thirty-one": 31, "thirty-two": 32,
}


# ---------------------------------------------------------------- source reading

def collapse(path):
    """Return (whitespace-collapsed text, [line number for each character]).

    The line map is the whole point of doing this by hand rather than with re.sub: a failure has to
    name a line the author can open, and after collapsing there are no newlines left to count.
    """
    out, lines = [], []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        if raw.lstrip().startswith("%"):          # a commented-out claim is not a claim
            continue
        for ch in raw + " ":
            if ch.isspace():
                if out and out[-1] == " ":
                    continue
                ch = " "
            out.append(ch)
            lines.append(lineno)
    return "".join(out), lines


# ---------------------------------------------------------------- number parsing

def parse(tok):
    """'+0.17' -> (0.17, 2); 'five' -> (5, 0). The second element is the printed precision."""
    t = tok.strip().replace("{", "").replace("}", "").replace("$", "")
    t = t.replace("\\,", "").replace("{,}", "").replace(",", "")
    w = WORDS.get(t.lower())
    if w is not None:
        return float(w), 0
    t = t.replace("\u2212", "-")
    m = re.fullmatch(r"([+-]?)(\d+(?:\.\d+)?)", t)
    if not m:
        raise ValueError(f"cannot parse number from {tok!r}")
    sign = -1.0 if m.group(1) == "-" else 1.0
    d = Decimal(m.group(2))
    return sign * float(d), max(0, -d.as_tuple().exponent)


def agrees(printed_tok, expected):
    """True iff the printed token is the expected value rounded to the precision it prints at."""
    got, nd = parse(printed_tok)
    return round(got, nd) == round(round(float(expected), nd), nd)


# ---------------------------------------------------------------- rederivation

def rederive():
    """Every expected value in one place, each from the records rather than from the last draft."""
    import math

    import numpy as np
    from scipy import stats

    import cell_matrix as cm
    import cluster_keys as ck

    R = {}
    rows = cm.build_rows()
    sf = cm.strict_freeze_cells()

    # -- the cell counts the paper keeps confusing, and now they carry a SPLIT as well as a
    # denominator. n_screened (32) is the screen, which is test-side because a test-side numerator
    # can be recomputed by loading the model. The primary gate is scored on the SELECTION windows,
    # and that is only available where a run stored a validation zero-shot, so its denominator is
    # n_val_scored (31): TimesFM/Electricity is screened but has no paired B/D run and therefore no
    # selection-split reference. n_int is the intervention denominator, also 31, for the same cell.
    # Mixing the two is how "5 of 32" and "7 of 31" would silently become "7 of 32".
    gate_side = json.load(open(ROOT / "results/gate_test_side.json"))
    val_side_primary = json.load(open(ROOT / "results/gate_val_side.json"))
    R["n_screened"] = len(gate_side)
    R["n_val_scored"] = len(val_side_primary)
    R["n_int"] = len(rows)
    # -- the screen's COMPOSITION, which S4 states as a four-way split of the 32. One claim, not four:
    # the relation is what is asserted, and four per-arm checks would all pass with two arms swapped.
    # The parenthetical "three capacities, five datasets, two horizons" is a COVERAGE statement about
    # the Moirai arm and not a product -- 3*5*2 is 30 and we ran 21 -- so it is asserted here over the
    # arm's own cell keys rather than captured as three more numbers a reader would try to multiply.
    _arm_n = {}
    for _v in gate_side.values():
        _arm_n[_v.get("arm")] = _arm_n.get(_v.get("arm"), 0) + 1
    for _a in ("moirai", "chronos", "timesfm", "ili"):
        R[f"n_arm_{_a}"] = _arm_n.get(_a, 0)
    assert sum(_arm_n.values()) == R["n_screened"] and set(_arm_n) == {
        "moirai", "chronos", "timesfm", "ili"}, (
        f"the screen's arms are now {_arm_n}; S4 enumerates the 32 as four arms and the enumeration "
        f"would silently stop adding up")
    _mo_parts = [k.split("_") for k, v in gate_side.items() if v.get("arm") == "moirai"]
    R["n_moirai_caps"] = len({p[0] for p in _mo_parts})
    R["n_moirai_ds"] = len({p[1] for p in _mo_parts})
    R["n_moirai_hz"] = len({p[2] for p in _mo_parts})
    assert (R["n_moirai_caps"], R["n_moirai_ds"], R["n_moirai_hz"]) == (3, 5, 2), (
        f"the Moirai arm now spans {R['n_moirai_caps']} capacities, {R['n_moirai_ds']} datasets and "
        f"{R['n_moirai_hz']} horizons; S4's parenthetical says three, five and two")
    # -- the three capacities' parameter counts, from scripts/emit_model_sizes.py's committed output.
    # They are printed by the loader (moirai_detector.py:348) and every run log has them, but *.log is
    # gitignored, so until that emitter existed these were the only backbone numbers in the body with
    # no record a clean clone could read -- and the ratio S4 quoted was wrong (6.3, from the Moirai
    # paper's rounded capacities, against 6.61 for the checkpoints we actually ran).
    _ms = json.loads((ROOT / "results/model_sizes.json").read_text())
    for _s in ("small", "base", "large"):
        R[f"params_{_s}"] = _ms["models"][_s]["n_params"] / 1e6
    R["base_over_small"] = _ms["base_over_small"]
    assert abs(R["base_over_small"] - R["params_base"] / R["params_small"]) < 1e-9, (
        f"results/model_sizes.json records base/small as {R['base_over_small']!r}, which is not its "
        f"own two counts' ratio -- the file was hand-edited")
    # -- the seed budget the screen paired each cell at, as a range over the intervention rows. The
    # histogram behind it is {3: 24, 5: 4, 10: 3}; S1 states only the endpoints, so only those are
    # registered, but they come from the rows rather than from the sentence.
    R["seeds_lo"], R["seeds_hi"] = min(r["seeds"] for r in rows), max(r["seeds"] for r in rows)
    # From `rows`, so it follows cell_matrix.GATE_SPLIT: primary = the selection split.
    R["n_gate_pass"] = sum(r["gate"] is not None and r["gate"] >= 0.20 for r in rows)
    R["n_gate_fail"] = R["n_val_scored"] - R["n_gate_pass"]
    R["n_clause_i_out"] = R["n_int"] - R["n_gate_pass"]
    assert cm.GATE_SPLIT == "val", (
        f"cell_matrix.GATE_SPLIT is {cm.GATE_SPLIT!r}: the counts registered here are the "
        f"selection-split primary, so a test-side default would check the prose against the "
        f"retrospective variant while the prose describes the primary one")
    # The retrospective variant, still reported as a sensitivity analysis, so it keeps its own name.
    R["n_gate_pass_test"] = sum(1 for v in gate_side.values() if v["r2_task"] >= 0.20)
    R["n_gate_fail_test"] = R["n_screened"] - R["n_gate_pass_test"]
    # degradation_cells() reports to stdout as its main job; here only the count is wanted, and the
    # banner would bury this script's own output when it runs inside rederive_all.sh.
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        R["n_degradation"] = len(cm.degradation_cells(rows))

    # -- the outcome span, the paper's one positive finding
    fb = [r["forg_b"] for r in rows if r["forg_b"] is not None]
    R["forg_b_min"] = abs(min(fb))
    R["forg_b_max"] = max(fb)
    chronos = [r["forg_b"] for r in rows if r["cell"].startswith("Chronos")]
    R["chronos_min"] = abs(min(chronos))
    R["chronos_max"] = max(chronos)
    R["chronos_span"] = max(chronos) - min(chronos)
    tfm = [r["forg_b"] for r in rows if r["cell"].startswith("TimesFM")]
    R["timesfm_min"] = abs(min(tfm))
    R["timesfm_max"] = abs(max(tfm))
    # S6 now states the TimesFM arm qualitatively -- "sit at milder drift and all improve" -- with the
    # per-cell numbers in Appendix Table tab:crossbackbone, because the review asked for the
    # cross-backbone arm to be read as architectural diversity rather than replication. "All" is a
    # quantifier no captured group can carry, so it is an assert: if one TimesFM cell ever degrades the
    # sentence becomes false while still parsing, which is the failure mode this file exists for.
    R["n_timesfm"] = len(tfm)
    assert tfm and max(tfm) < 0, (
        f'"the four TimesFM cells all improve" is false: the worst is {max(tfm):+.1f}%')

    # -- held-out vs selection-window scoring: the sign reversals
    R["n_heldout_rev"] = sum(
        1 for r in rows if r["bd_val"] is not None and r["bd_test"] * r["bd_val"] < 0)

    # -- Figure 1's separability claim, added 20 Sep 2026 with the panel (b) axis flip: the fewest
    # cells any single CKA cut misplaces. Sort by CKA, then for every cut position count the cells on
    # the wrong side of it, under BOTH orientations -- low-CKA-means-freezing-better (which is what a
    # "drift is damage" heuristic implies) and its reverse (which is the direction the positive rho
    # actually points). The minimum over both is the best case ANY threshold rule gets, so the
    # caption's "no cut separates them" is not resting on a choice of direction. On the current
    # records the two orientations give 5 and 14, so the claim is the same number either way.
    # fig1_diagnostic_flow.py computes this identically and asserts it at draw time; the caption is
    # the only prose site, which is why it is registered here rather than left to the figure.
    _sorted_by_cka = sorted(rows, key=lambda r: r["cka"])
    _sgn = [1 if r["bd_test"] > 0 else 0 for r in _sorted_by_cka]
    R["n_miscut"] = min(min(sum(_sgn[:k]) + sum(1 - s for s in _sgn[k:]),
                            sum(1 - s for s in _sgn[:k]) + sum(_sgn[k:]))
                        for k in range(len(_sgn) + 1))

    # -- the strict-freeze control. Recomputed here rather than read from the prose because this arm
    # grew mid-round twice and the prose lagged it both times.
    R["n_sf"] = len(sf)
    R["n_sf_reverse"] = sum(1 for v in sf.values() if v["bd_same"] * v["bh_test"] < 0)
    R["n_sf_agree"] = R["n_sf"] - R["n_sf_reverse"]
    R["n_sf_hneg"] = sum(1 for v in sf.values() if v["forg_h_pos"] == 0)
    R["n_d_only"] = R["n_int"] - R["n_sf"]

    # -- the relaxed-definition ladder. Registered for the first time on 21 Sep 2026, after the ladder
    # was found stale in both directions: rederive_all.sh ran the emitter WITHOUT --latex, so the table
    # still carried the pre-correction 6/5 gate-passing counts, and the body and appendix both said the
    # largest count anywhere is one when the seed-counts-only rung admits two. Imported from the
    # emitter so the prose is checked against the same predicates the table prints.
    import degradation_sensitivity as dsx
    _ds_keep, _ds_noref, _ds_conf = dsx.scorable(rows)
    R["ds_n_scorable"], R["ds_n_confounded"] = len(_ds_keep), len(_ds_conf)
    R["ds_n_rungs"] = len(dsx.LADDER)
    _adm = {(k, g): len(dsx.admits(_ds_keep, rung, g))
            for g in dsx.GATES for k, rung in ((r[0], r) for r in dsx.LADDER)}
    for g, tag in ((0.10, "010"), (0.20, "020")):
        R[f"ds_pass_{tag}"] = sum(1 for r in _ds_keep if r["gate"] >= g)
        R[f"ds_max_{tag}"] = max(_adm[(k, g)] for k, *_ in dsx.LADDER)
        # the six rungs that KEEP the means clause -- every rung except the two "seed counts only" ones
        R[f"ds_max_means_{tag}"] = max(_adm[(k, g)] for k, *_ in dsx.LADDER
                                       if not k.endswith("_only"))
        R[f"ds_n_zero_{tag}"] = sum(1 for k, *_ in dsx.LADDER if _adm[(k, g)] == 0)
    # Both thresholds now behave identically; the appendix used to say relaxing the gate raises the
    # maximum, which was true before the correction. An assert, because no captured group carries it.
    assert R["ds_max_010"] == R["ds_max_020"] and R["ds_pass_010"] == R["ds_pass_020"], \
        "the two gate thresholds no longer agree; the appendix's 'at either threshold' is now false"
    # The maximum is reached by exactly one rung, and it is one that drops the means clause.
    _at_max = [k for k, *_ in dsx.LADDER if _adm[(k, 0.20)] == R["ds_max_020"]]
    assert _at_max == ["half_only"], \
        f"the ladder's maximum is no longer the seed-counts-majority rung alone: {_at_max}"

    # -- the held-out-vs-selection decomposition. Read from the emitter rather than re-derived, so
    # this checks the prose against the same rows the table prints (the emitter already asserts the
    # (B-D)_test - (B-D)_val == dB - dD identity per cell, so the table's recipe is enforced there).
    import contextlib as _c
    import io as _io

    import heldout_decomposition as hd
    with _c.redirect_stdout(_io.StringIO()):
        dec = hd.build()
    rev_rows = [r for r in dec if r["bd_val"] > 0 > r["bd_test"]]
    chron = next(r for r in dec if r["cell"].startswith("Chronos/etth1"))
    R["chron_val"], R["chron_test"] = chron["bd_val"], abs(chron["bd_test"])
    R["chron_swing"] = chron["bd_val"] - chron["bd_test"]
    R["chron_dd"], R["chron_db"] = chron["d_d"], chron["d_b"]
    R["ratio_lo"] = min(r["zs_ratio"] for r in rev_rows)
    R["ratio_hi"] = max(r["zs_ratio"] for r in rev_rows)

    # -- the four ETTm2 strict-freeze shifts, quoted in the appendix as the divergence evidence
    for tag, key in (("s96", "small_ETTm2_h96"), ("b96", "base_ETTm2_h96"),
                     ("s192", "small_ETTm2_h192"), ("b192", "base_ETTm2_h192")):
        v = sf[key]
        R[f"sf_{tag}_bd"], R[f"sf_{tag}_bh"] = v["bd_same"], abs(v["bh_test"])
        # std -> SEM again: strict_freeze_cells() stores standard deviations, the table and the
        # prose both print SEMs (and agree with each other, which is why only a check catches it)
        R[f"sf_{tag}_bd_sem"] = v["bd_same_sd"] / math.sqrt(v["seeds"])
        R[f"sf_{tag}_bh_sem"] = v["bh_test_sd"] / math.sqrt(v["seeds"])
        R[f"sf_{tag}_shift"] = abs(v["bh_test"] - v["bd_same"])

    # -- condition D's OWN measured CKA, which S3 uses to make the paper's sharpest methodological
    # concession (D controls encoder weight updates, not representations) and which tab:cka_calibration
    # states a range and a run count for. Both were hand-typed and the two disagreed: S3 said
    # 0.969--0.9999, the appendix row said 0.889--0.9999 over "80 runs". The appendix's endpoints are
    # the right ones -- 0.969 is one root's minimum (v35_base_frozen) quoted as if it were the range --
    # and its count is stale by 48 runs. Derived here from the records instead.
    #
    # THE SCOPE, and it is not a convenience. Three roots store the PARTIAL-UNFREEZE arm in files named
    # condition_D_*.json whose own `condition` field also reads "D": v49_layerunfreeze_10seed's N3 runs
    # (unfreeze the top 3 of 6 layers) are recorded as "D / Frozen encoder" even though three layers are
    # training, and v26/v27 are the same experiment at fewer seeds. Their CKA runs down to 0.317, so a
    # sweep that globs condition_D by name silently reports the unfreeze arm's range as the frozen
    # encoder's. Excluded by root, with an assert that the mislabelling is still there -- if those
    # records are ever relabelled, this stops being an exclusion anyone needs and the assert says so.
    import glob as _glob
    _LU_ROOTS = {"v26_layer_unfreeze", "v27_layer_unfreeze", "v49_layerunfreeze_10seed"}
    _CELLDIR = re.compile(r"^(?:small|base|large)_")
    _dcka, _lu_seen = [], set()
    for _f in sorted(_glob.glob(str(ROOT / "results/**/condition_D*.json"), recursive=True)):
        _rel = Path(_f).relative_to(ROOT).parts
        _root = _rel[1]
        # RELATIVE, not the absolute path: the repository directory is itself called iotsf_demo, so
        # matching "iot" against an absolute path excludes every record in the tree and the range below
        # comes out empty rather than wrong -- which is the only reason this was caught immediately.
        _low = "/".join(_rel).lower()
        if _root in _LU_ROOTS:
            _lu_seen.add(_root)
            continue
        # The non-Moirai arms, by path: the two extra backbones have their own roots, and batch 3 keeps
        # its Chronos cells beside its Moirai ones under dataset-only directory names (ETTm2_h48/),
        # where the Moirai cells all carry a size prefix.
        if any(t in _low for t in ("chronos", "timesfm", "iot", "nbaiot")):
            continue
        if _root.startswith("v57") and not _CELLDIR.match(_rel[2]):
            continue
        try:
            _d = json.load(open(_f))
        except json.JSONDecodeError:
            continue
        if _d.get("final_cka") is None:
            continue
        # condition_name is absent from the oldest records (forecasting_finetune_20ep predates the
        # field), so it is checked only where it exists: requiring it would exclude the 20 runs that
        # carry the maximum, and defaulting it would defeat the point of checking it at all.
        assert _d.get("condition") == "D" and _d.get("condition_name") in (None, "Frozen encoder"), (
            f"{'/'.join(_rel)} is named condition_D but records condition "
            f"{_d.get('condition')!r}/{_d.get('condition_name')!r}; the frozen-encoder range below "
            f"would include a run of some other protocol")
        _dcka.append(_d["final_cka"])
    assert _lu_seen == _LU_ROOTS, (
        f"the partial-unfreeze roots excluded here are {sorted(_lu_seen)}, not {sorted(_LU_ROOTS)}: a "
        f"root was renamed or removed, so the exclusion no longer does what its comment says")
    _lu_d = []
    for _r in sorted(_LU_ROOTS):
        _mis = [f for f in _glob.glob(str(ROOT / f"results/{_r}/**/condition_D*.json"), recursive=True)]
        _lu_d += [json.load(open(f))["final_cka"] for f in _mis]
        assert _mis, (
            f"results/{_r} no longer stores any condition_D-named record. If the partial-unfreeze arm "
            f"has been relabelled, drop it from _LU_ROOTS -- the exclusion above is only justified by "
            f"the mislabelling")
    R["n_cond_d_runs"] = len(_dcka)
    R["cond_d_cka_lo"] = min(_dcka)
    # How far the excluded arm reaches, which is what makes the exclusion worth stating in the appendix
    # rather than performing silently: it is below the frozen-encoder minimum by more than half a unit.
    R["lu_cond_d_cka_lo"] = min(_lu_d)
    assert R["lu_cond_d_cka_lo"] < R["cond_d_cka_lo"], (
        f"the partial-unfreeze arm's condition_D-named records now bottom out at "
        f"{R['lu_cond_d_cka_lo']:.4f}, inside the frozen encoder's own range; the appendix says they "
        f"reach below it, which was the reason given for excluding them")
    # TRUNCATED to four places, not rounded, and the sentence is why: round(0.999977, 4) prints 1.0000,
    # which contradicts "near but never exactly 1" in the same clause. The claim itself is the assert.
    R["cond_d_cka_hi"] = int(max(_dcka) * 1e4) / 1e4
    assert max(_dcka) < 1.0, (
        f"S3 says condition D's CKA is near but never exactly 1 on Moirai; the maximum over "
        f"{len(_dcka)} runs is now {max(_dcka)!r}")
    # Unity itself, as a registered number, because the paper states it twice as the value one
    # condition HITS and another DOES NOT. agrees() cannot test a negation, so the two asserts carry
    # the claims -- H exactly 1 on every cell, D strictly below on every run -- and the chks only pin
    # the constant the prose compares against. Without this, "not $1.000$" and "exactly $1.000$" were
    # two unchecked numerals: note that D's maximum PRINTS as 1.000 at three places, so rounding
    # cannot distinguish them and the exactness is the whole content of both sentences.
    R["cka_unity"] = 1.0
    assert all(_c["cka_h"] == R["cka_unity"] for _c in sf.values()), (
        "S3 says strict freeze is the only Moirai condition at CKA exactly 1.000; the strict-freeze "
        f"cells now read {sorted({_c['cka_h'] for _c in sf.values()})}")
    # The TimesFM floor in the same sentence, and the weight drift that explains it: D leaves in_proj
    # and mask_encoding trainable, so the encoder stack itself does not move at all.
    _tf = [json.load(open(f)) for f in sorted(
        _glob.glob(str(ROOT / "results/v46_timesfm/*/condition_D/condition_D_*.json")))]
    R["cond_d_cka_timesfm_lo"] = min(d["final_cka"] for d in _tf)
    assert all(d["final_weight_drift"] == 0.0 for d in _tf), (
        "S3 says TimesFM's condition-D encoder weight drift is exactly zero; it is now "
        f"{sorted({d['final_weight_drift'] for d in _tf})}")
    # "H is the only condition with CKA exactly 1.000" -- true within Moirai only. Chronos's condition D
    # also reads 1.0000 (to 5e-8; emit_chronos_m4.py asserts it), because on that backbone the freeze
    # covers everything D leaves trainable on Moirai. S3 carries the Moirai scope for this reason.
    assert all(abs(json.load(open(f))["final_cka"] - 1.0) < 1e-6 for f in sorted(
        _glob.glob(str(ROOT / "results/v37_chronos_etth2/cond_D/**/condition_D_*.json"),
                   recursive=True))), (
        "Chronos's condition D no longer reads CKA 1.0; S3's Moirai scope on 'the only condition with "
        "CKA exactly 1.000' was added because of it and would now be understating the claim")

    # -- S3's "Training details", read out of the trainer and out of the records rather than retyped.
    # Every number in that paragraph was wrong before this block existed: weight decay 10^-2 against
    # the optimiser's 1e-4; batch size 32, which NO Moirai runner passes (only the two Chronos
    # launchers do, and the appendix scopes that 32 to the Chronos arm already); "early stopping with
    # patience 3", when this trainer has no patience counter anywhere and --early-stopping does not
    # stop anything -- it restores the best-selection-epoch checkpoint before the final eval; and
    # n=500, when 22 of the 31 cells ran at 1,000.
    #
    # AST and not import: importing finetune_forecasting.py needs torch, loguru and the dataset
    # loaders, none of which a clean clone has. paper_8/figA_window_layout.py reads the same file the
    # same way for the same reason. The DEFAULTS are what the runs used -- asserted against the runner
    # scripts below, because a default is only the protocol if nothing overrode it.
    import ast as _ast

    def _defaults(src):
        """Every argparse default in a trainer, as the value a run gets when the flag is not passed."""
        out = {}
        for _n in _ast.walk(_ast.parse(src)):
            if not (isinstance(_n, _ast.Call) and _ast.unparse(_n.func).endswith("add_argument")
                    and _n.args and isinstance(_n.args[0], _ast.Constant)):
                continue
            _kw = {k.arg: k.value for k in _n.keywords}
            if isinstance(_kw.get("default"), _ast.Constant):
                out[_n.args[0].value] = _kw["default"].value
            elif "action" in _kw and _ast.unparse(_kw["action"]) == "'store_true'":
                out[_n.args[0].value] = False            # a flag not passed is the protocol's value
        return out

    _ft = ROOT / "scripts/finetune_forecasting.py"
    _ft_src = _ft.read_text()
    _dflt, _adamw, _clip = _defaults(_ft_src), [], []
    for _n in _ast.walk(_ast.parse(_ft_src)):
        if not isinstance(_n, _ast.Call):
            continue
        _fn = _ast.unparse(_n.func)
        if _fn == "torch.optim.AdamW":
            _adamw.append({k.arg: _ast.literal_eval(k.value) for k in _n.keywords
                           if isinstance(k.value, _ast.Constant)})
        elif _fn == "torch.nn.utils.clip_grad_norm_":
            _clip.append(_ast.literal_eval(_n.args[-1]))
    # One of each, asserted: a second optimiser or a second clip call with a different norm would make
    # the paragraph true of one code path and false of the other, which is not a thing prose can say.
    assert len(_adamw) == 1 and len(_clip) == 1, (
        f"finetune_forecasting.py now builds {len(_adamw)} AdamW optimiser(s) and clips in "
        f"{len(_clip)} place(s); S3's training-details paragraph describes exactly one of each")
    R["batch_size"] = _dflt["--batch-size"]
    R["epochs_dflt"] = _dflt["--epochs"]
    R["lr_exp"] = round(-math.log10(_dflt["--lr"]))
    R["wd_exp"] = round(-math.log10(_adamw[0]["weight_decay"]))
    R["clip_norm"] = _clip[0]
    assert (R["batch_size"], R["epochs_dflt"], R["lr_exp"], R["wd_exp"], R["clip_norm"]) == (
        16, 20, 4, 4, 1.0), (
        f"the trainer's defaults moved: batch {R['batch_size']}, {R['epochs_dflt']} epochs, "
        f"lr 1e-{R['lr_exp']}, weight decay 1e-{R['wd_exp']}, clip {R['clip_norm']}. The runs on "
        "disk were made under (16, 20, 4, 4, 1.0) and S3 states those, so a changed default here "
        "means new runs would not be exchangeable with the recorded ones")
    # --early-stopping RESTORES a checkpoint; it never shortens training, and there is no patience
    # window to shorten it by. S3 said "early stopping with patience 3" for ten rounds; the only
    # patience=3 in this repository is scripts/train_moirai.py, the IoT-era trainer.
    assert _dflt["--early-stopping"] is False and "patience" not in _ft_src, (
        "finetune_forecasting.py has grown a patience parameter or flipped --early-stopping's "
        "default; S3 says 20 epochs with no early stopping and every Moirai cell read at its "
        "final epoch, which is a claim about this file")
    # The records' side of the same paragraph, over the seven directories the Moirai matrix is built
    # from -- derived from cell_matrix's own two constants, so a new matrix directory cannot be left
    # out of this check by being left out of a list here.
    _mx_roots = sorted(set(cm.DATASET_OF) | {Path(p).name for p in cm.OWN_A_DIRS})
    _mx = []
    for _r in _mx_roots:
        for _f in sorted(_glob.glob(str(ROOT / "results" / _r / "**/*.json"), recursive=True)):
            _d = json.load(open(_f))
            if isinstance(_d, dict) and _d.get("condition") in ("B", "D") and "test_mse" in _d:
                _mx.append(_d)
    assert len(_mx) >= 170, f"only {len(_mx)} Moirai matrix B/D records found under {_mx_roots}"
    assert {_d.get("epochs") for _d in _mx} == {R["epochs_dflt"]}, (
        f"S3 says every Moirai cell runs {R['epochs_dflt']} epochs; the records carry "
        f"{sorted({_d.get('epochs') for _d in _mx})}")
    # "no early stopping" over two generations of record: the newer ones store the flag's state and
    # must read False; the older ones predate the field, and their runners are what says it (none of
    # them passes --early-stopping -- run_v5_experiments.sh, run_prospective_arm.sh and
    # run_deadline_tail.sh are the three that wrote these directories).
    _es = {json.dumps(_d["early_stopping"], sort_keys=True) for _d in _mx if "early_stopping" in _d}
    assert _es == {'{"enabled": false}'}, (
        f"a Moirai matrix record now reports early stopping: {sorted(_es)}. S3 says the arm runs "
        "20 epochs with no early stopping and is read at its final epoch")
    for _sh in ("run_v5_experiments.sh", "run_prospective_arm.sh", "run_deadline_tail.sh"):
        _sh_src = (ROOT / "scripts" / _sh).read_text()
        assert "--early-stopping" not in _sh_src and "--batch-size" not in _sh_src, (
            f"scripts/{_sh} now passes --early-stopping or overrides --batch-size; it wrote part of "
            "the Moirai matrix, whose protocol S3 states as batch 16 with no early stopping")
    # The modal n, and the denominator it is a fraction of. From the rows, not the records: n is part
    # of a CELL's identity (cell_matrix._display's docstring says why every row prints it), and the
    # same cell contributes 6-20 records.
    _ns = [int(_m.group(1)) if (_m := re.search(r" n(\d+)$", _r["cell"])) else _r.get("n_train")
           for _r in rows]
    R["n_train_modal"] = max(set(_ns), key=_ns.count)
    R["n_train_modal_cells"] = _ns.count(R["n_train_modal"])
    assert R["n_train_modal"] == 1000 and R["n_train_modal_cells"] == 22, (
        f"the matrix's modal training size is now n={R['n_train_modal']} on "
        f"{R['n_train_modal_cells']} of {len(rows)} cells; S3 states both")

    # -- four constants the prose states and the CODE owns, each read from the module that uses it.
    # The gate's operating point and the interval level are stated in S3 as design choices and then
    # acted on 300 times downstream, so a constant changed in one place and a sentence left alone in
    # the other is a paper describing a design it did not run.
    import emit_sample_sweep as _ess
    import gate_all_cells as _gac
    import paired_inference as _pinf        # the BH level is read again below, at its own claim
    R["gate_threshold_code"] = _gac.GATE_THRESHOLD
    R["ci_pct"] = round(100 * (1 - _pinf.ALPHA))
    # The sweep grid, from the emitter's own GROUPS. The prose prints thousands as "1k", so the
    # PRINTED tokens are 500,1,2,5,10 -- registered in that form because agrees() compares what is
    # printed, and "1" is not a rounding of 1000.
    R["n_grid"] = tuple(g[0] for g in _ess.GROUPS)
    assert R["n_grid"] == (500, 1000, 2000, 5000, 10000), (
        f"emit_sample_sweep.GROUPS now sweeps {R['n_grid']}; S3 lists the grid")
    R["n_grid_printed"] = tuple(n if n < 1000 else n // 1000 for n in R["n_grid"])
    # The compute ceiling, over EVERY condition record in the tree rather than over the matrix: the
    # claim is about what this project cost and so covers every arm, including the ones the paper
    # reports only in an appendix.
    # EVERY json, not just condition_*.json: 16 records of the n=10k sweep are named by seed alone
    # (results/v8_final/n10000/s456.json), and one of them is the tree's argmax -- a glob on the
    # condition_ prefix reports the same maximum today and would miss the run that broke the claim.
    _all_n = []
    for _f in _glob.glob(str(ROOT / "results/**/*.json"), recursive=True):
        try:
            _d = json.load(open(_f))
        except Exception:
            continue
        if isinstance(_d, dict) and isinstance(_d.get("max_train_samples"), int):
            _all_n.append(_d["max_train_samples"])
    R["n_train_max"] = max(_all_n)
    assert R["n_train_max"] % 1000 == 0 and len(_all_n) > 700, (
        f"the largest training set in the tree is now {R['n_train_max']} over {len(_all_n)} records; "
        "the Compute statement prints it as a round number of thousands")
    R["n_train_max_k"] = R["n_train_max"] // 1000

    # -- the gate's geometry, and the forecast it is scored against. S3 defines the primary baseline
    # by its shape, its penalty and its window cap, and no run record stores any of the three: they
    # are FUNCTION-SIGNATURE defaults, so the code is the only evidence and the prose statement of
    # them was unfalsifiable until now. Same failure mode as the training details, one layer deeper --
    # argparse defaults at least appear in a runner's command line; these appear nowhere but here.
    def _sig_defaults(src, fn):
        """A function's keyword defaults: the values a caller gets by omitting the argument."""
        for _n in _ast.walk(_ast.parse(src)):
            if not (isinstance(_n, _ast.FunctionDef) and _n.name == fn):
                continue
            _pos = _n.args.posonlyargs + _n.args.args
            _pairs = list(zip(_pos[len(_pos) - len(_n.args.defaults):], _n.args.defaults))
            _pairs += [(a, d) for a, d in zip(_n.args.kwonlyargs, _n.args.kw_defaults) if d]
            return {a.arg: d.value for a, d in _pairs if isinstance(d, _ast.Constant)}
        raise AssertionError(f"{fn}() is gone from the source this claim reads")

    _gac_src = (ROOT / "scripts/gate_all_cells.py").read_text()
    _mg = _sig_defaults(_gac_src, "moirai_gates")
    R["ridge_lookback"] = _mg["lookback"]                  # 96
    R["gate_max_eval"] = _mg["max_eval"]                   # 300
    R["ridge_lam_exp"] = round(-math.log10(_sig_defaults(
        (ROOT / "scripts/gate_linear_baseline.py").read_text(), "fit_linear_map")["lam"]))   # 4
    assert (R["ridge_lookback"], R["gate_max_eval"], R["ridge_lam_exp"]) == (96, 300, 4), (
        f'the gate now fits a lookback-{R["ridge_lookback"]} ridge at 1e-{R["ridge_lam_exp"]} over '
        f'{R["gate_max_eval"]} windows; S3 states all three as the definition of the primary $b$')
    assert _mg["baseline"] == "fitted", (
        f'moirai_gates() now defaults to the {_mg["baseline"]!r} baseline; S3 states the primary '
        "gate is the FITTED least-squares map, and the unfitted extrapolation is the estimator the "
        "boxed lesson says an earlier version of this work got wrong")
    # The one production caller that passes a lookback explicitly has to pass the same one, or
    # "lookback-96" is true of the matrix and false of the pooled screen sharing its threshold.
    _pool_lb = re.search(r"^LOOKBACK = (\d+)$",
                         (ROOT / "scripts/pool_screen_prospective3.py").read_text(), re.M)
    assert _pool_lb and int(_pool_lb.group(1)) == R["ridge_lookback"], (
        f'pool_screen_prospective3 screens at lookback {_pool_lb and _pool_lb.group(1)} against the '
        f'matrix\'s {R["ridge_lookback"]}')
    # The context asymmetry is a claim about an EXPRESSION, not a number: S3 says the pre-trained
    # model sees 96+h steps where the baseline sees 96, and calls that generous to the pre-trained
    # model. gate_all_cells:232 records that lookback*2 -- the obvious alternative -- would pair the
    # stored zero-shot MSE with a baseline evaluated on different windows at h=192.
    assert re.search(r"ext_lb = lookback \+ h\b", _gac_src), (
        "gate_all_cells no longer evaluates at lookback + h; S3's asymmetry claim reads that "
        "expression, and lookback*2 would make the 96+h in the prose wrong at h=192")
    # The median is over a fixed number of sampled forecast paths: the evaluator's own default, and
    # again at the single production construction of the detector. num_samples=2 also appears in this
    # file and is NOT this constant -- it is the throwaway forward pass a hook uses to capture
    # encoder output for CKA, where the sample count cannot affect what is measured.
    R["fc_samples"] = _sig_defaults(_ft_src, "evaluate_forecasting")["num_samples"]
    _det = [_n for _n in _ast.walk(_ast.parse(_ft_src))
            if isinstance(_n, _ast.Call) and _ast.unparse(_n.func) == "MoiraiAnomalyDetector"]
    _det_ns = [k.value.value for d in _det for k in d.keywords
               if k.arg == "num_samples" and isinstance(k.value, _ast.Constant)]
    assert R["fc_samples"] == 20 and _det_ns == [R["fc_samples"]], (
        f'the evaluator defaults to {R["fc_samples"]} forecast samples and the detector is built '
        f'with {_det_ns}; S3 states the median is over one number of samples')
    # The horizons the body reports, from the matrix rather than from the runner: the ILI cell is the
    # one Moirai cell at another horizon, and it is not ETT, which is exactly how S3 scopes the claim.
    _hs = set()
    for _r in rows:
        if _r["cell"].startswith("Moirai") and "ILI" not in _r["cell"]:
            _m = re.search(r"[ _]h(\d+)(?![0-9])", _r["cell"])   # [ _] or "ETTh1" reads as h=1
            assert _m, f'no horizon in the cell label {_r["cell"]!r}'
            _hs.add(int(_m.group(1)))
    R["moirai_h"] = tuple(sorted(_hs))
    assert R["moirai_h"] == (96, 192), (
        f"the Moirai ETT cells now span horizons {R['moirai_h']}; S3 names them")

    # -- the two-cell from-scratch check, read out of the shell script that performs it. The
    # reproducibility statement describes both invocations in prose, and a flag changed in the script
    # with the prose left alone would misstate what was reproduced.
    _r2 = (ROOT / "scripts/rerun_two_cells.sh").read_text()

    def _inv(script):
        # Anchored on "$PY -u", not on the script name alone: the file's header comment names both
        # scripts twenty lines above the invocations, and a bare name match reads the comment and
        # then runs on to the `2>&1` in the usage line -- which parses, and yields nothing.
        _m = re.findall(rf"\$PY -u scripts/{re.escape(script)}(.*?)2>&1", _r2, re.S)
        assert len(_m) == 1, f"rerun_two_cells.sh now invokes {script} {len(_m)} times, not once"
        return _m[0]

    def _flag(inv, name):
        _m = re.search(rf"--{name}\s+(\S+)", inv)
        return _m.group(1) if _m else None

    _mo_inv, _tf_inv = _inv("finetune_forecasting.py"), _inv("finetune_timesfm.py")
    R["r2_moirai_h"] = int(_flag(_mo_inv, "horizon"))
    R["r2_moirai_n"] = int(_flag(_mo_inv, "max-train-samples"))
    R["r2_seed"] = int(_flag(_mo_inv, "seed"))
    # The TimesFM cell's horizon is the trainer's DEFAULT -- the script does not pass --horizon -- so
    # it is read from finetune_timesfm.py and asserted absent from the invocation. Reading the default
    # while the script overrode it would print a horizon that cell never ran at.
    assert _flag(_tf_inv, "horizon") is None, (
        "rerun_two_cells.sh now passes --horizon to finetune_timesfm.py; the reproducibility "
        "statement's h=24 is registered as that trainer's default and would stop being what ran")
    R["r2_tf_h"] = _defaults((ROOT / "scripts/finetune_timesfm.py").read_text())["--horizon"]
    assert int(_flag(_tf_inv, "max-train-samples")) == 1000 and \
        int(_flag(_tf_inv, "seed")) == R["r2_seed"] and \
        _flag(_mo_inv, "condition") == _flag(_tf_inv, "condition") == "B", (
        "rerun_two_cells.sh's two invocations no longer match the prose: it says both cells run "
        "condition B at seed 42, the TimesFM one at n=1,000")

    # -- every results/ path the reproducibility statement names in \texttt{} has to be a real file.
    # Not a number, so no chk() can carry it, and it is the one kind of claim in that section a
    # reviewer checks FIRST: the statement said the prospective predictions were written to
    # "preregistration.json under results/", and results/preregistration.json does not exist -- the
    # file is results/v47_prospective/preregistration.json, and the second batch's registration was
    # not named at all. Existence in a clean clone is the right test because an untracked file would
    # not be in one, which is exactly the reader's situation.
    _stmt = (ROOT / "paper_8/sections/09_statements.tex").read_text()
    _named = {m.replace("\\_", "_") for m in
              re.findall(r"\\texttt\{(results/[A-Za-z0-9_\\/.]+?)\}", _stmt)}
    assert _named, "no results/ path is named in the reproducibility statement any more"
    _absent = sorted(p for p in _named if not (ROOT / p).exists())
    assert not _absent, (
        f"the reproducibility statement names {_absent}, which a reader cloning the release will not "
        f"find; name the path that exists or add the file")
    R["n_named_paths"] = len(_named)

    # -- the positive control's dose ladder, from the REGISTRATION rather than from the module: the
    # registration is what was fixed before the runs, and it is the artifact a reader checks. The
    # module constant is asserted to agree, so a later edit to either one fails here.
    # The body named three rungs and stopped. The declared extension to 1e-1 WAS taken -- the appendix
    # says so and counts 21 runs on the strength of it -- so a reader multiplying the body's grid out
    # got 18 and a contradiction with the appendix the same sentence cites. The body now calls its
    # grid "the registered" one and the extension is checked at the appendix site that explains it;
    # the registered rungs and the extension are kept as separate values because the difference
    # between them is the difference between a pre-registration honoured and one exceeded.
    _pc = json.load(open(ROOT / "results/positive_control/preregistration.json"))["grid"]
    import preregister_positive_control as _ppc
    assert _pc["learning_rates"] == _ppc.LADDER_LRS, (
        f'preregister_positive_control.LADDER_LRS is now {_ppc.LADDER_LRS} against the registered '
        f'{_pc["learning_rates"]}; the registration is what the arm ran')
    R["pc_lr_exps"] = tuple(round(-math.log10(_l)) for _l in _pc["learning_rates"])
    R["pc_seeds"] = len(_pc["seeds"])
    R["pc_ext_exp"] = round(-math.log10(_pc["declared_extension_lr"]))
    assert R["pc_lr_exps"] == (4, 3, 2) and R["pc_seeds"] == 3 and R["pc_ext_exp"] == 1, (
        f'the positive control\'s ladder is now 1e-{R["pc_lr_exps"]} at {R["pc_seeds"]} seeds with '
        f'the extension at 1e-{R["pc_ext_exp"]}; S7 states all three')
    # The extension is stated in the body only because it was actually run: if no record carries it,
    # the body should say the rule permitted it and it was not needed, which is a different sentence.
    _pc_lrs = set()
    for _f in _glob.glob(str(ROOT / "results/positive_control/**/*.json"), recursive=True):
        if "prereg" in _f:
            continue
        try:
            _d = json.load(open(_f))
        except Exception:
            continue
        if isinstance(_d, dict) and "lr" in _d:
            _pc_lrs.add(_d["lr"])
    assert _pc["declared_extension_lr"] in _pc_lrs, (
        f"S7 says the arm went on to the pre-declared extension, but no positive-control record "
        f"carries lr={_pc['declared_extension_lr']}; the rungs present are {sorted(_pc_lrs)}")

    # -- the LoRA value-cell arm (app:loravaluecells). Read through cell_matrix rather than from the
    # emitted table, so the prose is checked against the records and not against the same file it was
    # written from.
    # MIN_SEEDS, not len(): emit_lora_latex() drops under-seeded cells by name, so counting the raw
    # dict would let the coverage paragraph claim a cell the table does not print.
    lo = {k: v for k, v in cm.lora_value_cells().items() if v["be_seeds"] >= cm.MIN_SEEDS}
    R["lora_cells"] = len(lo)
    # Drift is read from the raw records, not from the per-cell mean: "exactly 0.0 on all 24" is a
    # claim about every record, and a mean of zero is also what two equal-and-opposite values give.
    e_files = sorted(ROOT.glob("results/v54_lora_valuecells/*/condition_E/condition_E_h*_s*.json"))
    R["lora_records"] = len(e_files)
    R["lora_drift_max"] = max(abs(json.load(open(f))["final_weight_drift"]) for f in e_files)
    cka_e = [v["cka_e"] for v in lo.values()]
    R["lora_cka_lo"], R["lora_cka_hi"] = min(cka_e), max(cka_e)
    R["lora_be_neg"] = sum(1 for v in lo.values() if v["be_test"] < 0)
    # The appendix states its own decision rule -- paired D-E, unanimous sign -- so the rule is
    # reproduced here rather than the three counts being copied across. A caption that states its
    # recipe is a testable claim; this is the test.
    beats = sum(1 for v in lo.values() if v["de_pos"] == v["be_seeds"])
    loses = sum(1 for v in lo.values() if v["de_pos"] == 0)
    R["lora_e_beats_d"], R["lora_d_beats_e"] = beats, loses
    R["lora_inseparable"] = len(lo) - beats - loses
    R["lora_rho_be"] = stats.spearmanr(cka_e, [v["be_test"] for v in lo.values()]).statistic
    R["lora_rho_bd"] = stats.spearmanr(cka_e, [v["bd_same_e"] for v in lo.values()]).statistic
    for tag, ref in (("bh2_96", "base_ETTh2_h96"), ("bm2_192", "base_ETTm2_h192"),
                     ("lh2_96", "large_ETTh2_h96"), ("bm2_96", "base_ETTm2_h96"),
                     ("sh1_96", "small_ETTh1_h96")):
        v = lo[ref]
        R[f"lora_{tag}_forg"] = v["forg_e"]
        R[f"lora_{tag}_forg_pos"] = v["forg_e_pos"]
        R[f"lora_{tag}_cka"] = v["cka_e"]
        R[f"lora_{tag}_be"] = v["be_test"]
        R[f"lora_{tag}_be_sem"] = v["be_test_sd"] / math.sqrt(v["be_seeds"])
        R[f"lora_{tag}_seeds"] = v["be_seeds"]
    # The Small/ETTh1 "difference of means would mislead" claim: the gap between the two means, and
    # the SEM the appendix quotes beside it.
    R["lora_sh1_gap"] = abs(lo["small_ETTh1_h96"]["be_test"] - lo["small_ETTh1_h96"]["bd_same_e"])
    # The appendix names two cells as the extremes of CKA_E and reads the dissociation off them. The
    # digits are checked below; that they are still the extremes is not something a regex can see, so
    # it is asserted. A new E cell outside this range would leave every printed number correct and the
    # word "highest" wrong.
    assert R["lora_lh2_96_cka"] == R["lora_cka_hi"], "Moirai-Large/ETTh2 h96 is no longer max CKA_E"
    assert R["lora_bm2_96_cka"] == R["lora_cka_lo"], "Moirai-Base/ETTm2 h96 is no longer min CKA_E"
    # Seeds and rank are uniform across the arm, which is what lets the appendix state one seed set
    # and one rank for all eight cells instead of a per-cell column.
    seedsets = {tuple(v["e_seedset"]) for v in lo.values()}
    ranks = {v["lora_rank"] for v in lo.values()}
    assert len(seedsets) == 1 and len(ranks) == 1, f"arm is not uniform: {seedsets}, {ranks}"
    R["lora_seed1"], R["lora_seed2"], R["lora_seed3"] = seedsets.pop()
    R["lora_rank"] = ranks.pop()
    # Strict-freeze and LoRA coverage, for the condition-coverage paragraph.
    R["sf_cell_count"] = len([r for r in rows if r.get("bh_test") is not None
                              and (r.get("bh_seeds") or 0) >= cm.MIN_SEEDS])

    # -- the two-cell from-scratch check (Reproducibility Statement). Deviations are recomputed from
    # the two JSON pairs rather than copied out of the log, so a re-run that moves them moves the
    # statement's numbers too. Absent if the check has not been run; the claims are then unmatched,
    # which the zero-sites rule turns into a failure rather than a silent pass.
    fresh_m = ROOT / "results/v55_rerun_check/moirai/condition_B_h96_s42.json"
    fresh_t = ROOT / "results/v55_rerun_check/timesfm/condition_B_h24_s42.json"
    if fresh_m.exists() and fresh_t.exists():
        sm = json.load(open(ROOT / "results/v5_etth2_sweep/n500/condition_B_h96_s42.json"))
        fm = json.load(open(fresh_m))
        st = json.load(open(ROOT / "results/v46_timesfm/ETTh1_h24/condition_B/"
                            "condition_B_h24_s42.json"))
        ft = json.load(open(fresh_t))

        def dev(s, f, k):
            # Signed, because the statement prints the signs and they are the informative part: the
            # two MSE ends move in OPPOSITE directions, which is why the ratio between them moves so
            # much further than either. An abs() here would let a same-sign pair pass the check and
            # leave the paragraph's explanation of the mechanism unverified.
            return (f[k] - s[k]) / s[k] * 100

        R["rr_zs"] = dev(sm, fm, "zeroshot_mse")
        R["rr_val"] = dev(sm, fm, "final_val_mse")
        R["rr_test"] = dev(sm, fm, "test_mse")
        R["rr_cka"] = dev(sm, fm, "final_cka")
        R["rr_drift"] = dev(sm, fm, "final_weight_drift")
        R["rr_forg"] = dev(sm, fm, "forgetting_pct")
        # forg. is (val - zeroshot)/zeroshot, so the two MSE devs above fully determine this one.
        # Asserted rather than trusted: if it ever fails, the statement's causal explanation of the
        # amplification is wrong even if every printed number is right.
        for rec in (sm, fm):
            assert abs(rec["forgetting_pct"] - (rec["final_val_mse"] - rec["zeroshot_mse"])
                       / rec["zeroshot_mse"] * 100) < 1e-6, \
                "forgetting_pct is not (val-zs)/zs in one of the two records"
        R["rr_cka_stored"], R["rr_cka_fresh"] = sm["final_cka"], fm["final_cka"]
        R["rr_forg_stored"], R["rr_forg_fresh"] = sm["forgetting_pct"], fm["forgetting_pct"]
        # The statement explains the Moirai/TimesFM asymmetry by the sample count, which lives in the
        # runner rather than in any run record -- so it is read from the runner. The alternative was to
        # leave the one number in that paragraph that cannot go stale unchecked, which is the habit
        # this script exists to break.
        # It is the same constant rederive() already reads by AST from the evaluator's signature and
        # the detector's construction, so it is taken from there rather than re-found by regex. The
        # regex that used to be here matched the FIRST "\n *num_samples=(\d+)," in the file and got
        # the right one only because the CKA hook's num_samples=2 is the last argument in its call
        # and so carries no trailing comma: add one and this number silently became 2.
        R["rr_moirai_samples"] = R["fc_samples"]
        R["rr_tfm_fields"] = sum(
            1 for k in ("zeroshot_mse", "zeroshot_test_mse", "final_val_mse", "test_mse",
                        "final_cka", "final_weight_drift", "forgetting_pct_test",
                        "forgetting_pct_val", "best_epoch")
            if k in st and k in ft and st[k] == ft[k])

    # -- within-Moirai correlations, clustered. Same call the body's figures come from.
    moirai = [r for r in rows if r["cell"].startswith("Moirai")]
    cl = ck.clusters_for([r["cell"] for r in moirai])
    y = np.array([r["bd_test"] for r in moirai])
    R["n_moirai"] = len(moirai)
    # `gate` follows cell_matrix.GATE_SPLIT and is therefore the selection-split primary.
    # `gate_test` re-reads the same 22 rows against the retrospective test-side score, because the
    # paper quotes both -- the whole point of \S7 is that the magnitude is not stable across splits,
    # and a claim about two numbers needs both of them registered.
    preds = {"cka": [r["cka"] for r in moirai],
             "gate": [r["gate"] for r in moirai],
             "gate_test": [gate_side[r["ref"]]["r2_task"] for r in moirai]}
    for tag, xs in preds.items():
        x = np.array(xs)
        rho, clo, chi, _ = ck.cell_bootstrap_spearman(x, y, rng=np.random.default_rng(0))
        _, lo, hi, _, ncl = ck.cluster_bootstrap_spearman(
            x, y, cl, rng=np.random.default_rng(0))
        # magnitudes, because the sign is carried by the prose's own {+}/{-} and the pattern
        # anchors on it -- a flipped sign shows up as a zero-site match, not as a passing check
        R[f"{tag}_rho"], R[f"{tag}_lo"], R[f"{tag}_hi"] = abs(rho), abs(lo), hi
        R[f"{tag}_celllo"], R[f"{tag}_cellhi"] = abs(clo), abs(chi)
    R["n_moirai_clusters"] = ncl

    # -- the cross-backbone dataset ordering, which agrees on one split and not the other. An
    # earlier draft read the test-side +1.00 as evidence the ranking is a benchmark property; it is
    # registered here so the withdrawal in Exp 1 cannot drift from the numbers that forced it.
    for tag, src in (("xback_test", gate_side), ("xback_val", val_side_primary)):
        ar = {a: {k.split("_", 1)[1]: v["r2_task"] for k, v in src.items() if v.get("arm") == a}
              for a in ("chronos", "timesfm")}
        shared = sorted(set(ar["chronos"]) & set(ar["timesfm"]))
        R[f"{tag}_rho"] = abs(stats.spearmanr([ar["chronos"][k] for k in shared],
                                             [ar["timesfm"][k] for k in shared]).statistic)
        R[f"{tag}_n"] = len(shared)

    # -- the graded value axis and the pooled figure, from the stored artifact
    va = json.load(open(ROOT / "results/value_axis.json"))["correlations"]
    for key, tag in (("cka_vs_denc_all", "pooled"), ("cka_vs_denc_valuecells", "vc"),
                     ("cka_vs_denc_lowvalue", "lv"), ("value_vs_denc_all", "val"),
                     ("value_vs_denc_valuecells", "valvc")):
        c = va[key]
        R[f"{tag}_rho"], R[f"{tag}_lo"], R[f"{tag}_hi"] = c["rho"], abs(c["lo"]), c["hi"]
        R[f"{tag}_n"] = c["n"]
    inter = json.load(open(ROOT / "results/value_axis.json"))["interaction"]["cka_x_value"]
    R["b3"], R["b3_lo"], R["b3_hi"] = abs(inter["b"]), abs(inter["lo"]), inter["hi"]

    # -- the SECOND diagnostic's pooled correlation, from the clustered-inference artifact. S6 puts it
    # in the same sentence as the pooled CKA figure above, and until 25 Sep 2026 it was the one
    # correlation in that sentence with no check at all -- which is how the two l2-drift statistics
    # fixed in ee244a7 went stale. What S6 asserts is the number AND a property of the interval ("with
    # a CI including zero"), so the interval is an assert rather than a captured group: a records
    # change that moved it off zero would leave every numeral in the sentence correct and the sentence
    # false. rederive_all.sh regenerates this JSON immediately before running this script.
    _cl = json.loads((ROOT / "results/clustered_inference.json").read_text())["pooled"]
    R["pooled_drift_rho"] = _cl["drift"]["rho"]
    R["pooled_drift_n"] = _cl["drift"]["n"]
    _dlo, _dhi = _cl["drift"]["cluster_ci"]
    assert _dlo < 0 < _dhi, (
        f"the pooled l2-drift clustered CI is [{_dlo:+.3f}, {_dhi:+.3f}], which no longer includes "
        f"zero; S6's 'with a CI including zero' is false as written")
    # Two emitters compute the pooled CKA rho -- value_axis.py (read above) and clustered_inference.py
    # (read here) -- and S6 quotes it beside the drift figure. If they ever disagree the sentence is
    # citing one artifact and checked against the other, so they are required to agree here.
    assert abs(_cl["cka"]["rho"] - R["pooled_rho"]) < 1e-12, (
        f"pooled CKA rho is {R['pooled_rho']!r} in value_axis.json and {_cl['cka']['rho']!r} in "
        f"clustered_inference.json; S6 quotes one number and this script would check the other")

    # -- paired inference at the value-cell aggregate
    _pi = json.load(open(ROOT / "results/paired_inference.json"))
    agg = _pi["aggregate"]["value"]["d_enc"]
    R["agg_mean"] = abs(agg["mean"])
    R["agg_lo"], R["agg_hi"] = abs(agg["lo"]), agg["hi"]
    R["n_vc"], R["n_vc_pos"] = agg["n_cells"], agg["pos"]

    # -- the three-way call, which is the headline of this revision. Derived from the SAME file the
    # table is emitted from, and counted off the stored `call` rather than recomputed from q and the
    # mean here: a second implementation of the rule would let the check pass while the emitter drifts,
    # which is the failure mode the whole script exists to catch.
    _pic = _pi["cells"]
    _calls = [c["call"] for c in _pic]
    R["n_freeze_dec"] = _calls.count("freeze")
    R["n_adapt_dec"] = _calls.count("adapt")
    R["n_inconclusive"] = _calls.count("inconclusive")
    R["n_paired_tests"] = len(_pic)
    # The BH level itself, from the module the table is emitted by rather than from the JSON, which does
    # not store it. Two sites state it and both would read as pre-registered while the emitter used a
    # different alpha; this makes the prose's 0.05 an assertion about the code that produced the calls.
    import paired_inference as _pin
    R["bh_alpha"] = _pin.ALPHA
    # The two freeze-decisive cells and the two adapt-decisive cells that PASS the screen, by name, in
    # the order the appendix states them (h96 then h192 for ETTh1; Base h192 then Large h96 for ETTh2).
    _by = {c["cell"]: c for c in _pic}
    for tag, cell in (("fd96", "Moirai-small/ETTh1 h96 n1000"),
                      ("fd192", "Moirai-small/ETTh1 h192 n1000"),
                      ("ad192", "Moirai-base/ETTh2 h192 n1000"),
                      ("adL96", "Moirai-large/ETTh2 h96 n500")):
        c = _by[cell]
        R[f"{tag}_mean"] = abs(c["d_enc"]["mean"])
        R[f"{tag}_q"] = c["d_enc"]["q"]
        R[f"{tag}_gate"] = abs(c["gate"])
    # -- power. The MDE range and the two unresolvable cells' adjusted seed requirement. Taken over the
    # gate-PASSING cells only, which is the set the table reports, so the range cannot silently widen
    # to include cells the table does not show.
    _surv = [c for c in _pic if (c.get("gate") or -9) >= 0.20]
    _mdes = [c["d_enc"]["mde"] for c in _surv]
    R["mde_lo"], R["mde_hi"] = min(_mdes), max(_mdes)
    R["n_star_s96"] = _by["Moirai-small/ETTh2 h96 n500"]["d_enc"]["n_needed_adj"]
    # Underscores, not slashes: this one cell's record carries the older key spelling, and normalising
    # it here would hide that. It is the same cell the power table's last row names.
    R["n_star_b192m"] = _by["Moirai-base_ETTm2_h192"]["d_enc"]["n_needed_adj"]

    # -- the eight-rung ladder. "Admissible" = a denominator no worse than the training-mean floor,
    # which is the rule the caption states, so it is the rule reproduced here. Derived for BOTH
    # splits, under separate key prefixes: the body quotes the selection-split ladder (the split its
    # admission decisions are made on) and the appendix keeps the held-out one as the retrospective
    # variant. One derivation over two files, so the two can never be checked by one set of numbers.
    def ladder(path, gate):
        gb = json.load(open(ROOT / path))
        cells, T = gb["cells"], gb["gate_threshold"]
        rungs = ["fitted"] + [b for b in gb["ladder_order"] if b != gb["floor"]]
        out = {"n_cells": len(cells)}
        out["counts"] = [sum(
            1 for per in cells.values()
            if b in per and per[b].get("r2_task") is not None
            and per[b]["linear_test"] <= per[gb["floor"]]["linear_test"]
            and per[b]["r2_task"] >= T) for b in rungs]
        # Below-the-floor tallies, which the appendix narrates rung by rung.
        out["below_floor"] = {b: sum(
            1 for per in cells.values()
            if b in per and per[b].get("r2_task") is not None
            and per[b]["linear_test"] > per[gb["floor"]]["linear_test"]) for b in rungs}
        strongest = {}
        for k, per in cells.items():
            adm = [(e["linear_test"], b, e["r2_task"]) for b, e in per.items()
                   if b != gb["floor"] and e.get("r2_task") is not None
                   and e["linear_test"] <= per[gb["floor"]]["linear_test"]]
            if adm:
                strongest[k] = min(adm)
        out["n_clear_strongest"] = sum(1 for v in strongest.values() if v[2] >= T)
        # The union: cells clearing T against AT LEAST ONE admissible rung. Read from the stored
        # `graded` block rather than recomputed, so the paper, the figure and this checker cannot
        # disagree about which cells are in the analysis scope.
        graded = gb["graded"]["cells"]
        union = [k for k, v in graded.items() if v["clears_any_admissible"]]
        out["n_union"] = len(union)
        # Backbone read off the ref, not off a cell label: the ladder files are keyed by ref
        # (`small_ETTh1_h96`, `chronos_etth1`, `timesfm_etth1`) and carry no label column.
        def bb_of(ref):
            r = ref.lower()
            return "Chronos" if "chronos" in r else ("TimesFM" if "timesfm" in r else "Moirai")
        out["union_backbones"] = {bb: sum(1 for k in union if bb_of(k) == bb)
                                  for bb in ("Moirai", "Chronos", "TimesFM")}
        # The fitted-gate survivors, scored against the strongest admissible rung they face. Which
        # cells count as survivors comes from the gate PUBLISHED for this split, not from the
        # ladder's own `fitted` column, so this mirrors the prose's "the seven that clear the gate".
        surv = [v[2] for k, v in strongest.items() if k in gate and gate[k]["r2_task"] >= T]
        out["surv_hi"], out["surv_lo"] = max(surv), min(surv)
        return out

    # -- the prospective arm's positive class, under the CORRECTED gates. The frozen predictor's own
    # scores are pre-correction and are not touched; what is derived here is the answer to "did the
    # baseline correction empty the positive class?", which S7 now states as a number.
    pro = json.load(open(ROOT / "results/v47_prospective/preregistration.json"))
    pro_rows = {r["ref"]: r for r in rows}
    clearing = [c["cell"] for c in pro["cells"]
                if max(val_side_primary.get(c["cell"], {}).get("r2_task", -9),
                       gate_side.get(c["cell"], {}).get("r2_task", -9)) >= pro["gate_threshold"]]
    R["n_pro"] = len(pro["cells"])
    R["n_pro_clearing_corrected"] = len(clearing)
    assert len(clearing) == 1, f"S7 says exactly one prospective cell clears; records say {clearing}"
    R["pro_clearing_gate"] = val_side_primary[clearing[0]]["r2_task"]
    R["pro_clearing_forg_pos"] = pro_rows[clearing[0]]["forg_b_pos"]
    R["pro_clearing_seeds"] = pro_rows[clearing[0]]["seeds"]

    # -- the confusion counts and the threshold sweep. S7 has stated TP 0, FP 8, FN 0, precision 0.00
    # and the six flagged counts as hand-typed literals since the arm was added, matched by no pattern
    # here. Derived now from the emitter's own sweep, which is the same function that writes
    # tables/gate_sensitivity.tex, so the table and the prose cannot disagree.
    import gate_threshold_sensitivity as gts
    with contextlib.redirect_stdout(io.StringIO()):
        _pro_scored, _sweep = gts.sweep()
    _primary = [r for r in _sweep if abs(r[0] - gts.PRIMARY) < 1e-9]
    assert len(_primary) == 1, f"the primary threshold {gts.PRIMARY} is not on the sweep grid"
    _thr, R["pro_flagged"], R["pro_tp"], R["pro_fp"], R["pro_fn"], R["pro_prec"], _rec = _primary[0]
    R["pro_flagged_by_threshold"] = [r[1] for r in _sweep]
    R["pro_n_thresholds"] = len(_sweep)
    R["pro_threshold_grid"] = [f"{g:.2f}" for g in gts.GRID]
    # "gives precision 0.00 at every one" is a claim about all six rungs, not just the printed one, and
    # it cannot be captured as a group. Asserted instead, which is this file's convention for a claim
    # a pattern cannot hold.
    assert all(r[5] == 0.0 or r[5] != r[5] for r in _sweep), (
        "S7 says precision is 0.00 at every threshold from 0.10 to 0.60; the sweep now gives "
        + ", ".join(f"{r[0]:.2f}:{r[5]:.2f}" for r in _sweep))
    assert R["pro_tp"] == 0 and R["pro_fn"] == 0, (
        "S7's zero-prevalence reading assumes an empty positive class; the sweep now gives "
        f'TP {R["pro_tp"]}, FN {R["pro_fn"]}')

    # -- how many rungs admit a degradation cell, and how many cells each admitting rung admits. The
    # body sentence that reports this was hand-counted wrong once ("five of the eight" when six of the
    # eight admit none), which is exactly the arithmetic a derived check exists to stop.
    import gate_baseline_sensitivity as gbs
    with contextlib.redirect_stdout(io.StringIO()):
        dc_val = gbs.degradation_counts(
            json.load(open(ROOT / "results/gate_baselines_val.json"))["cells"],
            ["fitted", "persistence", "seasonal_naive", "ar", "dlinear", "ridge_tuned", "mlp", "gbm"])
    R["n_rungs_no_degradation"] = sum(1 for v in dc_val.values() if v["n"] == 0)
    R["n_degradation_persistence"] = dc_val["persistence"]["n"]
    R["n_degradation_gbm"] = dc_val["gbm"]["n"]
    R["n_degradation_manufactured"] = sum(v["n"] for v in dc_val.values())

    lad_val = ladder("results/gate_baselines_val.json", val_side_primary)
    lad_test = ladder("results/gate_baselines.json", gate_side)
    R["ladder_counts"] = lad_val["counts"]
    R["n_clear_strongest"] = lad_val["n_clear_strongest"]
    R["n_union"] = lad_val["n_union"]
    R["n_union_moirai"] = lad_val["union_backbones"]["Moirai"]
    R["n_union_chronos"] = lad_val["union_backbones"]["Chronos"]
    R["n_union_timesfm"] = lad_val["union_backbones"]["TimesFM"]
    # The body prints this range as "between $-0.55$ and $+0.13$", i.e. signed and ordered low-to-
    # high, so the registered values are signed too. The retrospective variant's range is all
    # negative and the appendix prints it as two magnitudes, which is why that one is abs()ed.
    R["surv_strongest_lo"], R["surv_strongest_hi"] = lad_val["surv_lo"], lad_val["surv_hi"]
    R["ladder_counts_test"] = lad_test["counts"]
    R["n_clear_strongest_test"] = lad_test["n_clear_strongest"]
    R["n_union_test"] = lad_test["n_union"]
    R["surv_strongest_hi_test"] = abs(min(lad_test["surv_lo"], lad_test["surv_hi"]))
    R["surv_strongest_lo_test"] = abs(max(lad_test["surv_lo"], lad_test["surv_hi"]))
    R["below_floor_val"] = lad_val["below_floor"]

    # -- the matched-lookback column (Appendix app:baselines:matchedlb, added 21 Sep 2026). Read from
    # the emitter's own summary block rather than recounted here: the emitter is the one place that
    # decides what "stronger" and "flips" mean, and a second implementation of the comparison is how
    # the body and the table would come to disagree about the same 21 cells. The two splits are kept
    # apart under explicit _val/_test suffixes because the effect REVERSES between them -- a single
    # unsuffixed record would let the prose quote whichever direction reads better.
    for tag, fname in (("val", "results/matched_lookback_val.json"),
                       ("test", "results/matched_lookback.json")):
        d = json.load(open(ROOT / fname))
        assert d["selfcheck_passed"], (
            f"{fname} records a failed self-check: its `fitted` column does not reproduce the "
            f"published denominator, so neither column may be quoted")
        s = d["summary"]
        R[f"mlb_n_{tag}"] = s["n_scored"]
        R[f"mlb_pass_nominal_{tag}"] = s["n_pass_nominal"]
        R[f"mlb_pass_matched_{tag}"] = s["n_pass_matched"]
        R[f"mlb_stronger_{tag}"] = s["n_matched_stronger"]
        R[f"mlb_rung_clearing_after_{tag}"] = d["as_ladder_rung"]["n_clearing_after"]
        R[f"mlb_rung_strongest_{tag}"] = d["as_ladder_rung"]["n_matched_strongest"]
        if tag == "val":
            # The one cell that flips on the split the body's count comes from, and both its values.
            # Named from the record, so a different cell flipping fails here rather than in review.
            flips = s["flips_to_fail"]
            assert flips == ["base_ETTm2_h192"], (
                f"the selection-split matched-lookback flip is now {flips}; the appendix names "
                f"Moirai-Base/ETTm2 h=192 explicitly")
            R["mlb_flip_before"] = d["per_cell"][flips[0]]["r2_nominal"]
            R["mlb_flip_after"] = d["per_cell"][flips[0]]["r2_matched"]
    # The appendix says the folded-in count is "0 of 21 on BOTH splits", and the registered check can
    # only pin one of them. Asserted here so the word "both" is covered rather than assumed.
    assert R["mlb_rung_clearing_after_val"] == R["mlb_rung_clearing_after_test"], (
        f"the folded-in ladder count differs between splits "
        f"({R['mlb_rung_clearing_after_val']} vs {R['mlb_rung_clearing_after_test']}); the appendix "
        f"states one number for both")
    assert R["mlb_n_val"] == R["mlb_n_test"], "the two splits no longer score the same 21 cells"

    # -- the third window set. Derived through emit_traintail_ladder.analyse() rather than re-counted
    # here, so the body, the table and this check cannot disagree: the emitter is the one place that
    # decides what "counted over the cells all three splits share" means, and that restriction is the
    # whole point of the comparison (19 of 21 against 20 of 31 would be a statement about coverage).
    import emit_traintail_ladder as ttl
    _tt = ttl.analyse()
    R["n_traintail"] = _tt["n_common"]
    R["n_tt_all_admissible"] = [_tt["n_all"][s] for s in _tt["splits"]]
    R["n_tt_any_sel"], R["n_tt_any_tail"] = _tt["n_any"]["Selection"], _tt["n_any"]["Train tail"]
    R["n_tt_fitted_sel"] = len(_tt["fitted_set"]["Selection"])
    R["n_tt_fitted_tail"] = len(_tt["fitted_set"]["Train tail"])
    R["n_tt_fitted_both"] = len(_tt["fitted_overlap"])
    # The three zeros are the confirmation the body leans on, so a nonzero anywhere must break the
    # build rather than quietly weaken a sentence that says "on any of the three window sets".
    assert R["n_tt_all_admissible"] == [0, 0, 0], (
        f'"no cell clears every admissible rung on any of the three window sets" is false: '
        f'{R["n_tt_all_admissible"]}')
    assert len(_tt["fitted_tail_only"]) == 1, (
        f'the appendix names ONE train-tail-only fitted cell; there are '
        f'{len(_tt["fitted_tail_only"])}: {_tt["fitted_tail_only"]}')
    # The mechanism figures the appendix quotes, read from the two JSONs rather than transcribed.
    _cell = "base_ETTh2_h96"
    _v = json.load(open(ROOT / "results/gate_baselines_val.json"))["cells"][_cell]["fitted"]
    _t = json.load(open(ROOT / "results/gate_baselines_traintail.json"))["cells"][_cell]["fitted"]
    R["tt_ridge_val"], R["tt_ridge_tail"] = _v["linear_test"], _t["linear_test"]
    R["tt_ridge_ratio"] = _v["linear_test"] / _t["linear_test"]
    R["tt_zs_val"], R["tt_zs_tail"] = _v["zs_test"], _t["zs_test"]

    # -- the unfitted-baseline correction, which is where "16 of 21" and "17" come from
    trend = json.load(open(ROOT / "results/gate_test_side_trend.json"))
    common = [k for k in gate_side if k in trend]
    mo_keys = [k for k in common if gate_side[k].get("arm") == "moirai"]
    R["n_moirai_screened"] = len(mo_keys)
    R["n_flip_moirai"] = sum(
        1 for k in mo_keys if (trend[k]["r2_task"] >= 0.20) != (gate_side[k]["r2_task"] >= 0.20))
    R["n_flip_moirai_p2f"] = sum(
        1 for k in mo_keys if trend[k]["r2_task"] >= 0.20 > gate_side[k]["r2_task"])
    R["n_flip_all"] = sum(
        1 for k in common if (trend[k]["r2_task"] >= 0.20) != (gate_side[k]["r2_task"] >= 0.20))
    for arm in ("moirai", "chronos", "timesfm"):
        ks = [k for k in common if gate_side[k].get("arm") == arm]
        R[f"n_screened_{arm}"] = len(ks)
        R[f"n_pass_{arm}"] = sum(1 for k in ks if gate_side[k]["r2_task"] >= 0.20)
        R[f"n_pass_trend_{arm}"] = sum(1 for k in ks if trend[k]["r2_task"] >= 0.20)
    R["n_gate_pass_val"] = sum(1 for v in val_side_primary.values() if v["r2_task"] >= 0.20)
    # -- Chronos on ETTh2, on BOTH splits. S4's claim is the contrast: the dataset where Moirai clears
    # the gate at every capacity is where Chronos sits below the baseline, and it does so on the
    # selection split and on held-out. One pattern carrying both numbers, because the claim is that the
    # sign holds across splits -- two separate checks would pass with the splits transposed, and this
    # arm is exactly where a split transposition has happened before (app:corrections).
    R["chronos_etth2_val"] = abs(val_side_primary["chronos_etth2"]["r2_task"])
    R["chronos_etth2_test"] = abs(gate_side["chronos_etth2"]["r2_task"])
    assert (val_side_primary["chronos_etth2"]["r2_task"] < 0
            and gate_side["chronos_etth2"]["r2_task"] < 0), (
        f"Chronos/ETTh2 now scores {val_side_primary['chronos_etth2']['r2_task']:+.4f} on selection "
        f"and {gate_side['chronos_etth2']['r2_task']:+.4f} on held-out; S4 says \\emph{{below}} the "
        f"fitted baseline on both, and the magnitudes below are registered unsigned")

    # -- the gate values quoted cell by cell. Whole passages of Exp 1 and the correction appendix
    # are lists of these, hand-typed, and they are the least likely thing to be re-checked by eye.
    # Two families, kept apart on purpose. `gate_of`/`mo_*`/`nonmo_*` are TEST-side, because the
    # correction appendix's "before and after" lists are a historical record of the test-side screen
    # and must not move when the primary split changes. `vgate_of`/`vmo_*`/`vnonmo_*` are the
    # SELECTION-side primary, which is what Experiment 1's own cell-by-cell prose quotes. A single
    # shared record here is exactly how a val-side sentence would silently pass against a test-side
    # number -- the failure this whole file exists to prevent.
    R["gate_of"] = {k: v["r2_task"] for k, v in gate_side.items()}
    R["vgate_of"] = {k: v["r2_task"] for k, v in val_side_primary.items()}
    for pfx, src in (("mo", gate_side), ("vmo", val_side_primary)):
        mo_gate = {k: v["r2_task"] for k, v in src.items() if v.get("arm") == "moirai"}
        for ds in ("ETTh1", "Weather", "Electricity7", "ETTm2"):
            vals = [v for k, v in mo_gate.items() if k.split("_")[1] == ds]
            # "spans X to Y" runs from the least negative to the most negative, as the prose reads
            R[f"{pfx}_{ds}_hi"], R[f"{pfx}_{ds}_lo"] = abs(max(vals)), abs(min(vals))
        nm = [v["r2_task"] for v in src.values() if v.get("arm") in ("chronos", "timesfm")]
        R[f"{'nonmo' if pfx == 'mo' else 'vnonmo'}_lo"] = abs(min(nm))
        R[f"{'nonmo' if pfx == 'mo' else 'vnonmo'}_hi"] = max(nm)
        R[f"{'nonmo' if pfx == 'mo' else 'vnonmo'}_n"] = len(nm)
    # The two failing ETTm2 cells are quoted individually now that the other two clear the gate.
    R["vmo_ETTm2_fail_hi"] = abs(R["vgate_of"]["small_ETTm2_h96"])
    R["vmo_ETTm2_fail_lo"] = abs(R["vgate_of"]["base_ETTm2_h96"])
    R["ili_gate"] = abs(next(v["r2_task"] for v in gate_side.values() if v.get("arm") == "ili"))
    # The 32nd cell. TimesFM/Electricity is screened but has no paired B/D run, so it has no
    # selection-split gate at all -- gate_val_side.json does not contain it, and its only gate value
    # is test-side. The cross-backbone caption is the one place that number is printed, and it sat
    # unregistered while the caption also called the whole column "held-out"; both are fixed together.
    assert "timesfm_electricity" not in val_side_primary, \
        "TimesFM/Electricity now has a selection-split gate; the caption's 'no paired run' is stale"
    R["timesfm_elec_gate_test"] = gate_side["timesfm_electricity"]["r2_task"]

    # -- the gate's own sampling noise (app:zsnoise), from scripts/gate_zs_noise.py. Read from its
    # JSON rather than recomputed here: that script asserts its grouping reproduces
    # gate_all_cells._zs_val_refs() exactly and that no admission flips under any replicate, so
    # recomputing the same quantities a second way here would give two answers to defend instead of
    # one. Keys are named for what the appendix says, not for what the JSON calls them.
    _zsn = json.loads((ROOT / "results/gate_zs_noise.json").read_text())
    R["zsn_cells"] = _zsn["n_cells"]
    R["zsn_measurements"] = _zsn["n_zs_measurements"]
    R["zsn_samples"] = _zsn["num_samples"]
    R["zsn_max_spread"] = _zsn["max_noise_any_cell"]
    R["zsn_min_ratio"] = _zsn["min_ratio_any_cell"]
    R["zsn_tight_margin"] = _zsn["tightest"]["margin"]
    R["zsn_tight_r2"] = _zsn["tightest"]["r2"]
    R["zsn_n_changed"] = _zsn["n_calls_changed_by_contaminated_denominator"]
    # The cell carrying the largest spread, and its distance from the operating point. The appendix's
    # argument is that these two numbers belong to DIFFERENT cells, so both are pinned: if the max
    # spread ever migrated to the tightest cell the prose would be wrong while every number in it
    # still parsed.
    _wide = max(_zsn["cells"], key=lambda k: _zsn["cells"][k]["noise_own"])
    R["zsn_wide_margin"] = _zsn["cells"][_wide]["margin"]
    assert _wide != _zsn["min_ratio_cell"], (
        f"{_wide} now carries both the largest spread and the tightest margin; app:zsnoise's "
        f"'they are different cells' argument no longer holds")
    for _i, _b in enumerate((1, 2, 3), 1):
        R[f"zsn_delta_b{_i}"] = _zsn["by_batch"][f"prospective batch {_b}"]["max_delta_r2"]

    # -- S6's four remaining unregistered pairs, and S5.2's sweep restatement. Registered 25 Sep 2026
    # after scripts/audit_chk_coverage.py found 15 uncovered numerals in 06_exp3_drift.tex. In every
    # one of these the claim is the RELATION between two numbers -- "near-identical CKA, opposite
    # sign", "indistinguishable on the task at very different CKA", "worse still" -- so each pair is
    # one pattern with two groups and the relation itself is an assert. A per-number check would pass
    # with the pair swapped, which is the likeliest error here.
    import glob

    # (a) Chronos/ETTh1, the one cross-backbone cell that degrades, and what freezing does to it. Its
    # forg_b is already pinned as R["chronos_max"] at S6's line-39 site; the line-56 restatement is a
    # different phrasing of the same number, and coverage is per phrasing, not per fact.
    _ch1 = next(r for r in rows if r["cell"].startswith("Chronos/etth1"))
    R["chronos_etth1_b"], R["chronos_etth1_d"] = _ch1["forg_b"], _ch1["forg_d"]
    assert R["chronos_etth1_d"] > R["chronos_etth1_b"] > 0, (
        f'Chronos/ETTh1 no longer degrades under both conditions with the frozen encoder worse '
        f'(B {R["chronos_etth1_b"]:+.1f}%, D {R["chronos_etth1_d"]:+.1f}%); S6\'s "freezing does not '
        f'save it" is stale')

    # (b) the random-init CKA floor S6 compares the Chronos arm against, and the cells at or below it.
    # "Four of the five" is DERIVED from the floor rather than typed: which cells fall below it is a
    # consequence of the records, and the fifth (Chronos/Electricity at 0.232) clears it by 0.018, so a
    # small records move could change the count while leaving both printed bounds correct.
    _floor = json.loads((ROOT / "results/v50_cka_floor/random_init_floor_reset.json").read_text())
    R["cka_floor"], R["cka_floor_sd"] = _floor["cka_mean"], _floor["cka_std_ddof1"]
    R["n_cka_floor_seeds"] = len(_floor["cka_values"])
    _chr_rows = [r for r in rows if r["cell"].startswith("Chronos")]
    assert len(_chr_rows) == 5, \
        f'S6 says "four of the five Chronos-T5-Small cells"; the arm now has {len(_chr_rows)}'
    _below = sorted(r["cka"] for r in _chr_rows if r["cka"] <= R["cka_floor"])
    R["n_chronos_below_floor"] = len(_below)
    R["chronos_cka_lo"], R["chronos_cka_hi"] = _below[0], _below[-1]
    assert R["n_chronos_below_floor"] == 4, (
        f'S6 says four of the five Chronos cells sit at or below the {R["cka_floor"]:.3f} floor; '
        f'{R["n_chronos_below_floor"]} now do')

    # (c) the layer-unfreeze comparison (app:layerunfreeze), restated in S6. Two matched 10-seed arms
    # of one cell. S6 prints only the two CKAs, so the "indistinguishable on the task" half is an
    # assert on the outcome difference against its own standard error: records that pulled the
    # outcomes apart would leave S6 numerically correct and its sentence false.
    for _n in (3, 6):
        _fs = sorted(glob.glob(str(
            ROOT / f"results/v49_layerunfreeze_10seed/N{_n}_seed*/*.json")))
        _v = np.array([[json.loads(Path(f).read_text())[k]
                        for k in ("final_cka", "forgetting_pct")] for f in _fs])
        assert len(_v) == 10, f"the N={_n} unfreeze arm has {len(_v)} seeds, not the matched 10"
        R[f"lu{_n}_cka"] = float(_v[:, 0].mean())
        R[f"lu{_n}_cka_sd"] = float(_v[:, 0].std(ddof=1))
        R[f"lu{_n}_forg"] = abs(float(_v[:, 1].mean()))
        R[f"lu{_n}_forg_sd"] = float(_v[:, 1].std(ddof=1))
        R[f"lu{_n}_sem"] = R[f"lu{_n}_forg_sd"] / len(_v) ** 0.5
        R[f"lu{_n}_neg"] = int((_v[:, 1] < 0).sum())
    R["lu_cka_gap"] = R["lu3_cka"] - R["lu6_cka"]
    R["lu_cka_se"] = R["lu_cka_gap"] / (
        (R["lu3_cka_sd"] ** 2 + R["lu6_cka_sd"] ** 2) ** 0.5 / 10 ** 0.5)
    R["lu_forg_gap"] = abs(R["lu3_forg"] - R["lu6_forg"])
    R["lu_forg_sed"] = (R["lu3_sem"] ** 2 + R["lu6_sem"] ** 2) ** 0.5
    R["lu_forg_se"] = R["lu_forg_gap"] / R["lu_forg_sed"]
    assert R["lu_forg_se"] < 1 < R["lu_cka_se"], (
        f'the two matched unfreeze depths now differ by {R["lu_forg_se"]:.2f} SE on the task and '
        f'{R["lu_cka_se"]:.2f} SE on CKA; S6 says indistinguishable on the former, separated on the '
        f'latter, which is the whole dissociation that sentence reports')

    # (d) Moirai-Large's LoRA arm: S6's "the learning rate, not the rank, decides the sign". Two
    # groups, both r=8 on ETTh2 h=96 -- the default learning rate and the 10x reduction, whose five
    # seeds were added in three batches and so live under three roots. The roots are LISTED rather
    # than globbed loosely, for the reason emit_sample_sweep.py lists its own: a new result directory
    # must not be able to join a group silently and move a published mean.
    for _tag, _pats in (("deflr", ["results/v8_etth2_large/condition_E_h96_s*.json"]),
                        ("lowlr", ["results/v12_lora_large_hp/lr1e-5/condition_E_h96_s42.json",
                                   "results/v13_lora_large_lr1e-5/seed*/condition_E_h96_s*.json",
                                   "results/v21_lora_large_k5/seed*/condition_E_h96_s*.json"])):
        _recs = [json.loads(Path(f).read_text())
                 for p in _pats for f in sorted(glob.glob(str(ROOT / p)))]
        assert _recs and all(r["lora_rank"] == 8 for r in _recs), \
            f"the {_tag} LoRA group is empty or a non-r8 record joined it"
        R[f"lora_{_tag}_n"] = len(_recs)
        for _k, _s in (("cka", "final_cka"), ("forg", "forgetting_pct")):
            _x = [r[_s] for r in _recs]
            R[f"lora_{_tag}_{_k}"] = abs(float(np.mean(_x)))
            R[f"lora_{_tag}_{_k}_sd"] = float(np.std(_x, ddof=1))
        R[f"lora_{_tag}_sign"] = float(np.mean([r["forgetting_pct"] for r in _recs]))
    assert R["lora_deflr_sign"] > 0 > R["lora_lowlr_sign"], (
        f'the learning rate no longer decides the sign on Moirai-Large '
        f'(default {R["lora_deflr_sign"]:+.1f}%, 10x lower {R["lora_lowlr_sign"]:+.1f}%)')
    # The other half of the same sentence: rank escalation does NOT decide it. One mean per rank,
    # including r=8's three seeds, so "stays +19 to +30%" is read off the four rungs and the claim
    # that none of them recovers the sign is an assert rather than a reader's inference.
    _by_rank = {8: [r["forgetting_pct"] for r in
                    (json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(
                        ROOT / "results/v8_etth2_large/condition_E_h96_s*.json"))))]}
    for _p in ("results/v11_large_lora_rank/r*/condition_E_h96_s42.json",
               "results/v9_large_lora_rank/condition_E_h96_s42.json"):
        for _f in sorted(glob.glob(str(ROOT / _p))):
            _d = json.loads(Path(_f).read_text())
            _by_rank.setdefault(_d["lora_rank"], []).append(_d["forgetting_pct"])
    R["n_lora_rungs"] = len(_by_rank)
    _rank_means = {k: float(np.mean(v)) for k, v in _by_rank.items()}
    assert sorted(_by_rank) == [8, 16, 32, 64], \
        f"app:lora_rank reports r in {{8,16,32,64}}; the records now give {sorted(_by_rank)}"
    assert min(_rank_means.values()) > 0, (
        f"a rank now recovers the sign at the default learning rate ({_rank_means}); app:lora_rank's "
        f"'no rank recovering Moirai-Small's negative forgetting' is false")
    R["lora_rank_lo"] = min(_rank_means.values())
    R["lora_rank_hi"] = max(_rank_means.values())

    # (e) the sample sweep's body restatement (S5.2). The TABLE has an emitter; the sentence that
    # restates four of its numbers had no check, so the two could drift apart. The groups are IMPORTED
    # from that emitter rather than re-globbed here, which is what makes this a check on the prose
    # rather than a second definition of the sweep.
    import emit_sample_sweep as ess
    _sw = {_n: ess.load(_pats) for _n, _pats, _ in ess.GROUPS}
    for _n in (500, 2000, 10000):
        _f = [r["forgetting_pct"] for r in _sw[_n]]
        R[f"sw{_n}_forg"] = abs(float(np.mean(_f)))
        R[f"sw{_n}_sem"] = float(np.std(_f, ddof=1)) / len(_f) ** 0.5
        R[f"sw{_n}_neg"] = sum(x < 0 for x in _f)
    for _n in (500, 10000):
        R[f"sw{_n}_cka"] = float(np.mean([r["final_cka"] for r in _sw[_n]]))
    # The sentence's two structural claims, neither of which any captured group can carry: drift falls
    # monotonically in n, and the outcome does not. If the first ever broke, the body's "drift grows
    # monotonically with n" would be false with both printed endpoints still right.
    _cka_seq = [float(np.mean([r["final_cka"] for r in _sw[_n]])) for _n, _, _ in ess.GROUPS]
    _forg_seq = [float(np.mean([r["forgetting_pct"] for r in _sw[_n]])) for _n, _, _ in ess.GROUPS]
    assert all(a > b for a, b in zip(_cka_seq, _cka_seq[1:])), \
        f"drift no longer grows monotonically with n; CKA over the sweep is {_cka_seq}"
    assert not all(a >= b for a, b in zip(_forg_seq, _forg_seq[1:])), \
        f"forgetting is now monotonic over the sweep ({_forg_seq}); S5.2 says it is not"

    # -- the five survivors' own numbers, which the body quotes cell by cell
    by = {r["cell"]: r for r in rows}
    improvers = ["Moirai-base/ETTh2 h96 n1000", "Moirai-base/ETTh2 h192 n1000",
                 "Moirai-large/ETTh2 h96 n500"]
    for i, c in enumerate(improvers, 1):
        R[f"imp{i}_b"] = abs(by[c]["forg_b"])
        R[f"imp{i}_d"] = abs(by[c]["forg_d"])
        R[f"imp{i}_cka"] = by[c]["cka"]
    # The fourth improver, added by the selection split. Its cell label is un-normalised in
    # cell_matrix ("Moirai-base_ETTm2_h192", not "Moirai-base/ETTm2 h192 n1000") unlike its siblings,
    # so it is keyed by `ref` rather than by display label -- a lookup by the sibling convention
    # raises KeyError here rather than silently skipping the check.
    _fb = [r["forg_b"] for r in rows if r["forg_b"] is not None]
    R["forg_b_lo"], R["forg_b_hi"] = abs(min(_fb)), max(_fb)
    imp4 = next(r for r in rows if r["ref"] == "base_ETTm2_h192")
    R["imp4_b"], R["imp4_d"], R["imp4_cka"] = abs(imp4["forg_b"]), abs(imp4["forg_d"]), imp4["cka"]
    # -- the two DEGRADERS the lead example is built on: Moirai-Base's two least-drifted cells, which
    # are the only two in that arm that get worse. The improvers above were registered; these were not,
    # and they are half of every statement of the dissociation -- including S1's "one checkpoint, two
    # datasets, opposite lessons", the first concrete numbers in the paper and the thesis in miniature.
    # Keyed by `ref` for the reason imp4 is. The asserts are the load-bearing part: what S1 claims is
    # not "22.8%" but the SIGN PATTERN -- least drifted degrade, most drifted improve -- so a records
    # change that preserved the magnitudes while flipping a sign would satisfy a numeric check and
    # falsify the sentence. Checked here once rather than in each of the four prose patterns below.
    for _i, _ref in enumerate(("base_ETTh1_h96", "base_ETTh1_h192"), 1):
        _deg = next(r for r in rows if r["ref"] == _ref)
        R[f"deg{_i}_b"], R[f"deg{_i}_cka"] = _deg["forg_b"], _deg["cka"]
        assert R[f"deg{_i}_b"] > 0, f"{_ref} no longer degrades; the lead example's premise is stale"
        assert R[f"deg{_i}_cka"] > R["imp1_cka"], \
            f"{_ref} is no longer less drifted than the improvers; the dissociation has reversed"
    small192 = by["Moirai-small/ETTh2 h192 n500"]
    small96 = by["Moirai-small/ETTh2 h96 n500"]
    # `forg_b_sd` is a standard deviation (ddof=1, via cell_matrix._sd) but the body declares every
    # +/- to be a SEM, so the conversion happens here. This is the one place the project's mixed
    # dispersion conventions can bite silently: both numbers are "3.32-ish" for a 10-seed cell and
    # neither is wrong, so a check against the raw sd would look like a stale-prose failure.
    R["s192_forg_b"] = small192["forg_b"]
    R["s192_sem"] = small192["forg_b_sd"] / math.sqrt(small192["seeds"])
    R["s192_pos"], R["s192_seeds"] = small192["forg_b_pos"], small192["seeds"]
    R["s96_forg_b"] = small96["forg_b"]
    R["s96_sem"] = small96["forg_b_sd"] / math.sqrt(small96["seeds"])
    R["s96_pos"], R["s96_seeds"] = small96["forg_b_pos"], small96["seeds"]

    # -- "improved by full fine-tuning": the MEAN reading and the EVERY-SEED reading differ by one
    # cell, and the body states both counts and says which one it uses. Registering both is the point:
    # the figure asserts the unanimous count at draw time, so if these two ever coincide (or diverge by
    # more than one) the body's sentence explaining the discrepancy becomes wrong while still parsing.
    _pass = [r for r in rows if r["gate"] is not None and r["gate"] >= 0.20]
    R["n_imp_mean"] = sum(r["forg_b"] < 0 for r in _pass)
    R["n_imp_unan"] = sum(r["forg_b"] < 0 and r["forg_b_pos"] == 0 for r in _pass)
    # The survivor whose mean improves but whose seeds disagree -- the cell that makes the two differ.
    _split = [r for r in _pass if r["forg_b"] < 0 and r["forg_b_pos"] != 0]
    assert len(_split) == 1, f"expected exactly 1 mean-improves/seeds-disagree survivor, got {len(_split)}"
    R["split_forg_b"] = abs(_split[0]["forg_b"])
    R["split_pos"], R["split_seeds"] = _split[0]["forg_b_pos"], _split[0]["seeds"]

    # -- WHICH datasets the survivors are, and the size of the improvement on them. Until 20 Sep 2026
    # the abstract and the introduction's LEAD both said "five cells clear it, and all five are ETTh2"
    # -- the pre-correction, test-split composition -- while contribution 1 twenty lines below said
    # "7 of 31 ... five on ETTh2 and two on ETTm2". Both counts were individually derivable and only
    # one was registered, so the paper contradicted itself in its first two paragraphs and the checker
    # slid past it: the lead's wording matched no pattern. The composition is therefore derived here
    # per dataset, not asserted, and the improvement range with it.
    def _ds(ref):  # "base_ETTh2_h192 ..." / "Moirai-small/ETTh2 h96 n500" -> ETTH2
        return re.sub(r"_h\d+.*$", "", re.sub(r"^(small|base|large|chronos|timesfm)[_ ]", "",
                                              ref.split()[0])).upper()
    _scored = [r for r in rows if r["gate"] is not None]
    # Scoped to MOIRAI, because that is what the sentence says: all seven survivors are Moirai, so
    # "two of four on ETTm2" is two of Moirai's four and not two of the six ETTm2 cells screened
    # across all three backbones. The unscoped denominator is 6, and the unscoped "every ETTh2 cell"
    # would be false -- Chronos and TimesFM each have an ETTh2 cell and both fail. Deriving it
    # Moirai-scoped is what keeps the prose from having to be read charitably.
    def _moirai(r):
        return not re.match(r"(chronos|timesfm)", r["ref"].lower())
    for tag in ("etth2", "ettm2"):
        R[f"n_scored_{tag}"] = sum(_ds(r["ref"]) == tag.upper() and _moirai(r) for r in _scored)
        R[f"n_pass_{tag}"] = sum(_ds(r["ref"]) == tag.upper() and _moirai(r) for r in _pass)
    assert R["n_pass_etth2"] + R["n_pass_ettm2"] == len(_pass), "a survivor outside ETTh2/ETTm2"
    # S5's opening says the strict-freeze reversals land on "every Moirai/ETTm2 cell, 2 of which are
    # among the seven survivors". Both halves are quantifiers, so both are asserts, and the second is
    # what licenses reporting the ETTm2 survivor count as the number of survivors affected.
    _sf_rev = {k for k, v in sf.items() if v["bd_same"] * v["bh_test"] < 0}
    assert all(_ds(k) == "ETTM2" for k in _sf_rev), \
        f"a strict-freeze reversal outside ETTm2: {sorted(_sf_rev)}"
    _pass_m2 = {r["ref"].split()[0] for r in _pass if _ds(r["ref"]) == "ETTM2" and _moirai(r)}
    assert _pass_m2 <= set(sf), \
        f"an ETTm2 survivor was never run under strict freeze: {sorted(_pass_m2 - set(sf))}"
    # The abstract and the intro both say "ALL five of its ETTh2 cells", which is a claim no captured
    # group can carry: if one Moirai ETTh2 cell ever failed the gate, both counts would still be 5 and
    # both sentences would still parse while being false.
    assert R["n_pass_etth2"] == R["n_scored_etth2"], \
        f'"all of its ETTh2 cells" is false: {R["n_pass_etth2"]} of {R["n_scored_etth2"]} pass'
    # S4 used to list the failing cells' gate values one dataset at a time -- four hand-typed ranges
    # restating Table r2task, none of them registered here. They are replaced by the qualitative claim
    # that every failing Moirai cell is BELOW the baseline rather than merely short of 0.20, which is
    # what the ranges were there to show. No captured group can carry "every", so it is an assert.
    _mfail = [r["gate"] for r in _scored if _moirai(r) and r["gate"] < 0.20]
    assert _mfail and max(_mfail) < 0, (
        '"every failing cell is below the baseline" is false: the best failing Moirai cell scores '
        f'{max(_mfail):+.3f}')
    # The three big improvers the abstract quotes a range for. Named by dataset rather than by cell so
    # that a fourth ETTh2 survivor entering the set breaks the count instead of widening the range
    # silently.
    _big = sorted(abs(r["forg_b"]) for r in _pass if _ds(r["ref"]) == "ETTH2" and r["forg_b"] < 0)
    R["n_imp_big"], R["imp_lo"], R["imp_hi"] = len(_big), _big[0], _big[-1]
    # The SAME improvers over all four (the ETTm2 cell the selection split added is the fourth and the
    # smallest at 8.4%), which is the range S5.3 quotes for the clause-(ii) failures. A different range
    # over a different set, so it is registered separately rather than reusing imp_lo/imp_hi -- reusing
    # them is how "three of them by 31--42%" and "four, by 8--42%" would silently become one claim.
    # "with every seed agreeing" is the unanimity the sentence rests on, and no group can carry it.
    _unan = sorted((abs(r["forg_b"]), r["ref"]) for r in _pass
                   if r["forg_b"] < 0 and r["forg_b_pos"] == 0)
    R["imp_unan_lo"], R["imp_unan_hi"] = _unan[0][0], _unan[-1][0]
    assert len(_unan) == R["n_imp_unan"] == 4, (
        f'S5.3 says four of the seven survivors improve with every seed agreeing; the records now '
        f'give {len(_unan)}: {[r for _, r in _unan]}')

    # -- how many of the four selection-vs-held-out sign reversals are themselves gate survivors.
    # S7 leans on this overlap to argue the reversals are not confined to cells nobody would inspect,
    # so it has to move with both the reversal set and the gate split.
    R["n_rev_gate_pass"] = sum(
        1 for r in rows
        if r["bd_val"] is not None and r["bd_test"] is not None
        and (r["bd_val"] > 0) != (r["bd_test"] > 0)
        and r["gate"] is not None and r["gate"] >= 0.20)

    # -- the pre-registered LOCO ladder. Read from the emitter's JSON rather than recomputed here: the
    # bootstrap figures are only reproducible at the emitter's pinned TEX_BOOT/TEX_SEED, so a second
    # implementation in this file would disagree with the table on the third decimal and the
    # disagreement would be in the checker, not in the paper. The whole ladder is registered, not just
    # dR2, because the body quotes four of the five rungs and the rungs are what make the point.
    _loco = json.load(open(ROOT / "results/cka_loco.json"))
    for s in ("M0", "M1", "M2", "M3", "M4"):
        R[f"loco_r2_{s.lower()}"] = _loco["r2_loco"][s]
    R["loco_dr2"] = _loco["dr2"]
    R["loco_folds"] = _loco["n_folds"]
    R["loco_frac"] = round(_loco["dr2_frac_not_helping"] * 100)
    R["loco_b1"] = _loco["cka_coef_m3_insample"]
    R["loco_b1_lo"], R["loco_b1_hi"] = _loco["cka_coef_ci"]
    # The sign of dR2 IS the pre-registered deciding statistic, so the paper's "it does not help"
    # framing is false the moment this flips. No captured group can carry that, hence the assert.
    assert R["loco_dr2"] <= 0, (
        f'the registered prediction no longer holds: dR2 = {R["loco_dr2"]:+.3f} > 0. The body and '
        f'Appendix app:loco assert the dR2 <= 0 branch; switch to the branch the registration '
        f'pre-wrote for a positive dR2 instead of editing the number.')
    assert _loco["branch"] == "dR2 <= 0", f'unexpected branch: {_loco["branch"]!r}'

    # -- the pre-registered positive control. Read from the emitter's JSON, like the LOCO ladder and
    # for the same reason: the retention numbers come out of 21 stored run records through
    # emit_positive_control.py's own scaling, and a second implementation here would disagree with
    # Table tab:poscontrol in the last decimal. What is registered here is every number the body and
    # app:poscontrol state, plus three asserts on the claims no captured group can carry.
    _pc = json.load(open(ROOT / "results/positive_control.json"))
    R["pc_v_ridge"] = _pc["task_a"]["v_ridge_selection"]
    R["pc_v_rung"] = _pc["task_a"]["v_best_admissible_rung"]
    R["pc_sn_ratio"] = _pc["task_b"]["seasonal_naive_over_its_negation"]
    R["pc_horizon"] = _pc["task_a"]["horizon"]
    _val = _pc["validity_taskb_learned"]
    R["pc_n_runs"] = _val["n_runs"]
    R["pc_n_learned"] = _val["n_meeting"]
    R["pc_learn_min"] = _val["min_pct"]
    R["pc_learn_thresh"] = _val["threshold"]
    _des = _pc["destruction"]["primary"]
    R["pc_ret_max"] = max(v["max"] for v in _des["per_rung"].values())
    # The per-rung maxima app:poscontrol lists, in ascending learning rate so the tuple order in the
    # prose is the ladder's order rather than a dict's.
    for lr, key in (("0.0001", "r1"), ("0.001", "r2"), ("0.01", "r3"), ("0.1", "r4")):
        R[f"pc_max_{key}"] = _des["per_rung"][lr]["max"]
    R["pc_des_thresh"] = _pc["destruction"]["threshold_pct"]
    R["pc_noise_max"] = _pc["evaluator_noise_pp"]["max"]
    R["pc_noise_mean"] = _pc["evaluator_noise_pp"]["mean"]
    R["pc_noise_pairs"] = _pc["evaluator_noise_pp"]["n_identical_state_pairs"]
    R["pc_rho"] = _pc["cka_ordering"]["primary"]["rho"]
    R["pc_rho_p"] = _pc["cka_ordering"]["primary"]["p"]
    R["pc_rho_n"] = _pc["cka_ordering"]["primary"]["n"]
    _top, _bot = _pc["by_rung"]["0.1"]["B"], _pc["by_rung"]["0.0001"]["D"]
    R["pc_top_cka"] = _top["cka_task_a"]["mean"]
    R["pc_top_drift"] = _top["drift"]["mean"]
    R["pc_top_ret"] = abs(_top["retention_primary"]["mean"])
    R["pc_top_ret_sem"] = _top["retention_primary"]["sem"]
    R["pc_bot_cka"] = _bot["cka_task_a"]["mean"]
    R["pc_bot_drift"] = _bot["drift"]["mean"]
    # The three frozen-encoder rungs app:poscontrol quotes for validity condition (ii).
    for lr, key in (("0.0001", "r1"), ("0.001", "r2"), ("0.01", "r3")):
        R[f"pc_d_{key}"] = _pc["by_rung"][lr]["D"]["retention_primary"]["mean"]
        R[f"pc_d_{key}_sem"] = _pc["by_rung"][lr]["D"]["retention_primary"]["sem"]
    R["pc_n_diverged"] = len(_pc["diverged_cells"])
    # The extension was registered as one rung under BOTH conditions, so its per-condition seed count
    # is half the planned total. app:poscontrol's "diverged in 3 of 3" is a per-condition fraction and
    # would read as 3 of 6 if this took the planned count directly.
    R["pc_ext_seeds"] = _pc["extension_runs_planned"] // 2
    R["pc_seeds"] = _pc["by_rung"]["0.0001"]["B"]["n_runs"]
    assert _pc["extension_lr"] == 0.1, (
        f'both sites write the declared extension as lr = 10^-1; the emitter now reports '
        f'{_pc["extension_lr"]}')
    # Which outcome fired IS the section's thesis, and the body states it in words ("the third",
    # "destruction was not achieved") that no captured group can check. Three asserts instead.
    # str(), because the emitter carries the outcome as the registration's own key, which is a JSON
    # object key and therefore a string. `== 3` compared int to str and failed while the outcome was
    # in fact 3 -- an assert that fires on the branch it is meant to permit is worse than none.
    assert str(_pc["outcome"]) == "3", (
        f'S7.3 and app:poscontrol are written for registered outcome 3, but the emitter now reports '
        f'outcome {_pc["outcome"]} ({_pc["outcome_key"]}). Rewrite to the branch the registration '
        f'pre-wrote for that outcome instead of editing numbers.')
    assert not _pc["outcome_provisional"], (
        'the positive control is still provisional -- the table carries a PARTIAL GRID banner -- so '
        'the body must not state the outcome as final')
    assert _val["all_meet"], (
        '"task B was learned in every run" is false, so the arm measures weight thrashing rather '
        'than forgetting and the registered validity condition has failed')
    # Condition D preserving task A at every rung it completed is app:poscontrol's validity condition
    # (ii); "inside +5%" is a quantifier over three rungs, so it is an assert.
    assert all(R[f"pc_d_{k}"] < 5.0 for k in ("r1", "r2", "r3")), (
        '"retention stays inside +5% at all three completed rungs" is false: '
        f'{[round(R[f"pc_d_{k}"], 1) for k in ("r1", "r2", "r3")]}')
    # Every seed at the top rung improves task A -- that is what licenses "in every seed".
    assert _des["per_rung"]["0.1"]["max"] < 0, (
        f'"task A gets better in every seed at the top rung" is false: the worst seed is '
        f'{_des["per_rung"]["0.1"]["max"]:+.1f}%')

    # -- Chronos-T5-Small on M4-Monthly. This block is the point of the whole file: app:chronos_detail
    # was ~140 lines of hand-typed numbers with no run record and no check here, for ten rounds. Every
    # number that section now states is registered below, and the six runs, the gate ladder and the
    # diagnostic reproduction of the superseded figure are the only sources.
    _m4g = json.load(open(ROOT / "results/chronos_m4/gate.json"))
    _m4sel, _m4ho = _m4g["splits"]["selection"], _m4g["splits"]["heldout"]
    R["m4_gate"] = _m4sel["primary_r2_task"]
    R["m4_trend"] = _m4sel["rungs"]["trend"]["r2_task"]
    R["m4_trend_mse"] = _m4sel["rungs"]["trend"]["baseline_mse"]
    R["m4_const_mse"] = _m4sel["constant_floor_mse"]
    R["m4_best_rung"] = _m4sel["rungs"][_m4sel["best_admissible_rung"]]["r2_task"]
    R["m4_worst_rung"] = _m4sel["worst_case_r2_task"]
    R["m4_gate_ho"] = _m4ho["primary_r2_task"]
    R["m4_mlp_ho"] = _m4ho["rungs"]["mlp"]["r2_task"]
    R["m4_train_windows"] = _m4g["data"]["train_windows_used"]
    R["m4_dropped"] = _m4g["data"]["degenerate_train_contexts_dropped"]
    R["m4_series"] = _m4g["data"]["n_series"]
    R["m4_minlen"] = _m4g["data"]["min_len"]
    # The two claims the appendix makes in words rather than in digits, and that no captured group
    # could carry: that the primary gate FAILS, and that the superseded denominator is inadmissible.
    # If either flips, the section's argument inverts and the numbers would still check out.
    assert not _m4sel["primary_passes"], (
        f'app:chronos_detail is written for a gate-FAIL on this cell, but the primary gate is now '
        f'{_m4sel["primary_r2_task"]:+.4f} at threshold {_m4g["threshold"]}. The registration '
        f'pre-wrote the gate-clears branch; switch to it rather than editing the number.')
    assert not _m4sel["rungs"]["trend"]["admissible"], (
        'app:chronos_detail asserts the superseded trend denominator is worse than the constant '
        'floor on the selection split; it is no longer')
    assert _m4sel["best_admissible_rung"] == "seasonal_naive", (
        f'the appendix names the seasonal naive as the best admissible rung; it is now '
        f'{_m4sel["best_admissible_rung"]}')

    # The diagnostic reproduction of the superseded 84.5%. Written after the gate and unable to move
    # it; what it supports is the appendix's claim that the old figure does not reproduce.
    _m4l = json.load(open(ROOT / "results/chronos_m4/legacy_baseline.json"))
    R["m4_legacy_pct"] = 100 * _m4l["legacy"]["r2_task"]
    R["m4_legacy_mse"] = _m4l["legacy"]["baseline_mse"]
    R["m4_legacy_const"] = _m4l["constant_floor"]["baseline_mse"]
    R["m4_legacy_fitted_mse"] = _m4l["fitted"]["baseline_mse"]
    R["m4_legacy_fitted_pct"] = 100 * _m4l["fitted"]["r2_task"]
    R["m4_legacy_overparam"] = _m4l["legacy"]["n_series_with_fewer_windows_than_coefficients"]
    R["m4_legacy_minlen"] = _m4l["legacy_selection"]["min_len"]
    R["m4_legacy_hist_min"] = _m4l["legacy_selection"]["history_min"]
    R["m4_legacy_hist_max"] = _m4l["legacy_selection"]["history_max"]
    R["m4_lookback"] = _m4g["lookback"]
    # "it is not that the denominator was too weak in the ladder's sense" rests on this: on the legacy
    # window set the legacy estimator is ADMISSIBLE. Stating the opposite would be the easy version of
    # this paragraph and it would be false.
    assert _m4l["legacy_admissible"], (
        'the appendix says the legacy estimator beats the training mean on its own window set; it no '
        'longer does, so that paragraph now understates the defect and must be rewritten')

    # The six runs, through the emitter's own summary so the prose and Table tab:chronosm4runs cannot
    # disagree in the last decimal.
    _m4runs = {}
    for _c in ("B", "D"):
        for _s in (42, 43, 44):
            _m4runs[(_c, _s)] = json.load(open(
                ROOT / f"results/chronos_m4/cond_{_c}/seed{_s}/condition_{_c}_s{_s}.json"))
    _fb = [_m4runs[("B", s)]["forgetting_pct"] for s in (42, 43, 44)]
    _fd = [_m4runs[("D", s)]["forgetting_pct"] for s in (42, 43, 44)]
    _dd = np.array(_fb) - np.array(_fd)
    R["m4_forg_b"] = abs(float(np.mean(_fb)))
    R["m4_forg_d"] = abs(float(np.mean(_fd)))
    R["m4_denc"] = float(_dd.mean())
    _sem = float(_dd.std(ddof=1) / np.sqrt(len(_dd)))
    _t = float(stats.t.ppf(0.975, len(_dd) - 1))
    R["m4_ci_lo"] = R["m4_denc"] - _t * _sem
    R["m4_ci_hi"] = R["m4_denc"] + _t * _sem
    R["m4_mde"] = _t * _sem
    R["m4_cka_b_lo"] = min(_m4runs[("B", s)]["final_cka"] for s in (42, 43, 44))
    R["m4_cka_b_hi"] = max(_m4runs[("B", s)]["final_cka"] for s in (42, 43, 44))
    R["m4_l2_lo"] = min(_m4runs[("B", s)]["weight_drift"] for s in (42, 43, 44))
    R["m4_l2_hi"] = max(_m4runs[("B", s)]["weight_drift"] for s in (42, 43, 44))
    R["m4_stop_lo"] = min(_m4runs[("B", s)]["stopped_epoch"] for s in (42, 43, 44))
    R["m4_stop_hi"] = max(_m4runs[("B", s)]["stopped_epoch"] for s in (42, 43, 44))
    R["m4_n_train"] = _m4runs[("B", 42)]["n_train_windows"]
    for _i, _s in enumerate((42, 43, 44), start=1):
        R[f"m4_best_epoch_{_i}"] = _m4runs[("B", _s)]["best_epoch"]
    # The cell's configuration, from the record rather than from the caption it is printed in. Every
    # one of these was a hand-typed constant in the deleted version of the section.
    _b42 = _m4runs[("B", 42)]
    R["m4_horizon"] = _b42["horizon"]
    R["m4_batch"] = _b42["batch_size"]
    R["m4_epochs"] = _b42["epochs_requested"]
    R["m4_patience"] = _b42["patience"]
    R["m4_cka_windows"] = _b42["cka_windows"]
    R["m4_lr_exp"] = int(round(-math.log10(_b42["lr"])))
    # The tail each series holds back from the training pool: one selection window (lookback context
    # plus horizon target). Derived, not typed, so it cannot disagree with the two constants above.
    R["m4_heldback"] = R["m4_lookback"] + R["m4_horizon"]
    # The zero-shot numerator's sample count is NOT in gate.json: the gate was computed before this
    # check existed, and re-running it to add the field would move every number in the section, because
    # Chronos forecasts by sampling. So it is read from the literal in the script that produced the
    # record, which ships in the release, and required to be single-valued.
    _zs_src = (ROOT / "scripts/gate_chronos_m4.py").read_text()
    _ns = set(re.findall(r"num_samples=(\d+)", _zs_src))
    assert len(_ns) == 1, f"gate_chronos_m4.py draws a varying number of samples: {sorted(_ns)}"
    R["m4_num_samples"] = int(_ns.pop())
    # The two file:line citations the section makes about the superseded script. A line citation rots
    # silently -- it stays syntactically valid while pointing at something else -- so each is anchored
    # on the text it is supposed to name.
    _old_src = (ROOT / "scripts/finetune_chronos_m4.py").read_text().split("\n")
    for _ln, _needle in ((303, "def linear_baseline_mse"), (505, "best_val_mse = zs_mse"),
                         (507, "best_state = copy.deepcopy"), (522, "val_mse = chronos_zs_mse")):
        assert _needle in _old_src[_ln - 1], (
            f"app:chronos_detail cites finetune_chronos_m4.py:{_ln} for {_needle!r}; that line now "
            f"reads {_old_src[_ln - 1].strip()!r}")
    # "cut a randomly subsampled window pool at the 80th percentile of its own index" is a claim about
    # that script, so it is checked against that script and not against a number.
    assert "n_val = max(int(n_total * 0.2), 10)" in "\n".join(_old_src), (
        'the appendix says the superseded script cut its window pool at the 80th percentile; it no '
        'longer does')
    # The one comparison the surviving descriptive observation makes. It used to be an unrecorded CKA
    # for a Moirai-Small n=10k arm that no emitter reads; it is now a cell of the 31-cell matrix, which
    # is also the cell Table tab:dissociation prints first.
    R["m4_moirai_cka"] = _by["Moirai-base/ETTh2 h192 n1000"]["cka"]
    # The operating point, from the primary gate's own record rather than from any one arm's. It had no
    # prose check at all before this round -- a constant stated at four sites and derived from none of
    # them -- and the M4 arm is required to have screened against the same value.
    R["gate_threshold"] = json.load(open(ROOT / "results/gate_baselines_val.json"))["gate_threshold"]
    # Three sources, tied together: the constant the screen is coded against, the value the stored
    # gate artifact was actually computed at, and the value the M4 arm screened at. This assert is
    # why the earlier read of gate_all_cells.GATE_THRESHOLD is kept under its own key rather than
    # overwritten here -- a code change that the caches predate would otherwise pass silently.
    assert _m4g["threshold"] == R["gate_threshold"] == R["gate_threshold_code"], (
        f'the Chronos/M4 gate screened at {_m4g["threshold"]}, the stored gate artifact at '
        f'{R["gate_threshold"]}, and gate_all_cells.GATE_THRESHOLD is {R["gate_threshold_code"]}')
    # The superseded script's train/val fraction, read from the script so the appendix's "80th
    # percentile" cannot drift from the code it describes.
    _frac = re.search(r"n_val = max\(int\(n_total \* ([\d.]+)\)", "\n".join(_old_src))
    R["m4_old_split_pct"] = int(round(100 * (1 - float(_frac.group(1)))))
    # Both arms improving, and every run early-stopping past epoch 0, are the two claims the section
    # makes in words. The second is what the old arm could not say.
    assert np.mean(_fb) < 0 and np.mean(_fd) < 0, (
        f'the appendix says both arms improve on zero-shot on average; they now read '
        f'B {np.mean(_fb):+.2f}%, D {np.mean(_fd):+.2f}%')
    assert all(r["best_epoch"] >= 1 for r in _m4runs.values()), (
        'a run selected the pre-trained checkpoint, which is the defect this arm was re-run to fix')

    # --- the n=10k sign fragility on Moirai-Small/ETTh2 h96 ---------------------------------------
    # WHY THIS IS DERIVED, AND WHY IT DELIBERATELY STOPS SHORT OF A DECOMPOSITION. The appendix has
    # long printed an earlier three-seed record of this cell (results/v16_etth2_n10k, +10.6%) beside
    # the reported ten-seed one (results/v19_cuda_etth2_n10k, -5.3%) and attributed the reversal to
    # "protocol dependence" -- i.e. to reading the final epoch instead of the best-validation one.
    # That attribution is not identified, and the records say so: v16 differs from v19 in FOUR ways
    # at once, not one. (1) seeds -- {101,456,789}, a subset of v19's ten that happens to contain two
    # of its three most positive seeds; (2) epoch budget -- 20 against 10; (3) the zero-shot
    # reference the percentage is taken against -- 0.1325 against 0.1266, so the denominators differ;
    # (4) the stopping rule. Picking a different early-stopped baseline flips which factor looks
    # dominant: against v19 the seed-matched subset reads -2.0% (so 3.3 pp seed, 12.6 pp residual),
    # against results/v18_mps_deterministic_n10k it reads +4.1% (9.8 pp seed, 6.5 pp residual). Two
    # baselines, opposite verdicts, which is what "not identified" means. So the body and the
    # appendix state the REVERSAL and the confounds and claim no cause, and the asserts below fail if
    # either the reversal or any of the four differences stops holding.
    _pro = {}
    for _tag, _dir in (("fe3", "v16_etth2_n10k"), ("es10", "v19_cuda_etth2_n10k"),
                       ("es10mps", "v18_mps_deterministic_n10k")):
        _fs = sorted((ROOT / "results" / _dir).glob("*/condition_B_h96_s*.json"))
        _pro[_tag] = {(_r := json.load(open(_f)))["seed"]: _r for _f in _fs}
    R["pro_fe3"] = float(np.mean([r["forgetting_pct"] for r in _pro["fe3"].values()]))
    R["pro_es10"] = float(np.mean([r["forgetting_pct"] for r in _pro["es10"].values()]))
    R["pro_n_fe3"], R["pro_n_es10"] = len(_pro["fe3"]), len(_pro["es10"])
    R["pro_fe3_sd"] = float(np.std([r["forgetting_pct"] for r in _pro["fe3"].values()], ddof=1))
    R["pro_fe3_seeds"] = sorted(_pro["fe3"])
    # The two seed-matched readings the appendix prints to show the attribution is not identified.
    # Both are the SAME three seeds under the other record's protocol, which is the only reason they
    # bound anything; taken over different seed sets they would just be two more unmatched numbers.
    for _tag, _key in (("es10", "pro_sub3_cuda"), ("es10mps", "pro_sub3_mps")):
        R[_key] = float(np.mean([_pro[_tag][s]["forgetting_pct"] for s in _pro["fe3"]]))
    # The two zero-shot references, which is the confound a reader is least likely to guess at.
    R["pro_zs_fe3"] = _pro["fe3"][R["pro_fe3_seeds"][0]]["zeroshot_mse"]
    R["pro_zs_es10"] = _pro["es10"][R["pro_fe3_seeds"][0]]["zeroshot_mse"]
    R["pro_ep_fe3"] = _pro["fe3"][R["pro_fe3_seeds"][0]]["epochs"]
    R["pro_ep_es10"] = _pro["es10"][R["pro_fe3_seeds"][0]]["epochs"]
    # The appendix says the subset "happens to contain two of the three most positive" seeds of the
    # ten. That is a property of the reported record and is the reason the subset is not a fair draw,
    # so it is computed rather than asserted.
    _rank = sorted(_pro["es10"], key=lambda s: -_pro["es10"][s]["forgetting_pct"])
    assert len(set(_rank[:3]) & set(_pro["fe3"])) == 2, (
        f'the appendix says the final-epoch seed set holds two of the three most positive seeds of '
        f'the reported ten; the top three are now {_rank[:3]} against {sorted(_pro["fe3"])}')
    # The reversal is the claim the body makes; if it stops holding the sentence is wrong, not stale.
    assert R["pro_es10"] < 0 < R["pro_fe3"], (
        f'S5.2 says the n=10k sign reverses between these two records; they now read '
        f'v19 {R["pro_es10"]:+.2f}%, v16 {R["pro_fe3"]:+.2f}%')
    # The three differences the corrected prose names, each checked against the records rather than
    # asserted in prose. "at once" is the whole point of the sentence, so all three must hold.
    assert set(_pro["fe3"]) < set(_pro["es10"]), (
        'the prose says the final-epoch record covers a subset of the reported seeds; it no longer does')
    _e3 = {r["epochs"] for r in _pro["fe3"].values()}
    _e10 = {r["epochs"] for r in _pro["es10"].values()}
    assert len(_e3) == len(_e10) == 1 and _e3 != _e10, (
        f'the prose says the two records differ in epoch budget; they now read {_e3} and {_e10}')
    assert (abs(next(iter({r["zeroshot_mse"] for r in _pro["fe3"].values()}))
                - next(iter({r["zeroshot_mse"] for r in _pro["es10"].values()}))) > 1e-6), (
        'the prose says the two records take their percentage against different zero-shot '
        'references; those references now agree, so that clause must be dropped')
    return R


# ---------------------------------------------------------------- the registry

def build_checks(R):
    """(name, pattern, expected-per-group). One entry per claim the body makes about a number."""
    C = []

    def chk(name, pattern, *expect, min_sites=1):
        C.append(dict(name=name, pattern=pattern, expect=expect, min_sites=min_sites))

    # --- denominators
    # "held-out cells" was retired where "held-out" modified the screening set rather than the
    # outcome: the gate is scored on selection windows, so calling the screened cells held-out
    # was the ambiguity a reviewer caught. The three bold sites (abstract, intro, S4) remain.
    chk("screened-cell count", r"\\textbf\{(\d+) (?:screened |held-out )?cells\}",
        R["n_screened"], min_sites=3)
    chk("screened-cell count (prose)", r"(?:Of|across) (\d+) screened cells", R["n_screened"])
    # The one sentence that states the 31-vs-32 relation rather than one side of it. S4's wording
    # moved on 20 Sep 2026 (", so the screen's denominator" became an em-dash clause naming the cell),
    # so the pattern follows it: both numbers in one site is what stops the two denominators being
    # swapped, which is the failure the comment below describes.
    chk("selection-split denominator (S4)", r"which exist for (\d+) of the (\d+)",
        R["n_val_scored"], R["n_screened"])
    # S4's four-way split of the 32. One site, four groups, because the arms' ORDER is the claim: with
    # four separate checks the sentence could name 5 Chronos cells and 5 TimesFM cells the other way
    # round and every check would still pass. The Moirai parenthetical's "three, five, two" is asserted
    # in rederive() over the arm's own keys instead of captured here -- see the comment there.
    chk("the screen's four arms",
        r"screened cells\}---(\d+) Moirai \(three capacities, five datasets, two horizons\), (\d+)\s*"
        r"Chronos-T5, (\d+) TimesFM-2\.5 and (\d+) Moirai/ILI",
        R["n_arm_moirai"], R["n_arm_chronos"], R["n_arm_timesfm"], R["n_arm_ili"])
    # S1's three "31"s. Each is a RESTATEMENT of the selection-split denominator in wording no existing
    # pattern reaches, and this paper's recorded failure is precisely a restatement drifting while one
    # canonical site stays green. Two of the three carry a second number, so they capture the relation:
    # "seven of 31" and "2 of 31" are the two counts a reader takes away from the intro.
    chk("the headline count (S1's three clauses)",
        r"\\textbf\{(\w+) of the (\d+) scored cells\}", R["n_gate_pass"], R["n_val_scored"])
    chk("the freeze-decisive count (S1's three clauses)",
        r"only (\d+) of the (\d+) favour freezing", R["n_freeze_dec"], R["n_paired_tests"])
    chk("the funnel's denominator (Figure 1's caption)",
        r"funnel over the (\d+) cells that carry a", R["n_val_scored"])
    # The seed budget, as endpoints over the intervention rows. Three sites and three phrasings: S1's
    # design paragraph, the limitations list ("3 to 10 per cell") and the reproducibility statement
    # ("cells at 3--10 seeds"). One pattern loose enough to reach all three, because the endpoints are
    # one fact and the alternative is three patterns that rot independently.
    chk("the paired seed budget", r"(\d+)(?:--| to )(\d+) (?:seeds|per cell)",
        R["seeds_lo"], R["seeds_hi"], min_sites=3)
    # The BH level, at all three sites that name it, against paired_inference.ALPHA -- the constant the
    # calls were actually made at. A pre-registered level stated in prose and not tied to the code is a
    # number that can drift in the one direction nobody checks.
    chk("the BH level", r"multiplicity-adjusted \$q\{<\}([\d.]+)\$", R["bh_alpha"], min_sites=2)
    chk("the BH level (appendix)", r"called decisive only at \$q\{<\}([\d.]+)\$", R["bh_alpha"])

    # --- the backbones' sizes. S1's screen description and S4's capacity claim, which is the one that
    # had drifted: it said 6.3x, the ratio of the Moirai paper's rounded capacities (91M / 14.4M),
    # where the checkpoints we ran give 6.61. Both sites read scripts/emit_model_sizes.py's output.
    chk("the three Moirai capacities",
        r"Moirai Small ([\d.]+)M, Base (\d+)M, Large (\d+)M",
        R["params_small"], R["params_base"], R["params_large"])
    chk("the capacity ratio (S4)",
        r"Moirai-Base is ([\d.]+)\$\\times\$ larger than", R["base_over_small"])
    # Chronos on ETTh2, both splits in one site: the claim is that the sign survives the split change.
    chk("Chronos on ETTh2, below the baseline on both splits",
        r"below\} the fitted baseline \(\$-([\d.]+)\$ on selection, \$-([\d.]+)\$ on\s*held-out\)",
        R["chronos_etth2_val"], R["chronos_etth2_test"])
    # "no cell of the N clears" appears on both splits with different N -- 31 in S4 and the
    # selection-split appendix, 32 in the retrospective one -- so one pattern spanning both would
    # have to agree with two numbers. Split in two, under "no cell clears every admissible rung
    # (selection split)" and "union count (retrospective)" below, each anchored on wording only its
    # own split uses.
    chk("intervention-cell count", r"(?:all |the )(\d+) intervention cells",
        R["n_int"], min_sites=3)
    # "intervention-cell count (paired run)" is retired on 22 Sep 2026 (C2). Its only site was S5.1's
    # clause naming the drift analyses' 31-cell scope, which was the THIRD statement of one scope (S4's
    # "One rung of eight" paragraph and S6's opening state it too) and was cut for the page limit. The
    # number keeps three prose checks: "intervention-cell count" at 3 sites, "(every denominator)" at
    # S4's site, and "(conclusion)". Retired rather than retargeted, because no surviving sentence
    # phrases 31 as the PAIRED-run count -- and inventing a sentence to satisfy a chk is backwards.
    chk("intervention-cell count (every denominator)",
        r"every\s+denominator below is (\d+)", R["n_int"])
    chk("intervention-cell count (pooled)", r"Pooled across all (\d+) cells", R["n_int"])
    chk("intervention-cell count (conclusion)", r"across the (\d+) cells", R["n_int"])
    # Every gate count below is the SELECTION-split primary, so its denominator is n_val_scored
    # (31), not n_screened (32). The screen's own 32 is checked by "32 screened cells" above; the
    # one cell of difference -- TimesFM/Electricity, no paired run and so no selection reference --
    # is why these two numbers must not share a record.
    chk("gate-passing count", r"only (\d+) of (\d+) cells show a pre-trained advantage",
        R["n_gate_pass"], R["n_val_scored"], min_sites=2)
    # The conclusion used to open by re-enumerating the screen ("7 of 31 cells beat a lookback-96
    # regression fit ... all seven on one backbone and two datasets"). Cut for page room: it was the
    # fourth restatement of a count the abstract, the intro lead and S4 all pin above, and the
    # conclusion's own content is the prescription that follows it. No site remains, so the check is
    # retired rather than left to fail -- the fact itself is still covered four times.
    # The abstract and the introduction's lead. These two sentences are where the stale "five cells,
    # all ETTh2" survived longest, so each of the three quantities they state -- the count, the
    # per-dataset composition, the improvement range -- is pinned separately rather than as one phrase.
    # The abstract was rewritten to the short version on 20 Sep 2026; this claim survived the cut but
    # its clause did not ("that carry a paired intervention run" moved to S4), so the pattern follows
    # the surviving sentence. Both numbers still in one site, for the same reason as above.
    chk("gate-passing count (abstract)",
        r"only \\textbf\{(\d+) of the (\d+)\} scored cells show the",
        R["n_gate_pass"], R["n_val_scored"])
    chk("gate-passing count (intro lead)",
        r"\\textbf\{(\w+) of the (\d+) scored\s+cells clear it\}",
        R["n_gate_pass"], R["n_val_scored"])
    # "all five of its ETTh2 cells and two of its four on ETTm2" -- three numbers, and the word "all"
    # is a fourth claim, asserted in rederive() rather than matched here.
    # One site, not two: the short abstract states the composition as "all seven are Moirai on two of
    # the six datasets" and leaves the per-dataset split to the intro, which still carries it in full.
    # The three numbers keep two further registered sites below ("survivor composition (contribution 1
    # and S4)"), so lowering the floor tracks a deliberate cut rather than loosening the check.
    chk("survivor composition by dataset",
        r"all (\w+) of its\s+ETTh2 cells and (\w+) of its (\w+) on ETTm2",
        R["n_pass_etth2"], R["n_pass_ettm2"], R["n_scored_ettm2"])
    # Two sites -- contribution 1 (comma) and S4's lead (colon). Deliberately one pattern over both:
    # they are the same claim, and the failure mode being guarded is one of them moving alone.
    chk("survivor composition (contribution 1 and S4)",
        r"all seven are Moirai[,:]\s+(\w+) on ETTh2 and (\w+) on ETTm2",
        R["n_pass_etth2"], R["n_pass_ettm2"], min_sites=2)
    # The emphasis is optional as of 20 Sep 2026: the abstract keeps \emph{improves} and the intro's
    # result paragraph puts the emphasis on "helps" a clause earlier and writes this one plain. Both
    # sites are still required -- the count is the one the whole "adaptation usually helps" reading
    # rests on, and it was the abstract's copy that went stale last time.
    chk("survivors improved on the mean",
        r"(?:\\emph\{improves\}|improves) MSE (?:in|on) (\w+)", R["n_imp_mean"], min_sites=2)
    # One site: the abstract's copy of the range was cut with the rest of the short-abstract trim.
    # Contribution 1 still states it, under its own check immediately below.
    chk("the big improvers' range",
        r"([\d.]+)--([\d.]+)\\% on the (\w+) ETTh2 cells",
        R["imp_lo"], R["imp_hi"], R["n_imp_big"])
    chk("the big improvers' range (contribution 1)",
        r"(\w+) of them by ([\d.]+)--([\d.]+)\\%", R["n_imp_big"], R["imp_lo"], R["imp_hi"])
    # S5.3's range over all FOUR unanimous improvers, which is a different set and a different range
    # from the three ETTh2 cells above. It was the last unchecked claim in 05_exp2_adaptation.tex that
    # was not a horizon, an n, or a confidence level.
    chk("the unanimous improvers' range (clause (ii) failures)",
        r"\\emph\{improves\} MSE, by (\d+)--(\d+)\\% with every seed agreeing",
        R["imp_unan_lo"], R["imp_unan_hi"])
    # The abstract's ladder-union clause was cut with the short-abstract rewrite. n_union keeps three
    # registered sites -- "ladder union and its backbones" (S4 and app:baselines:val, min_sites=2) and
    # "union count (limitations)" -- and all three carry the backbone composition the abstract did not,
    # so the fact is better covered after the cut than before it. Retired, not loosened.

    # --- the third window set (S4's closing paragraph and app:baselines:traintail)
    # S4's train-tail paragraph moved to app:baselines:traintail on 20 Sep 2026, so these three now
    # read the appendix's own wording. The arm size is stated once there, in the sentence that names
    # the script and the JSON, which is the site worth pinning.
    chk("train-tail arm size", r"arm covers the (\d+) Moirai cells", R["n_traintail"])
    chk("train-tail fraction", r"the last \$(\d+)\\%\$ of (?:each|the)", 20, min_sites=2)
    # The body's copy of the fitted-rung transfer is gone with the paragraph; the appendix sentence it
    # was restating is registered immediately below and carries all four numbers.
    chk("fitted rung, selection vs train tail (appendix)",
        r"it admits \\textbf\{(\d+) cells on the selection windows and (\d+) on the\s+"
        r"train tail, with only (\d+) in both\}",
        R["n_tt_fitted_sel"], R["n_tt_fitted_tail"], R["n_tt_fitted_both"])
    chk("the three zeros", r"\\textbf\{(\d+), (\d+) and\s+(\d+) of (\d+)\}",
        *R["n_tt_all_admissible"], R["n_traintail"])
    chk("the union on the two splits", r"The union moves the other way, (\d+) to\s+(\d+)",
        R["n_tt_any_sel"], R["n_tt_any_tail"])
    chk("the ridge's own-region advantage",
        r"the fitted ridge scores \$([\d.]+)\$ on the selection windows\s+"
        r"and \$([\d.]+)\$ on the train tail---\$([\d.]+)\\times\$ better---while zero-shot barely "
        r"moves,\s+\$([\d.]+)\$ to \$([\d.]+)\$",
        R["tt_ridge_val"], R["tt_ridge_tail"], R["tt_ridge_ratio"], R["tt_zs_val"], R["tt_zs_tail"])
    chk("gate-passing count (Moirai arm)", r"Moirai: (\d+) of (\d+)\}",
        R["n_gate_pass"], R["n_moirai_screened"])
    chk("gate-failing count", r"(\d+) of (?:the|our) (\d+) cells we screened",
        R["n_gate_fail"], R["n_val_scored"])
    chk("gate-failing count (disqualifier)", r"(\d+) of our (\d+) fail",
        R["n_gate_fail"], R["n_val_scored"])
    # Limitation (1) used to spell the screen out again ("24 of 31 cells lose to a fitted lookback-96
    # ridge map, the seven that beat it are all Moirai ..."); it now points at S4 instead. Same reason
    # as the conclusion above, and the count keeps three other registered sites.
    # This pair sat UNREGISTERED and went stale: through 20 Sep 2026 S2 read "5 of our 32 screened
    # cells clear it, all of them ETTh2 on Moirai, and the remaining 27", which were the test-side
    # numbers, and it survived the selection-split flip because no pattern reached it -- "screened
    # cells clear it" is not "cells we screened", so the check above slid past it. Registering both
    # halves, and the pass/fail split must reconcile to n_val_scored.
    chk("gate counts (background)",
        r"\\textbf\{(\d+) of the (\d+) scored cells clear it\}",
        R["n_gate_pass"], R["n_val_scored"])
    chk("gate-failing count (background)", r"in the other (\d+) the pre-trained model",
        R["n_gate_fail"])
    # --- the n=10k sign fragility, added 2026-09-24. Every number in the two sentences that state it
    # gets a pattern, including the two confound values a reader would otherwise have to trust
    # (the epoch budgets and the two zero-shot references). The reason for that thoroughness is the
    # failure this file keeps having: the claim is ABOUT a record disagreeing with another record, so
    # an unregistered number here would be an unchecked number inside a paragraph whose whole subject
    # is unchecked numbers.
    chk("n=10k sign fragility (S5.2 body)",
        r"read at its final epoch gives \$\+([\d.]+)\\%\$\s+against the \$-([\d.]+)\\%\$ above",
        R["pro_fe3"], abs(R["pro_es10"]))
    chk("n=10k final-epoch record (appendix)",
        r"final-epoch runs gave \$\+\$([\d.]+)\$\\pm\$([\d.]+)\\% on \$k\{=\}(\d+)\$",
        R["pro_fe3"], R["pro_fe3_sd"], R["pro_n_fe3"])
    chk("n=10k fragility: the final-epoch seed set",
        r"its three seeds\s+\((\d+), (\d+), (\d+)\) are a subset of the (\w+)",
        *R["pro_fe3_seeds"], R["pro_n_es10"])
    chk("n=10k fragility: the epoch-budget confound",
        r"it trains for (\d+) epochs rather than (\d+)", R["pro_ep_fe3"], R["pro_ep_es10"])
    chk("n=10k fragility: the zero-shot-reference confound",
        r"zero-shot reference \(\$([\d.]+)\$ against \$([\d.]+)\$\)",
        R["pro_zs_fe3"], R["pro_zs_es10"])
    chk("n=10k fragility: the two seed-matched readings",
        r"the CUDA record reads\s+\$-\$([\d.]+)\\% and the MPS one \$\+\$([\d.]+)\\%",
        abs(R["pro_sub3_cuda"]), R["pro_sub3_mps"])
    # Also unregistered and also stale until 20 Sep 2026: S7 said the reversals overlap "the five
    # gate-passing cells". Both halves are derived in rederive() -- how many of the sign reversals are
    # gate-passing, and the survivor count they are a subset of -- so the overlap cannot drift from
    # either the reversal set or the gate.
    chk("reversals overlapping the survivors",
        r"(\w+) of the four are among the (\w+) gate-passing cells",
        R["n_rev_gate_pass"], R["n_gate_pass"])
    # The abstract acquired this overlap on 21 Sep 2026, because it is what makes the held-out
    # correction practical rather than pedantic: the reversals hit cells a reader would have acted on.
    # It says "cells that pass the screen" and not "gate-passing cells" -- the abstract does not use
    # the word gate -- so it is a site the check above cannot reach and needs its own pattern. This is
    # the exact shape of failure this file has had before: a restatement in fresh words that no
    # pattern matched, left to drift.
    chk("reversals overlapping the survivors (abstract)",
        r"and (\w+) of the four are cells that pass the screen", R["n_rev_gate_pass"])
    # S5.1 carried the survivor count as a SPELLED-OUT word in six places (title, lead, both
    # sub-headings, the "among the five" comparison, the sweep aside) and every one of them was still
    # the test-side five after the selection-split flip, because no pattern reached a bare word. These
    # five patterns are deliberately phrase-anchored rather than general: the point is that each site
    # is individually pinned, since that is the failure mode that actually occurred.
    # The S5 lead paragraph ("Experiment 1 leaves N cells...") was deleted for the page limit --
    # it restated the S5.1 lead sentence three lines later -- so its check is retired rather than
    # loosened. The count stays pinned at the six sites below.
    chk("survivor count (S5.1 subsection title and lead)",
        r"The (\w+) cells Experiment 1 admits are the entire evidence base", R["n_gate_pass"])
    chk("survivor count (S5.1 improver heading)",
        r"On (\w+) of the (\w+), full fine-tuning improves the checkpoint in every",
        R["n_imp_unan"], R["n_gate_pass"])
    chk("survivor count (S5.1 comparison)",
        r"representational change among the (\w+)", R["n_gate_pass"])
    chk("non-improver remainder (S5.1 heading)",
        r"On the remaining (\w+), the mean harm does not survive",
        R["n_gate_pass"] - R["n_imp_unan"])
    chk("survivor count (S5.2 sweep aside)",
        r"one of the (\w+) survivors, so a sweep", R["n_gate_pass"])
    chk("survivor count (S5.1 subsection title)",
        r"What Fine-Tuning Does to the (\w+) Survivors", R["n_gate_pass"])
    # The fourth improver, which the selection split added. It is the only survivor where the frozen
    # encoder does BETTER than full fine-tuning, so its two percentages must not be swapped.
    chk("the fourth improver (S5.1)",
        r"Moirai-Base/ETTm2 at \$h\{=\}192\$, improves by a smaller \$([\d.]+)\\%\$ and is the one "
        r"survivor where freezing does slightly better \(\$-([\d.]+)\\%\$\), at CKA \$([\d.]+)\$",
        R["imp4_b"], R["imp4_d"], R["imp4_cka"])
    chk("the split-sign survivor (S5.1)",
        r"a mean \$([\d.]+)\\%\$ improvement with (\d+) of (\d+) seeds harmful",
        R["split_forg_b"], R["split_pos"], R["split_seeds"])
    # The outcome range. This is a headline number -- it is in the abstract, the contributions and the
    # conclusion -- and it was UNREGISTERED until 20 Sep 2026, which is how the survivor counts around
    # it went stale unnoticed. The negative end is TimesFM/ETTm2 and the positive end is
    # Moirai-Small/Weather h192, so the pair also pins that the span is taken across all three arms.
    # Anchored on the word "outcome": an unanchored pair-of-percentages pattern also matches the
    # Chronos-only span ($-41.4$ to $+20.6$) three lines below the contributions site, which is a
    # different quantity over 5 cells rather than 31.
    # The outcome range, at all three sites that state it, under ONE check. It used to be three checks
    # over three fixed phrasings ("Outcomes from", "outcome still spans", "outcome spanning"), and
    # rewriting the abstract and the conclusion this round broke two of them at once while the fact
    # itself stayed correct everywhere -- which is the failure mode of pinning a sentence instead of a
    # claim. One pattern with the three verbs it is actually written with, and min_sites=3 so that
    # losing any single site still fails. The subject is required (an "outcome"/$\denc$ anchor): an
    # unanchored pair of percentages also matches the Chronos-only span three lines from the
    # contributions site, which is a different quantity over 5 cells rather than 31.
    chk("outcome span across the intervention cells",
        r"(?:[Oo]utcomes? (?:from|spanning|(?:still |nonetheless )?spans)|\\denc\$ spans) "
        r"\$-([\d.]+)\\%\$ to \$\{?\+\}?([\d.]+)\\%\$",
        # Two sites, not three, since the conclusion's copy was cut for the page limit: the abstract
        # and contribution 2 both state the span, which is where a reader meets it.
        R["forg_b_lo"], R["forg_b_hi"], min_sites=2)
    chk("clause-(i) exclusions", r"removes (\d+) of the (\d+)",
        R["n_clause_i_out"], R["n_int"])

    chk("Chronos span", r"\$-([\d.]+)\\%\$ on ETTm2 to \$\{?\+\}?([\d.]+)\\%\$ on ETTh1",
        R["chronos_min"], R["chronos_max"])
    # The contributions' Chronos clause was cut when the intro was restructured into five paragraphs.
    # chronos_span keeps its site at "Chronos span (point count)" below and the two ends keep theirs at
    # "Chronos span" above, both in S6 where the dissociation is argued.
    chk("Chronos span (point count)", r"a (\d+)-point spread", R["chronos_span"])
    # The two ends of the TimesFM range moved to Appendix Table tab:crossbackbone with the demotion of
    # the cross-backbone arm; what stays in S6 is the cell count and the quantifier, and the quantifier
    # is enforced by the assert in rederive() rather than by a pattern.
    chk("TimesFM arm (qualitative, with the count)",
        r"the (\w+) TimesFM-2\.5\s+cells sit at milder drift and all improve", R["n_timesfm"])

    # --- held-out scoring
    chk("held-out sign reversals",
        r"reverses its sign in (\d+) of the (\d+) intervention cells",
        R["n_heldout_rev"], R["n_int"])
    # One site, not two: contribution 3 was folded into contribution 2, which now names the
    # measurement choice without repeating the count the abstract and S7 both carry.
    chk("held-out sign reversals (abstract)",
        r"(\d+) of (\d+) cells (?:otherwise )?reverse",
        R["n_heldout_rev"], R["n_int"])
    # A third phrasing, in the boxed lesson. The other two patterns reached its numerator and left the
    # denominator uncovered -- and the denominator is what decides whether the lesson is a footnote or
    # the reason to read the paper, so it is the half that most needs a record.
    chk("held-out sign reversals (boxed lesson)",
        r"reverses its sign on (\d+) of our (\d+)\s+cells",
        R["n_heldout_rev"], R["n_int"])

    # --- the decomposition behind the reversals
    # The body sentence naming the largest reversal (Chronos/ETTh1, +6.8 to -39.2, a 46-point swing) is
    # cut on 22 Sep 2026 (C2) from S7.1, which spills onto page 10. It was the size of one row of
    # tab:heldout_decomp, in a paragraph whose claim is the DIRECTION of all four reversals. So the chk
    # follows the prose: the held-out value keeps a prose site in app:randominit, where the four
    # below-floor Chronos cells are given with their outcome span, and chron_val/chron_swing are now
    # carried by tab:heldout_decomp alone -- which scripts/heldout_decomposition.py emits from the run
    # records, so they are derived at build time rather than typed. A hand-typed copy is what this chk
    # existed to police; with no copy left there is nothing to police.
    chk("largest reversal (held-out, appendix)", r"outcomes span \$-([\d.]+)\$ to",
        R["chron_test"])
    chk("reversal mechanism",
        r"\(\$\\Delta\$ frozen \$\+([\d.]+)\$ against \$\\Delta\$ adapted \$\+([\d.]+)\$",
        R["chron_dd"], R["chron_db"])
    chk("selection/held-out difficulty ratio",
        r"ratio spans \$([\d.]+)\$ to \$([\d.]+)\$ across these four cells",
        R["ratio_lo"], R["ratio_hi"])

    # --- the relaxed-definition ladder. None of this was registered before 21 Sep 2026, which is how
    # the body, the appendix and the generated table came to disagree with the emitter in five places.
    chk("ladder maximum (body)",
        r"across (\d+) rungs from the weakest defensible one up to ours, at either gate "
        r"threshold, \\textbf\{the largest count anywhere is (\d+)\}",
        R["ds_n_rungs"], R["ds_max_020"])
    chk("ladder maximum among means-clause rungs (body)",
        r"clause admits at most (\d+), the 5/10 seed split", R["ds_max_means_020"])
    chk("ladder maximum (appendix)",
        r"the largest count anywhere on the ladder is \\textbf\{(\w+)\}", R["ds_max_020"])
    chk("ladder scorable and gate-passing counts",
        r"That leaves (\d+) scorable cells, of which (\d+) clear a gate of \$0\.10\$ and "
        r"(\d+) clear \$0\.20\$",
        R["ds_n_scorable"], R["ds_pass_010"], R["ds_pass_020"])
    chk("ladder rungs yielding zero",
        r"(\d+) of\s+the (\d+) rungs yield zero at both", R["ds_n_zero_020"], R["ds_n_rungs"])
    chk("ladder rungs yielding zero (restated)",
        r"at gate \$0\.20\$ (\d+) of the (\d+) rungs\s+also yield zero",
        R["ds_n_zero_020"], R["ds_n_rungs"])
    # "the five Chronos cells are excluded" is spelled as a word in the appendix, so it cannot carry a
    # captured group; the count is enforced here instead.
    assert R["ds_n_confounded"] == 5, \
        f'"the five Chronos cells are excluded" is now {R["ds_n_confounded"]}'

    # --- strict freeze
    chk("the CKA strict freeze reaches and the frozen encoder does not",
        r"the only Moirai condition at CKA exactly \$([\d.]+)\$", R["cka_unity"])
    chk("the same unity, as the value condition D's CKA is not",
        r"which is why its CKA is not\s*\n?\s*\$([\d.]+)\$", R["cka_unity"])
    chk("the four ETTm2 strict-freeze shifts",
        r"shifts of \$-([\d.]+)\$, \$-([\d.]+)\$, \$-([\d.]+)\$ and \$-([\d.]+)\$~pp",
        R["sf_s96_shift"], R["sf_b96_shift"], R["sf_s192_shift"], R["sf_b192_shift"])
    chk("the weakest ETTm2 cell, named as weak",
        r"Moirai-Base \$h\{=\}192\$ from \$\+([\d.]+)\{\\pm\}([\d.]+)\$ to "
        r"\$-([\d.]+)\{\\pm\}([\d.]+)\$",
        R["sf_b192_bd"], R["sf_b192_bd_sem"], R["sf_b192_bh"], R["sf_b192_bh_sem"])
    chk("strict-freeze coverage", r"closes that gap on (\d+) of the (\d+) cells",
        R["n_sf"], R["n_int"])
    chk("strict-freeze coverage (protocol)", r"on (\d+) of the (\d+) cells we ran it on",
        R["n_sf_reverse"], R["n_sf"])
    # Promoted from limitations item (2) into S5's opening this round, and registered on the way: the
    # claim is that the reversals hit cells the paper leans on, so it is the ETTm2 SURVIVOR count, and
    # it is only equal to it if every reversing cell is ETTm2 and every ETTm2 survivor was run under H.
    chk("survivors resting on the freeze boundary",
        r"(\d+) of which are among the seven survivors", R["n_pass_ettm2"])
    # "D alone" became "the ordinary freeze alone" under the naming convention of S3: the body no
    # longer asks a reader to decode a bare condition letter, so the pattern follows the name.
    chk("strict-freeze D-only remainder",
        r"and (\d+) cells rest on the ordinary freeze alone", R["n_d_only"])

    # --- the LoRA value-cell arm (app:loravaluecells). This section is hand-written prose over a
    # generated table, which is the configuration that has gone stale most often in this project, so
    # every derived number in it is registered here including the ones spelled as words.
    # The body's one LoRA sentence. It used to read "LoRA does [help]" off the single spectrum cell,
    # which is the generalisation the arm was run to test; registered here so the body's version of
    # the count cannot drift from the appendix's.
    chk("LoRA in the body: the one-cell reading, scoped",
        r"extending LoRA to (\w+) further value-cells leaves full fine-tuning ahead on "
        r"(\w+) of them",
        R["lora_cells"], R["lora_be_neg"])
    chk("LoRA arm: scope and protocol",
        r"We ran E on the (\w+) Moirai value-cells that lacked it, seeds "
        r"\$(\d+)/(\d+)/(\d+)\$",
        R["lora_cells"], R["lora_seed1"], R["lora_seed2"], R["lora_seed3"])
    chk("LoRA arm: drift is zero on every record",
        r"drift is exactly \$([\d.]+)\$ on all (\d+) records", R["lora_drift_max"],
        R["lora_records"])
    chk("LoRA arm: the CKA range that replaces the one-cell reading",
        r"CKA\$_\\text\{E\}\$ spans \$([\d.]+)\$ to \$([\d.]+)\$ across these (\w+) cells",
        R["lora_cka_lo"], R["lora_cka_hi"], R["lora_cells"])
    chk("LoRA arm: full fine-tuning ahead of LoRA",
        r"\(B\$-\$E\$\{\}<0\$\) on \\textbf\{(\d+) of the (\d+)\}",
        R["lora_be_neg"], R["lora_cells"])

    # --- S3's training details. Five numerals, all of which were wrong, and all of which are now read
    # out of scripts/finetune_forecasting.py by AST (see rederive()) rather than out of the last draft.
    # The two exponents are captured as exponents: the prose prints 10^{-4}, and capturing "4" from it
    # is the only form in which the pattern and the value can be compared at all.
    chk("the optimiser S3 states", r"AdamW \(\$\\eta\{=\}10\^\{-(\d)\}\$, weight decay "
        r"\$10\^\{-(\d)\}\$\), batch size (\d+), gradient clipping\s*at ([\d.]+)",
        R["lr_exp"], R["wd_exp"], R["batch_size"], R["clip_norm"])
    # The epoch budget and the absence of early stopping in ONE pattern, because the relation between
    # them is the claim: 20 epochs that may be cut short is a different protocol from 20 epochs that
    # are all run, and the paper's forgetting numbers are the final epoch's.
    chk("the epoch budget, and that nothing stops it early",
        r"\\textbf\{(\d+) epochs with no early stopping\}", R["epochs_dflt"])
    chk("the matrix's modal training size", r"\\textbf\{(\d+) of the (\d+)\} cells train on",
        R["n_train_modal_cells"], R["n_int"])
    # The sweep grid as one pattern: five numbers whose ORDER is the claim, so five separate chks
    # would all pass with two of them transposed.
    chk("the sample-size grid S3 sweeps",
        r"\$n\{\\in\}\\\{(\d+),(\d+)\{\\text\{k\}\},(\d+)\{\\text\{k\}\},(\d+)\{\\text\{k\}\},"
        r"(\d+)\{\\text\{k\}\}\\\}\$", *R["n_grid_printed"])

    # --- S3's definition of the primary baseline. The ridge's shape and penalty and the window cap
    # live in a function signature, so nothing downstream would contradict a wrong number here.
    chk("the primary baseline's shape and penalty",
        r"ridge map \$\\R\^\{(\d+) \\times F\} \\to \\R\^\{h \\times F\}\$ "
        r"\(\$\\lambda\{=\}10\^\{-(\d)\}\$\)", R["ridge_lookback"], R["ridge_lam_exp"])
    chk("the gate's window cap", r"\(up to (\d+)\s*\n?\s*windows\)", R["gate_max_eval"])
    # The asymmetry, as one pattern: "96+h against 96" is a claim about the RELATION between the two
    # context lengths, and the two numbers are the same constant, so separate chks would both pass
    # with the comparison inverted.
    chk("the context asymmetry between the model and its baseline",
        r"\$(\d+)\{\+\}h\$ steps of context against the baseline's \$(\d+)\$",
        R["ridge_lookback"], R["ridge_lookback"])
    chk("the number of forecast samples the median is taken over",
        r"median of (\d+) forecast samples", R["fc_samples"])
    # The reproducibility statement's "median of 20 sampled forecasts" is the same constant in a
    # different phrasing, and it already has a chk ("two-cell: Moirai's sample count") -- which is why
    # no second one is registered here. The two now read one value; see rederive().
    chk("the horizons the body reports at", r"at \$h\{=\}(\d+)\$ and \$h\{=\}(\d+)\$", *R["moirai_h"])
    # The same two horizons in set form, which is how S6 and the limitations contrast the Moirai arm
    # against the h=24 backbones. One pattern, two groups, and it reaches every site of that phrasing.
    chk("the horizons, in the form the cross-backbone caveat uses",
        r"\$h\{\\in\}\\\{(\d+),(\d+)\\\}\$", *R["moirai_h"])
    # The baseline's lookback where the body names it in words rather than in the ridge's shape: the
    # gate is called "the lookback-96 linear regression" in S1 and S2, two sites this reaches and the
    # shape pattern in S3 does not.
    chk("the baseline named by its lookback", r"lookback-(\d+) linear regression",
        R["ridge_lookback"])
    # And the asymmetry again, in S4's robustness form ("giving the baseline Moirai's own 96+h context
    # instead of 96"), where the two numbers straddle a line break.
    chk("the context asymmetry, where S4 tests removing it",
        r"Moirai's own \$(\d+)\{\+\}h\$ context instead of\s+\$(\d+)\$",
        R["ridge_lookback"], R["ridge_lookback"])
    chk("the interval level, where S5 reports the paired intervals",
        r"Paired \$(\d+)\\%\$ intervals", R["ci_pct"])
    # The sweep grid's endpoints in the two abbreviated forms the experiment sections use. The full
    # five-value grid is registered at its S3 site; these name only the ends, and the low end is
    # printed with a thousands separator, so the separator is in the pattern.
    chk("the sweep's endpoints, abbreviated",
        r"\$n\{\\in\}\\\{(\d+),\\ldots,(\d+)\\mathrm\{k\}\\\}\$",
        R["n_grid_printed"][0], R["n_grid_printed"][-1])
    # Anchored on the sentence, not on the "$n{=}X$k" form: that form labels five different sample
    # sizes across the paper, so a bare pattern binds every one of them to the top of the grid and
    # fails at n=1k and n=5k. A pattern that matches a LABEL is not a claim about a value.
    chk("the top of the sweep, where S5 calls its sign fragile",
        r"\\textbf\{The \$n\{=\}(\d+)\$k sign is fragile\}", R["n_grid_printed"][-1])
    chk("the bottom of the sweep, with its separator",
        r"fine-tuning on (\d+)--(\d+)\{,\}000 samples",
        R["n_grid_printed"][0], R["n_grid_printed"][1])
    # The positive control's dose ladder, as REGISTERED. The word "registered" is load-bearing and is
    # inside the pattern: the arm also ran the declared extension to 1e-1, so a body sentence naming
    # three rungs without that word implies its grid is the whole list of runs, and a reader
    # multiplying it out gets 18 against the appendix's 21. One word rather than a clause because the
    # body has no line to spare -- spelling the extension out here cost two typeset lines and pushed
    # the Ethics Statement off page 10, which is the submission's hard page gate.
    chk("the positive control's registered dose ladder",
        r"the registered\s+\$\\text\{lr\}\\in\\\{10\^\{-(\d)\},10\^\{-(\d)\},10\^\{-(\d)\}\\\}\$ "
        r"crossed with both conditions at (\d) seeds", *R["pc_lr_exps"], R["pc_seeds"])
    # The extension itself, at the appendix site that explains it, where there is no page limit. Both
    # numbers in one pattern because the claim is conditional -- the rung the rule permitted AND the
    # rung whose outcome triggered it -- and either alone would pass with the condition inverted.
    chk("the declared extension, and the rung that triggered it",
        r"one extension to \$\\text\{lr\}\{=\}10\^\{-(\d)\}\$ if destruction was not\s+"
        r"achieved at \$10\^\{-(\d)\}\$", R["pc_ext_exp"], R["pc_lr_exps"][-1])

    # --- the two design constants S3 states and the code acts on everywhere downstream.
    chk("the gate's operating point, where S3 defines it",
        r"thresholding it at \$([\d.]+)\$---a prespecified", R["gate_threshold"])
    # The same constant again, in clause (i) of the definition twenty lines below, which is where a
    # reader checks what the criterion actually tests. Two phrasings of one number, so two chks:
    # perturbing the definitional site left this one silently unflagged.
    chk("the gate's operating point, in clause (i) of the definition",
        r"\\Vb\{b\} \\ge\s*([\d.]+)\$ on selection windows", R["gate_threshold"])
    chk("the interval level", r"the \$(\d+)\\%\$ CI on", R["ci_pct"])
    # The compute ceiling. The thousands separator is in the pattern rather than captured: "10{,}000"
    # cannot be tokenised as one number, and a max that stopped being a round number of thousands
    # would fail this chk's min_sites rather than pass it at the wrong value.
    chk("the compute ceiling (reproducibility statement)",
        r"no run\s*exceeds (\d+)\{,\}000 training windows", R["n_train_max_k"])
    # The two-cell from-scratch check's two invocations, each against the script that runs it.
    chk("the two-cell check's Moirai cell",
        r"\(Small/ETTh2 \$h\{=\}(\d+)\$, \$n\{=\}(\d+)\$, seed (\d+), full fine-tuning~\(B\)\)",
        R["r2_moirai_h"], R["r2_moirai_n"], R["r2_seed"])
    chk("the two-cell check's TimesFM cell",
        r"\(TimesFM/ETTh1 \$h\{=\}(\d+)\$, \$n\{=\}1\{,\}000\$, seed (\d+),",
        R["r2_tf_h"], R["r2_seed"])

    # --- condition D's own CKA. S3's concession and tab:cka_calibration's row stated this range by
    # hand and disagreed with each other for ten rounds: S3 said 0.969, which is one root's minimum
    # (v35_base_frozen) quoted as the range, against the records' 0.889. Both sites now read the same
    # derivation. The upper endpoint is registered TRUNCATED -- see rederive(), where rounding it to
    # four places would print 1.0000 inside a sentence that says "never exactly 1".
    chk("condition D's measured CKA on Moirai, and the TimesFM floor beside it",
        r"never exactly \$1\$ on Moirai \(\$([\d.]+)\$--\$([\d.]+)\$\) and falls as low as\s*"
        r"\$([\d.]+)\$ on TimesFM",
        R["cond_d_cka_lo"], R["cond_d_cka_hi"], R["cond_d_cka_timesfm_lo"])
    chk("condition D's CKA range and run count (calibration table)",
        r"cond\.\\ D, Moirai \((\d+) runs\) & ([\d.]+)--([\d.]+) &",
        R["n_cond_d_runs"], R["cond_d_cka_lo"], R["cond_d_cka_hi"])
    # The row's scope, stated in the paragraph under it: the run count again (a count printed twice is
    # a count that drifts once) and how far the excluded partial-unfreeze arm reaches, which is the
    # whole justification for excluding it.
    chk("condition D's scope: what the row excludes",
        r"its (\d+) runs exclude the\s*partial-unfreeze arm .*?reach down to CKA \$([\d.]+)\$",
        R["n_cond_d_runs"], R["lu_cond_d_cka_lo"])
    chk("condition D on TimesFM (calibration scope paragraph)",
        r"TimesFM's D falls as\s*low as \$([\d.]+)\$", R["cond_d_cka_timesfm_lo"])
    chk("LoRA arm: the two cells where LoRA is worse than not fine-tuning",
        r"forg\$_\\text\{E\}=\{\+\}([\d.]+)\\%\$, (\d+) of (\d+) seeds, on "
        r"Moirai-Base/ETTh2 \$h\{=\}96\$; \$\{\+\}([\d.]+)\\%\$, (\d+) of (\d+), on "
        r"Moirai-Base/ETTm2",
        R["lora_bh2_96_forg"], R["lora_bh2_96_forg_pos"], R["lora_bh2_96_seeds"],
        R["lora_bm2_192_forg"], R["lora_bm2_192_forg_pos"], R["lora_bm2_192_seeds"])
    # The justification for pairing, not the result of it: if the gap ever widens past the SEM this
    # sentence stops being a reason and the paragraph needs rewriting, not renumbering.
    chk("LoRA arm: why D-E is paired",
        r"means sit \$([\d.]+)\$~pp apart with a SEM of \$([\d.]+)\$",
        R["lora_sh1_gap"], R["lora_sh1_96_be_sem"])
    chk("LoRA arm: the paired D-E verdict",
        r"LoRA beats freezing on (\d+) of (\d+) cells, freezing beats LoRA on (\d+), and "
        r"(\d+) are inseparable",
        R["lora_e_beats_d"], R["lora_cells"], R["lora_d_beats_e"], R["lora_inseparable"])
    chk("LoRA arm: CKA does not order the outcome",
        r"\$\\rho\{=\}\{-\}([\d.]+)\$ against B\$-\$E; \$\{-\}([\d.]+)\$ against B\$-\$D",
        -R["lora_rho_be"], -R["lora_rho_bd"])
    chk("LoRA arm: the two extreme cells",
        r"Moirai-Large/ETTh2 at \$([\d.]+)\$, is one where the constrained option "
        r"\\emph\{loses\} to full fine-tuning by \$([\d.]+)\$~pp, and the lowest, "
        r"Moirai-Base/ETTm2 \$h\{=\}96\$ at \$([\d.]+)\$",
        R["lora_lh2_96_cka"], -R["lora_lh2_96_be"], R["lora_bm2_96_cka"])
    chk("condition coverage per cell",
        r"Condition~H \(strict freeze\) adds (\d+) cells "
        r"\(Appendix~\\ref\{app:strictfreeze\}\) and condition~E \(LoRA\) adds (\d+)",
        R["sf_cell_count"], R["lora_cells"])

    # --- correlations: within-Moirai CKA
    # rho must be labelled within-backbone wherever it appears: pooled across backbones the same
    # quantity reads differently, and that is a distinction this paper's own rule turns on.
    # "within Moirai, the only backbone with enough cells to ask" is the same labelling requirement as
    # "within-backbone" and is how the abstract, S6 and app:valueaxis all phrase it; only contribution 2
    # uses the hyphenated form. Accepting both is not a loosening -- the point of the check is that the
    # scope is NAMED next to the number, and a bare "$\rho{=}{+}0.168$" still matches nothing.
    # The lookahead excludes exactly one thing, and it is the same hazard the gate's RETRO list below
    # handles: app:clustered prints the L2 weight drift's OWN within-Moirai rho as "within Moirai
    # ($\rho{=}{+}0.318$)", three words from the CKA one, in the same notation. Every CKA site names
    # its scope with a clause first ("within Moirai, the only backbone...", "within Moirai at either
    # level"), so requiring anything other than an immediate "($" separates them without loosening
    # what the check is for -- a bare "$\rho{=}{+}0.168$" with no named scope still matches nothing.
    # 5, not 4, since 21 Sep 2026: the claims table (app:claims) states C1's evidence in the body's
    # own words on purpose, so this pattern now pins that row too.
    chk("within-Moirai CKA rho",
        r"within(?:-backbone|\s+Moirai(?! \(\$))[^$]{0,70}\$\\rho\{=\}\{\+\}([\d.]+)",
        R["cka_rho"], min_sites=5)
    # A "label after symbol" variant ("$\rho{=}{+}0.168$ within backbone") existed only in the
    # conclusion, whose copy of this statistic was removed on 20 Sep 2026; the conclusion now points
    # at S6. The check is retired rather than loosened -- the value is still pinned at six prose sites
    # by the three checks around this comment, so retiring it costs no coverage.
    chk("within-Moirai CKA rho (body and the claims table)",
        r"cells to ask, \$?\\rho\{=\}\{\+\}([\d.]+)", R["cka_rho"], min_sites=2)
    # The preceding {+}rho is required and non-capturing: since the flip, the gate's own clustered
    # CI is printed in the same shape a few words away in Figure 1's caption, and without this the
    # gate's interval would be checked against CKA's numbers.
    # Four word orders join the rho to its CI, so the gap is a bounded run that may not cross a `$`.
    # The [^$] is what makes the preceding {+} load-bearing: it cannot be borrowed from an earlier
    # math group, so the gate's own negative-rho interval a few words away does not match here.
    chk("within-Moirai CKA clustered CI",
        r"\{\+\}[\d.]+\$[^$]{0,25}?clustered CI \$\[-([\d.]+),\{\+\}([\d.]+)\]\$",
        # 4, not 5: the conclusion's copy of this CI was removed on 20 Sep 2026 to pay for the
        # selection-split prose, and Figure 1's caption stopped re-typing it on 22 Sep 2026 (C1/C2)
        # because panel (b) prints both clustered rhos in its own text block. Four remain -- the intro's
        # contribution 2, S6, the abstract and app:claims. Lowering the floor twice now to match
        # deliberate deletions; each is recorded at the site it left.
        R["cka_lo"], R["cka_hi"], min_sites=4)
    # "within-Moirai CKA rho (figure caption)" and "gate rho, primary, clustered CI in the figure
    # caption" are both retired on 22 Sep 2026 (C1/C2). The caption was 11 printed lines, most of them
    # re-typing numbers the panels themselves render: fig1_diagnostic_flow.py computes both clustered
    # rhos from cell_matrix and prints them ON panel (b), so the caption's copies were hand-typed
    # duplicates of build-time output -- the exact drift failure this checker exists for. They are
    # deleted, not re-verified. Both rhos keep prose coverage elsewhere: "within-Moirai CKA rho" and
    # "gate rho within Moirai" (min_sites=3) still run, and the figure's own asserts cover the panel.
    chk("Moirai cell count", r"\$n\{=\}(\d+)\$ cells from six series", R["n_moirai"])
    # The intro's second copy of this denominator went with the restructure into five paragraphs; it
    # was attached to the retrospective rho, which S7 states with the same denominator two lines from
    # the site the check above pins.

    # --- correlations: the gate against the intervention
    # Deliberately unanchored: the DEFAULT reading of a negative rho in this paper is the primary
    # selection-split gate, and a new site that quotes a different value should fail here rather
    # than pass unnoticed. Four contexts legitimately print a different negative rho, and each is
    # excluded by the words that FOLLOW it (lookbehind would have to be fixed-width) and then
    # re-checked on its own below. So a swapped value -- the retrospective number dropped into a
    # primary sentence -- fails the unanchored check, and a wrong value in an excluded context
    # fails that context's own check. Neither can pass by being in the other's list.
    # The trailing \$ is load-bearing: without it [\d.]+ backtracks a digit at a time until the
    # lookahead is satisfied, so the exclusion silently becomes "match a prefix of any number".
    # The gate rho is quoted at six sites and three OTHER negative rhos share its notation, so each
    # exclusion here names the phrase that follows one of them. ", CI $[-" is the graded value axis
    # (-0.270 over 31 cells, -0.229 over the 20 value-cells); the gate sites all write "clustered CI"
    # or continue with prose, so the lookahead cannot swallow one of them.
    RETRO = (r"(?! against B\$-\$E)(?! on the same cells)(?! with a cell-level CI)"
             r"(?! between the gate)(?! over the four shared datasets)(?!, CI \$\[-)")
    chk("gate rho within Moirai", r"\\rho\{=\}\{-\}([\d.]+)\$" + RETRO,
        R["gate_rho"], min_sites=3)
    # the retrospective variant, quoted in Exp 4 and twice in the audit appendix
    chk("gate rho, retrospective variant (body)",
        r"instead gives \$\\rho\{=\}\{-\}([\d.]+)\$ on the same cells", R["gate_test_rho"])
    chk("gate rho, retrospective variant (appendix)",
        r"\$\\rho\{=\}\{-\}([\d.]+)\$ with a cell-level CI of \$\[-([\d.]+), -([\d.]+)\]\$",
        R["gate_test_rho"], R["gate_test_celllo"], R["gate_test_cellhi"])
    chk("gate rho, retrospective variant (audit bullet)",
        r"\$\\rho\{=\}\{-\}([\d.]+)\$ between the gate and B\$-\$D moves from "
        r"\$\[-([\d.]+), -([\d.]+)\]\$ over cells to \$\[-([\d.]+), \+([\d.]+)\]\$",
        R["gate_test_rho"], R["gate_test_celllo"], R["gate_test_cellhi"],
        R["gate_test_lo"], R["gate_test_hi"])
    # the primary gate's own two intervals, which the appendix prints side by side
    chk("gate rho, primary, both resampling levels",
        r"primary\. Neither resampling level supports an interval claim: cells give "
        r"\$\[-([\d.]+), \+([\d.]+)\]\$ and the six clusters give \$\[-([\d.]+), \+([\d.]+)\]\$",
        R["gate_celllo"], R["gate_cellhi"], R["gate_lo"], R["gate_hi"])
    # Retired 22 Sep 2026 (C1/C2) with the CKA caption chk above: the caption no longer prints either
    # rho or either interval, because panel (b) computes and draws both. The gate's clustered interval
    # keeps its prose site in app:clustered, pinned by "gate rho, primary, both resampling levels"
    # directly above, and the rho itself by "gate rho within Moirai" at 3 sites.
    # the withdrawn cross-backbone ordering claim, both splits
    chk("cross-backbone dataset ordering, both splits",
        r"identically on the held-out windows \(\$\\rho\{=\}\{\+\}([\d.]+)\$\) and disagree on the "
        r"selection windows \(\$\\rho\{=\}\{-\}([\d.]+)\$ over the four shared datasets\)",
        R["xback_test_rho"], R["xback_val_rho"])

    # --- correlations: pooled and graded
    chk("pooled CKA rho and CI",
        r"\\rho\{=\}\{\+\}([\d.]+)\$, clustered CI \$\[\{\+\}([\d.]+),\{\+\}([\d.]+)\]",
        R["pooled_rho"], R["pooled_lo"], R["pooled_hi"])
    # S6's graded-screen paragraph became a one-sentence pointer on 20 Sep 2026 and its four numbers
    # went back to app:valueaxis, which had been stating them all along -- the body was restating the
    # appendix. So the three body checks here are re-pointed at the appendix's own wording rather than
    # retired: vc_* keeps the check immediately below, and lv_*, b3 and val_* are re-pointed just after
    # it. On the selection-split ladder the value-cell interval no longer straddles zero, so the sign
    # of the lower bound is part of the claim and is written into each pattern: if the union moved and
    # a bound went negative, the check would stop matching rather than quietly agree.
    chk("value-cell CKA rho and CI (appendix)",
        r"the (\d+) value-cells gives \$\\rho\{=\}\{\+\}([\d.]+)\$ with clustered "
        r"CI \$\[\{\+\}([\d.]+), \{\+\}([\d.]+)\]\$",
        R["vc_n"], R["vc_rho"], R["vc_lo"], R["vc_hi"])
    chk("low-value CKA rho and CI",
        r"the (\d+) cells with no admissible value give \$\\rho\{=\}\{\+\}([\d.]+)\$ with CI "
        r"\$\[-([\d.]+), \{\+\}([\d.]+)\]",
        R["lv_n"], R["lv_rho"], R["lv_lo"], R["lv_hi"])
    # vc_n's body site went with the same paragraph and keeps three appendix sites: the two rho
    # sentences registered here and "value-cell count (aggregate)" in the paired-inference block.
    chk("cells with nothing to lose", r"because (\d+) of the (\d+) cells beat no rung",
        R["lv_n"], R["n_int"])
    chk("interaction coefficient",
        r"\$b_3\{=\}\{-\}([\d.]+)\$~pp per unit CKA per unit value with "
        r"CI \$\[-([\d.]+), \{\+\}([\d.]+)\]",
        R["b3"], R["b3_lo"], R["b3_hi"])
    # On the selection-split axis this rho is NEGATIVE, so the sign is in the pattern and the
    # registered value is its magnitude: a sign flip in the record breaks the match instead of
    # passing on the magnitude alone. The body clause that stated it is now a pointer (S7 says only
    # that no other quantity orders \denc either), so the appendix sentence is the site, and it
    # carries both the 31-cell and the 20-value-cell readings rather than one of them.
    chk("value score rho and CI (appendix)",
        r"gives \$\\rho\{=\}\{-\}([\d.]+)\$, CI \$\[-([\d.]+), \{\+\}([\d.]+)\]\$ across all (\d+) "
        r"cells and \$\\rho\{=\}\{-\}([\d.]+)\$, CI \$\[-([\d.]+), \{\+\}([\d.]+)\]\$ across "
        r"the (\d+) value-cells",
        abs(R["val_rho"]), R["val_lo"], R["val_hi"], R["n_int"],
        abs(R["valvc_rho"]), R["valvc_lo"], R["valvc_hi"], R["vc_n"])

    # --- paired inference
    # Sign in the pattern, magnitude in the record: on the selection-split value-cells the aggregate
    # is POSITIVE (freezing marginally ahead on average) where the test-side one was negative, and
    # the direction is the whole content of the sentence.
    chk("value-cell aggregate Delta_enc",
        r"\\denc\{=\}\{\+\}([\d.]+)\$ with CI \$\[-([\d.]+),\{\+\}([\d.]+)\]\$ and (\d+) of (\d+)",
        R["agg_mean"], R["agg_lo"], R["agg_hi"], R["n_vc_pos"], R["n_vc"])
    chk("value-cell count (aggregate)", r"over the (\d+) value-cells", R["n_vc"])

    # --- the ladder, selection split (the body's primary)
    # S4's eight-rung enumeration moved to app:baselines:val this round: a reviewer asked for the
    # ladder to read as a robustness check rather than as a second headline, and the appendix had been
    # printing the same eight counts all along. So the per-rung counts keep exactly one site, the one
    # below, and the body now states only the two facts that are claims -- that no cell clears every
    # admissible rung, and the union's size and composition, both still registered.
    chk("ladder pass counts (appendix, selection split)",
        r"the eight rungs give \\textbf\{"
        + r", ".join([r"(\d+)"] * (len(R["ladder_counts"]) - 1))
        + r" and (\d+) of (\d+)\} on this split",
        *R["ladder_counts"], R["n_val_scored"])
    # Two sites, S4 and the ladder appendix, and both must carry the backbone composition: the union
    # is what answers "gate-positive evidence is concentrated in Moirai/ETTh2", and a bare count of
    # 20 does not answer it. The pattern requires all four numbers, so dropping the composition from
    # either site fails here rather than silently weakening the claim.
    chk("ladder union and its backbones",
        r"\\textbf\{(\d+)(?: of the \d+)? clear it against at least one"
        r"(?: admissible rung)?[-:]+ ?(\d+) Moirai, (\d+) Chronos, (\d+) TimesFM\}",
        R["n_union"], R["n_union_moirai"], R["n_union_chronos"], R["n_union_timesfm"],
        min_sites=2)
    # --- the prospective arm under the corrected gates
    chk("prospective cells clearing the corrected gate",
        r"only (\w+) of the (\w+) clears \$0\.20\$ on either corrected split "
        r"\(base/ETTm2 \$h\{=\}192\$, \$\+([\d.]+)\$ on selection windows\) and it fails "
        r"clause~\(ii\) in (\d+) of (\d+) seeds",
        R["n_pro_clearing_corrected"], R["n_pro"], R["pro_clearing_gate"],
        R["pro_clearing_forg_pos"], R["pro_clearing_seeds"])
    # The confusion counts, in one pattern with the precision beside them: the failure being guarded is
    # one of the four moving alone, which is exactly what "TP 0, FP 8" invites when the cell count
    # changes and nobody re-derives the precision.
    chk("prospective confusion counts and precision",
        r"\\textbf\{TP~(\d+), FP~(\d+), FN~(\d+)\}: precision \$([\d.]+)\$",
        R["pro_tp"], R["pro_fp"], R["pro_fn"], R["pro_prec"])
    # Three further phrasings of the same two facts, in S1's lead and in the corrections appendix.
    # Registered separately because coverage here is per PHRASING, not per fact: the intro's "TP~0,
    # FP~8" and the appendix's "TP~0, FP~8, precision~0.00" are different strings, and the pattern
    # above reaches neither. The appendix's parenthetical "(precision 0.25, TP 2, FP 6, recall 1.00)"
    # is the SUPERSEDED reading and is deliberately not matched by any of these -- it is a historical
    # statement about a number that was wrong, and registering it would force it to track the
    # correction it exists to record.
    chk("prospective confusion counts (S1 lead)",
        r"TP~(\d+), FP~(\d+) on eight cells", R["pro_tp"], R["pro_fp"])
    chk("prospective confusion counts and precision (corrections appendix)",
        r"\\textbf\{TP~(\d+), FP~(\d+), precision~([\d.]+)\}",
        R["pro_tp"], R["pro_fp"], R["pro_prec"])
    # S7's own restatement, the one that carries the threshold-independence claim ("at all of them").
    # The "all of them" is an assert in rederive() over the whole sweep; this is the printed value.
    chk("prospective precision at every threshold (S7)",
        r"gives precision \$([\d.]+)\$\s*at all of them", R["pro_prec"])
    chk("prospective threshold sweep, flagged counts (slash form)",
        r"\$" + r"/".join([r"(\d+)"] * 6) + r"\$ cells flagged",
        *R["pro_flagged_by_threshold"])
    # The grid itself, so adding or removing a rung in the emitter cannot leave the appendix listing
    # the old set beside counts derived from the new one.
    chk("threshold sweep grid",
        r"every\} threshold in \$\\\{" + r", ".join([r"([\d.]+)"] * 6) + r"\\\}\$",
        *R["pro_threshold_grid"])
    chk("prospective threshold sweep, flagged counts",
        r"with " + r", ".join([r"(\d+)"] * 5) + r" and (\d+) cells flagged",
        *R["pro_flagged_by_threshold"])
    chk("rungs admitting a degradation cell",
        r"(\w+) of the ladder's eight rungs admit none, persistence admits (\w+) and GBM (\w+)",
        R["n_rungs_no_degradation"], R["n_degradation_persistence"], R["n_degradation_gbm"])
    chk("degradation cells a weak rung manufactures",
        r"All (\w+) rest on a denominator within", R["n_degradation_manufactured"])
    chk("union count (limitations)",
        r"the (\d+)-cell union spanning all three backbones", R["n_union"])
    chk("no cell clears every admissible rung (selection split)",
        r"\\textbf\{no cell of the (\d+) clears \$0\.20\$\}", R["n_val_scored"])
    # Same move as the per-rung counts: the survivors' graded range left S4 with the enumeration and is
    # stated once, in app:baselines:val, where the rung that sets each end is named beside it.
    chk("survivor range against the strongest rung (appendix)",
        r"\$\\Vbest\$ between \$-([\d.]+)\$ and \$\+([\d.]+)\$",
        abs(R["surv_strongest_lo"]), R["surv_strongest_hi"])
    # --- the ladder, held-out split (the retrospective variant in the appendix)
    chk("ladder pass counts (retrospective)",
        r"the eight rungs give \\textbf\{"
        + r", ".join([r"(\d+)"] * (len(R["ladder_counts_test"]) - 1))
        + r" and (\d+) of (\d+)\} in ladder order",
        *R["ladder_counts_test"], R["n_screened"])
    chk("union count (retrospective)",
        r"no cell of the (\d+) clears \$0\.20\$ against every admissible rung\}, and (\d+) "
        r"clear it against at least one",
        R["n_screened"], R["n_union_test"])

    # --- the matched-lookback column. Both splits are registered, and the check on each names its
    # own split in the pattern, because the effect reverses: 7 -> 6 on the selection windows and
    # 5 -> 7 on the held-out ones. A pattern that spanned both would agree with either direction.
    chk("matched-lookback self-check denominator",
        r"on all (\d+) Moirai cells of each split", R["mlb_n_val"])
    chk("matched-lookback stronger count (body)",
        r"stronger denominator on \\textbf\{(\d+) of (\d+)\} cells",
        R["mlb_stronger_val"], R["mlb_n_val"])
    chk("matched-lookback survivor count (body)",
        r"takes that (\w+) to \\textbf\{(\w+)\}",
        R["mlb_pass_nominal_val"], R["mlb_pass_matched_val"])
    chk("matched-lookback stronger count (appendix, selection)",
        r"stronger of the two on \\textbf\{(\d+) of (\d+)\}",
        R["mlb_stronger_val"], R["mlb_n_val"])
    chk("matched-lookback survivor count (appendix, selection)",
        r"clearing \$0\.20\$ falls from \\textbf\{(\d+) to (\d+)\}",
        R["mlb_pass_nominal_val"], R["mlb_pass_matched_val"])
    chk("matched-lookback flipped cell's two values",
        r"one cell that flips, from \$\+([\d.]+)\$ to \$\+([\d.]+)\$",
        R["mlb_flip_before"], R["mlb_flip_after"])
    chk("matched-lookback stronger count (appendix, held-out)",
        r"matched map is stronger on only \\textbf\{(\d+) of (\d+)\}",
        R["mlb_stronger_test"], R["mlb_n_test"])
    chk("matched-lookback survivor count (appendix, held-out)",
        r"the count rises from \\textbf\{(\d+) to (\d+)\}",
        R["mlb_pass_nominal_test"], R["mlb_pass_matched_test"])
    # The disclosure. Registered so that "the separation protects no count" cannot survive the day the
    # numbers stop supporting it -- which is the whole reason the column is reported on its own axis.
    chk("matched-lookback as a ladder rung (both splits)",
        r"every admissible rung at \\textbf\{(\d+) of (\d+) on both splits\}",
        R["mlb_rung_clearing_after_val"], R["mlb_n_val"])
    chk("matched-lookback strongest-rival counts",
        r"strongest admissible rival on \\textbf\{(\d+)\} cells on the selection split and "
        r"\\textbf\{(\d+)\} on the held-out split",
        R["mlb_rung_strongest_val"], R["mlb_rung_strongest_test"])

    # --- the unfitted-baseline correction
    # The abstract stopped saying "correcting the estimator MOVES n of m" when its provenance clause was
    # compressed to one sentence; it now states the same count from the failure side, "puts n of m on
    # the wrong side of the screen". Same two numbers, so the pattern follows the wording rather than
    # the wording being kept alive to suit the pattern.
    chk("Moirai gate flips (abstract)",
        r"puts (\d+) of (\d+) Moirai cells on the wrong side",
        R["n_flip_moirai_p2f"], R["n_moirai_screened"])
    # Both body sites must name the Moirai denominator. 17 is the Moirai flip count; across all
    # arms it is 26 of 32, so an unqualified "17 cells" in a 32-cell context reads as 17 of 32 and
    # understates the correction. The pattern requires the qualifier, so dropping it fails the check.
    chk("gate flips across the threshold",
        r"moves (\d+) of Moirai's (\d+) cells across the threshold",
        R["n_flip_moirai"], R["n_moirai_screened"])
    # A third phrasing, in the retraction statement, where the count is the retraction's substance: it
    # is the number that makes "the entire class of cell the diagnostic was built to predict" true.
    chk("gate flips (retraction statement)",
        r"changes the gate status\s*of (\d+) of the (\d+) Moirai cells",
        R["n_flip_moirai"], R["n_moirai_screened"])
    # 16 and 17 are both correct and sit twenty lines apart, which reads as one fact unless each
    # site says which direction it counts. The abstract's 16 is pass-to-fail; the two 17s are
    # either-direction and now carry the pass-to-fail subcount, so a reader cannot conflate them.
    # One site, not two, since contribution 3 was deleted: S3's box is the remaining either-direction
    # site, and the abstract states the same subcount as "puts 16 of 21 on the wrong side", which the
    # check above matches. Both directions therefore still have a site apiece.
    chk("gate flips, the pass-to-fail subcount", r", (\d+) of them to gate-fail",
        R["n_flip_moirai_p2f"], min_sites=1)
    chk("gate flips (appendix)", r"(\w+) cells change status: (\d+) flip",
        R["n_flip_moirai"], R["n_flip_moirai_p2f"])
    # Per-arm, because the three arms' before/after counts sit in three adjacent list items and a
    # single loose pattern matches all three with one arm's expected values. This is what caught
    # TimesFM's "4 of 5 before", which had been 5 of 5 since the correction was computed.
    for arm, label in (("moirai", "Moirai"), ("chronos", "Chronos"), ("timesfm", "TimesFM")):
        chk(f"{label} gate passes before/after the correction",
            r"\\textbf\{" + label + r":\} \\textbf\{(\d+) of (\d+)\}.{0,120}?"
            r"against (\d+) of (\d+) before",
            R[f"n_pass_{arm}"], R[f"n_screened_{arm}"],
            R[f"n_pass_trend_{arm}"], R[f"n_screened_{arm}"])
    chk("gate passes across arms, both splits",
        r"\\textbf\{(\d+) of (\d+) cells pass on the selection split---the\s+"
        r"paper's primary---and (\d+) of (\d+) on the retrospective test-side variant\}",
        R["n_gate_pass"], R["n_val_scored"], R["n_gate_pass_test"], R["n_screened"])

    # --- the closest candidate cell, quoted number for number
    chk("h192 survivor forgetting",
        r"forg\.\$_\\text\{B\}\{=\}\{\+\}([\d.]+)\{\\pm\}([\d.]+)\$ with (\d+) of (\d+) seeds",
        # One site, not two: S5.3 used to restate this cell number for number and now
        # points back at S5.1 instead (a deliberate deletion for the page limit, not a
        # stale pattern).  The seed split is still pinned at three other sites below.
        R["s192_forg_b"], R["s192_sem"], R["s192_pos"], R["s192_seeds"])
    chk("h96 survivor forgetting",
        r"\(\$\+([\d.]+)\{\\pm\}([\d.]+)\$, (\d+) of (\d+) seeds\)",
        R["s96_forg_b"], R["s96_sem"], R["s96_pos"], R["s96_seeds"])
    chk("seed splits in the introduction",
        r"only (\d+)/(\d+) and (\d+)/(\d+) seeds agreeing",
        R["s96_pos"], R["s96_seeds"], R["s192_pos"], R["s192_seeds"])
    # The abstract's seed-agreement clause was cut with the short-abstract rewrite. Both splits keep
    # three registered sites between them -- the intro check above and the two survivor-forgetting
    # checks -- so no seed count lost its last site.

    # --- the three-way call (round 7's headline). FOUR phrasings, registered separately and on
    # purpose: this paper's recurring failure has been a reworded restatement drifting away from a
    # corrected count while one canonical site kept the check green. Coverage is per phrasing.
    chk("the three-way call (abstract)",
        r"the (\d+) cells split \\textbf\{(\d+) freezing-better, (\d+) adaptation-better, "
        r"(\d+) inconclusive\}",
        R["n_paired_tests"], R["n_freeze_dec"], R["n_adapt_dec"], R["n_inconclusive"])
    chk("the three-way call (contributions)",
        r"\((\d+) freezing-better, (\d+) adaptation-better, (\d+) inconclusive at a",
        R["n_freeze_dec"], R["n_adapt_dec"], R["n_inconclusive"])
    chk("the three-way call (S5.3)",
        r"call \\textbf\{(\d+) cells freezing-better, (\d+) adaptation-better and "
        r"(\d+) inconclusive\}",
        R["n_freeze_dec"], R["n_adapt_dec"], R["n_inconclusive"])
    chk("the three-way call (appendix)",
        r"\\textbf\{(\d+) cells where freezing is decisively better, (\d+) where adapting is "
        r"decisively better, and (\d+) inconclusive\}",
        R["n_freeze_dec"], R["n_adapt_dec"], R["n_inconclusive"])
    chk("the conclusion's freeze-decisive count", r"(\d+) of (\d+) favour freezing",
        R["n_freeze_dec"], R["n_paired_tests"])
    # The number of tests BH adjusts over, at both sites that name it. If a cell is added or dropped
    # this is the number that must move first, and every q in the table moves with it.
    chk("the BH test count", r"adjusted across the (\d+) cells", R["n_paired_tests"])
    chk("the BH test count (appendix)", r"(?:Thirty-one|\d+) cells are tested,\s*so we adjust",
        min_sites=1)

    # --- where the decisive cells sit relative to the screen. This is the round's sharpest claim
    # against our own instrument, so every number in it is pinned, including the two gate values --
    # which is what the over-broad ILI pattern used to swallow.
    chk("the two freeze-decisive cells",
        r"Moirai-Small/ETTh1 at \$h\{=\}96\$ \(\$\\denc\{=\}\{\+\}([\d.]+)\$, "
        r"\$q\{=\}([\d.]+)\$\) and \$h\{=\}192\$ \(\$\+([\d.]+)\$, "
        r"\$q\{=\}([\d.]+)\$\)",
        R["fd96_mean"], R["fd96_q"], R["fd192_mean"], R["fd192_q"])
    chk("the two freeze-decisive cells' gate values",
        r"\\Vb\{\\text\{ridge\}\}\$ of \$-([\d.]+)\$ and \$-([\d.]+)\$",
        R["fd96_gate"], R["fd192_gate"])
    chk("the two adapt-decisive cells that pass the screen",
        r"Moirai-Base/ETTh2 \$h\{=\}192\$ \(\$-([\d.]+)\$, \$q\{=\}([\d.]+)\$\) and\s*"
        r"Moirai-Large/ETTh2 \$h\{=\}96\$ \(\$-([\d.]+)\$, \$q\{=\}([\d.]+)\$\)",
        R["ad192_mean"], R["ad192_q"], R["adL96_mean"], R["adL96_q"])
    chk("the adapt-decisive passers' gate values",
        r"decisively wins \\emph\{pass\} it, at \$\+([\d.]+)\$ and\s*\$\+([\d.]+)\$",
        R["ad192_gate"], R["adL96_gate"])

    # --- power. The MDE range is over the gate-passing cells only, and the two unresolvable cells are
    # named at two sites (S5.3 and limitations item 3) because the concession is the point of reporting
    # them at all: an unresolvable cell reported as "no effect" is the error this table exists to stop.
    chk("the MDE range over the survivors",
        r"minimum detectable\s*effects of \$([\d.]+)\$--\$([\d.]+)\$~pp",
        R["mde_lo"], R["mde_hi"])
    chk("the two unresolvable cells' adjusted seed requirement",
        r"would need ([\d.]+) and\s*([\d.]+) paired seeds",
        R["n_star_s96"], R["n_star_b192m"], min_sites=2)

    # --- the gate values quoted cell by cell
    g, vg = R["gate_of"], R["vgate_of"]
    chk("the seven survivors' gate values (selection split)",
        r"Small \$h\{=\}96\$ \(\$\+([\d.]+)\$\) and \$h\{=\}192\$ \(\$\+([\d.]+)\$\), Base "
        r"\$h\{=\}96\$ \(\$\+([\d.]+)\$\) and \$h\{=\}192\$ \(\$\+([\d.]+)\$\), Large "
        r"\$h\{=\}96\$ \(\$\+([\d.]+)\$\).{0,120}?Small \$h\{=\}192\$ \(\$\+([\d.]+)\$\) and Base "
        r"\$h\{=\}192\$ \(\$\+([\d.]+)\$\)",
        vg["small_ETTh2_h96"], vg["small_ETTh2_h192"], vg["base_ETTh2_h96"],
        vg["base_ETTh2_h192"], vg["large_ETTh2_h96"],
        vg["small_ETTm2_h192"], vg["base_ETTm2_h192"])
    # S4's four per-dataset failing ranges are gone -- eight printed numbers restating Table r2task,
    # cut to make page room. What replaced them ("every failing cell is below the baseline rather than
    # short of the threshold") is a claim about all 15 failing Moirai cells at once, so it is enforced
    # by the max(_mfail) < 0 assert in rederive() rather than by a pattern here: no captured group can
    # carry "every", and a range check would pass while the word it qualifies went false.
    # The four vmo_* range values stay derived -- Appendix Table r2task still prints them.
    # The count reads as a word ("All nine cells"), which parse() cannot take, so the arm counts are
    # checked by the "0 of 5"/"0 of 4" headings just above and only the band is checked here.
    chk("non-Moirai gate band (selection split)",
        r"cells fall in a narrow band about zero, \$-([\d.]+)\$ to \$\+([\d.]+)\$",
        R["vnonmo_lo"], R["vnonmo_hi"])
    for arm, label in (("chronos", "Chronos"), ("timesfm", "TimesFM")):
        chk(f"{label} per-cell gate values",
            r"\\textbf\{" + label + r":\}.{0,60}?before: "
            r"ETTh1 \$-([\d.]+)\$, ETTh2 \$-([\d.]+)\$, Weather \$-([\d.]+)\$, "
            r"ETTm2 \$-([\d.]+)\$, Electricity \$\+([\d.]+)\$",
            *(abs(g[f"{arm}_{d}"]) for d in
              ("etth1", "etth2", "weather", "ettm2", "electricity")))
    chk("TimesFM/Electricity, the screened-only 32nd cell",
        r"no selection-split gate: it scores \$\+([\d.]+)\$ on the held-out windows",
        R["timesfm_elec_gate_test"])
    chk("ILI gate", r"\\Vb\{\\text\{ridge\}\}\{=\}\{-\}([\d.]+)", R["ili_gate"])
    chk("ILI gate (appendix list)",
        r"\\textbf\{ILI\} \(Moirai-Small\): \$\\mathbf\{-([\d.]+)\}\$", R["ili_gate"])
    chk("ILI gate (appendix, restated)", r"it now carries \$-([\d.]+)\$", R["ili_gate"])

    # --- app:zsnoise, the gate's sampling-noise floor. Registered at the same density as the gate
    # values themselves, because this section's whole function is to say how precisely those values
    # can be read -- a stale number here would understate or overstate the paper's own resolution.
    chk("sampling noise: the number of sampled paths",
        r"median of \$(\d+)\$ sampled paths", R["zsn_samples"])
    chk("sampling noise: the replicate census",
        r"Across the \$(\d+)\$ Moirai cell-scorings \(the \$(\d+)\$-cell retrospective grid and "
        r"batch 3's five\) there are \$(\d+)\$ independent zero-shot measurements",
        R["zsn_cells"], R["n_moirai_screened"], R["zsn_measurements"])
    chk("sampling noise: the largest per-cell spread",
        r"per-cell spread of at most \$([\d.]+)\$", R["zsn_max_spread"])
    chk("sampling noise: the three batches' denominator shifts",
        r"moves \$R\^2_\\text\{task\}\$ by at most \$([\d.]+)\$ on batch 1, \$([\d.]+)\$ on batch 2 "
        r"and \$([\d.]+)\$ on batch 3",
        R["zsn_delta_b1"], R["zsn_delta_b2"], R["zsn_delta_b3"])
    # Both numbers of the "different cells" argument in one pattern, for the reason the assert in
    # rederive() exists: the claim is a RELATION between them, so a check that could pass with one of
    # them re-pointed at the other cell would not be checking the claim.
    chk("sampling noise: the mispairing the section exists to reject",
        r"That largest spread, \$([\d.]+)\$, is bigger than the tightest margin to the operating "
        r"point anywhere \(\$([\d.]+)\$, Moirai-Base/ETTm2 \$h\{=\}192\$ at "
        r"\$R\^2_\\text\{task\}\{=\}\{\+\}([\d.]+)\$\)",
        R["zsn_max_spread"], R["zsn_tight_margin"], R["zsn_tight_r2"])
    chk("sampling noise: how far the widest cell sits from the operating point",
        r"sits \$([\d.]+)\$ from \$0\.20\$", R["zsn_wide_margin"])
    chk("sampling noise: the decisive ratio",
        r"minimum of that ratio over all \$(\d+)\$ cell-scorings is \$([\d.]+)\\times\$",
        R["zsn_cells"], R["zsn_min_ratio"])
    # The two sites that state the contaminated denominator would have changed nothing -- app:zsnoise
    # and audit item (g). Two phrasings, two patterns: this is the count a reader would most want to
    # be reassuring, so it is the one that must not be able to drift at one site only.
    chk("sampling noise: admissions the pre-repair denominator would have changed",
        r"\\textbf\{\$(\d+)\$ of the \$(\d+)\$\} admissions would have differed",
        R["zsn_n_changed"], R["zsn_cells"])
    chk("audit item (g): the shift and the admissions it would have changed",
        r"at most \$([\d.]+)\$ in \$R\^2_\\text\{task\}\$ and would have changed \$(\d+)\$ of the "
        r"\$(\d+)\$ admissions",
        R["zsn_delta_b1"], R["zsn_n_changed"], R["zsn_cells"])

    # --- the three survivors full fine-tuning improves
    chk("the three improvers' forgetting",
        r"forgetting is \$-([\d.]+)\\%\$, \$-([\d.]+)\\%\$ and \$-([\d.]+)\\%\$ against "
        r"zero-shot, negative in 3/3 seeds each, at CKA \$([\d.]+)\$, \$([\d.]+)\$ and "
        r"\$([\d.]+)\$",
        R["imp1_b"], R["imp2_b"], R["imp3_b"],
        R["imp1_cka"], R["imp2_cka"], R["imp3_cka"])
    chk("the three improvers under a frozen encoder",
        r"\(\$-([\d.]+)\\%\$, \$-([\d.]+)\\%\$, \$-([\d.]+)\\%\$\)",
        R["imp1_d"], R["imp2_d"], R["imp3_d"])

    # --- the LEAD DISSOCIATION EXAMPLE, at all four of its sites.
    # Added 25 Sep 2026, and the reason it is worth four patterns rather than one: these numbers were
    # already registered -- imp1/imp2 just above pin 0.460, 0.396, -31.4% and -36.3% at their S5.1
    # sites -- and every restatement of them elsewhere was nonetheless unchecked. That is "coverage is
    # per phrasing, not per fact" in its purest form, and it had already gone wrong: S6 printed
    # -36.4% where S5.1, the appendix and tables/dissociation.tex all print -36.3% (imp2_b = 36.345,
    # so -36.4 is a double-rounding of the appendix's 2-dp -36.35). One digit, in the sentence that
    # states the paper's strongest claim, surviving a clean checker run. The four sites do not share a
    # pattern because they do not share a phrasing: S1 splits the pair across two sentences with prose
    # between them, S6 gives all four numbers as two CKA-then-forgetting lists, and the appendix walks
    # the ordering from one end. A pattern loose enough to match all three would be loose enough to
    # match a swapped pair, which is the error most likely to be made here.
    chk("the lead example (S1): the degrading cell",
        r"CKA \$([\d.]+)\$ against the pre-trained encoder---and the model ends \$([\d.]+)\\%\$ "
        r"\\emph\{worse\}",
        R["deg1_cka"], R["deg1_b"])
    chk("the lead example (S1): the improving cell",
        r"restructures far more, to CKA \$([\d.]+)\$, and the model ends \$([\d.]+)\\%\$ "
        r"\\emph\{better\}",
        R["imp1_cka"], R["imp1_b"])
    chk("the lead example (S6): the two degraders",
        r"ETTh1 at CKA \$([\d.]+)\$ and \$([\d.]+)\$, \$\+([\d.]+)\\%\$ and \$\+([\d.]+)\\%\$",
        R["deg1_cka"], R["deg2_cka"], R["deg1_b"], R["deg2_b"])
    chk("the lead example (S6): the two improvers",
        r"ETTh2 at CKA \$([\d.]+)\$ and \$([\d.]+)\$, \$-([\d.]+)\\%\$ and \$-([\d.]+)\\%\$",
        R["imp1_cka"], R["imp2_cka"], R["imp1_b"], R["imp2_b"])
    # The appendix's version, which is the one that states the ordering as monotone. It runs from the
    # MOST drifted end, so its first number is imp2 (the h192 improver) and its last two are the
    # degraders in descending order -- the reverse of S6's direction, deliberately matched as written.
    chk("the lead example (appendix): the monotone ordering",
        r"runs monotonically from \$-([\d.]+)\\%\$ at the most drifted cell to \$\+([\d.]+)\\%\$ "
        r"and \$\+([\d.]+)\\%\$ at the two least drifted",
        R["imp2_b"], R["deg2_b"], R["deg1_b"])

    # --- S6's FOUR REMAINING PAIRS and S5.2's sweep restatement, registered 25 Sep 2026 from the
    # coverage audit. Each claim is a relation between two numbers, so each is one pattern with both
    # of them in it; the relations themselves (opposite signs, one comparison indistinguishable and
    # the other not, monotone drift against non-monotone outcome) are asserted in rederive(), because
    # no captured group can carry a relation and every one of these sentences is about one.
    chk("pooled l2 weight drift correlation",
        r"pooled \$\\ell_2\$ weight drift gives \$\\rho\{=\}\{\+\}([\d.]+)\$ with a CI including zero",
        R["pooled_drift_rho"])
    chk("Chronos/ETTh1: freezing is worse still",
        r"the frozen encoder is worse still \(\$\+([\d.]+)\\%\$ against \$\+([\d.]+)\\%\$\)",
        R["chronos_etth1_d"], R["chronos_etth1_b"])
    # The Chronos arm's drift range and the untrained-encoder floor it is compared against, in one
    # pattern: the sentence's force is entirely in "at or below", so a check that could pass with the
    # floor re-pointed at a different measurement would not be checking it.
    chk("Chronos CKA range against the random-init floor",
        r"CKA \$\{\\in\}\[([\d.]+), ([\d.]+)\]\$, at or below the \$([\d.]+)\{\\pm\}([\d.]+)\$",
        R["chronos_cka_lo"], R["chronos_cka_hi"], R["cka_floor"], R["cka_floor_sd"])
    chk("the random-init CKA floor (app:cka_calibration table)",
        r"random re-init, library-default & \$\\mathbf\{([\d.]+)\{\\pm\}([\d.]+)\}\$",
        R["cka_floor"], R["cka_floor_sd"])
    chk("the random-init CKA floor (it is not zero)",
        r"\\textbf\{([\d.]+)\}, not zero", R["cka_floor"])
    chk("the random-init CKA floor (normalisation left at their pre-trained values)",
        r"reports \$([\d.]+)\$ once normalisation", R["cka_floor"])
    chk("layer-unfreeze: same task outcome, different representation",
        r"indistinguishable on the task at CKA \$([\d.]+)\$ against \$([\d.]+)\$",
        R["lu3_cka"], R["lu6_cka"])
    chk("layer-unfreeze: the CKA gap in standard errors (appendix)",
        r"CKA \$([\d.]+)\$ against \$([\d.]+)\$ is a gap of \$([\d.]+)\$ at \$([\d.]+)\$~SE",
        R["lu3_cka"], R["lu6_cka"], R["lu_cka_gap"], R["lu_cka_se"])
    chk("layer-unfreeze: the two task outcomes and their difference (appendix)",
        r"gives forg\.\\ \$-([\d.]+)\$\\% \(SEM \$([\d.]+)\$, (\d+)/10 seeds negative\) against "
        r"\$N\{=\}6\$'s \$-([\d.]+)\$\\% \(SEM \$([\d.]+)\$, (\d+)/10 negative\), a ([\d.]+)-point "
        r"gap on a ([\d.]+)-point standard error of the difference, so \$([\d.]+)\$~SE",
        R["lu3_forg"], R["lu3_sem"], R["lu3_neg"], R["lu6_forg"], R["lu6_sem"], R["lu6_neg"],
        R["lu_forg_gap"], R["lu_forg_sed"], R["lu_forg_se"])
    # Both rows of tab:layerunfreeze in one pattern, so the two arms cannot be swapped -- which is the
    # one error that would leave the table internally consistent and its conclusion reversed. The
    # ridge column is matched but not captured: it is the WITHDRAWN probe quantity (S:retraction),
    # reported there only as a record, so registering it would give a retracted number a check.
    chk("layer-unfreeze: the matched 10-seed table",
        r"3 \(top-3\) & 3/6 & \$([\d.]+)\{\\pm\}([\d.]+)\$ & \$\+[\d.]+\{\\pm\}[\d.]+\$ \(\d+/10\) "
        r"& \$-([\d.]+)\{\\pm\}([\d.]+)\$\\% \((\d+)/10 neg\.\) \\\\ 6 \(full, B\) & 6/6 & "
        r"\$([\d.]+)\{\\pm\}([\d.]+)\$ & \$\+[\d.]+\{\\pm\}[\d.]+\$ \(\d+/10\) & "
        r"\$-([\d.]+)\{\\pm\}([\d.]+)\$\\% \((\d+)/10 neg\.\)",
        R["lu3_cka"], R["lu3_cka_sd"], R["lu3_forg"], R["lu3_forg_sd"], R["lu3_neg"],
        R["lu6_cka"], R["lu6_cka_sd"], R["lu6_forg"], R["lu6_forg_sd"], R["lu6_neg"])
    chk("Moirai-Large LoRA: the learning rate decides the sign, not the rank",
        r"near-identical CKA \(\$([\d.]+)\$ against \$([\d.]+)\$",
        R["lora_deflr_cka"], R["lora_lowlr_cka"])
    chk("Moirai-Large LoRA: the two outcomes (mitigation spectrum)",
        r"fails at the default learning rate \(\$\+([\d.]+)\\%\$\) and only a \$10\\times\$ LR "
        r"reduction rescues it \(\$-([\d.]+)\\%\$, CKA \$([\d.]+)\$\)",
        R["lora_deflr_forg"], R["lora_lowlr_forg"], R["lora_lowlr_cka"])
    chk("Moirai-Large LoRA: the rescued arm's dispersion and seed count",
        r"\(forg\.\$=-([\d.]+)\\pm([\d.]+)\\%\$, CKA\$=\$([\d.]+)\$\\pm\$([\d.]+), (\d+)~seeds\)",
        R["lora_lowlr_forg"], R["lora_lowlr_forg_sd"], R["lora_lowlr_cka"],
        R["lora_lowlr_cka_sd"], R["lora_lowlr_n"])
    chk("Moirai-Large LoRA: no rank recovers the sign",
        r"forgetting stays \$\+\$(\d+) to \$\+\$(\d+)\\% across",
        R["lora_rank_lo"], R["lora_rank_hi"])
    chk("sample sweep: the drift endpoints (body restatement)",
        r"\(CKA \$([\d.]+)\$ at 500 to \$([\d.]+)\$ at 10k\)",
        R["sw500_cka"], R["sw10000_cka"])
    chk("sample sweep: the three outcomes the sign reverses across (body restatement)",
        r"\$\+(\d+)\\%\$ at 500, \$-([\d.]+)\\%\$ by 2k, \$-([\d.]+)\{\\pm\}([\d.]+)\\%\$ at 10k "
        r"\((\d+)/10 negative\)",
        R["sw500_forg"], R["sw2000_forg"], R["sw10000_forg"], R["sw10000_sem"], R["sw10000_neg"])

    # --- the mean-vs-unanimous improver counts. The appendix states BOTH and says which the paper
    # uses; the figure asserts the unanimous one at draw time. `word()` is not available here, so the
    # counts are matched as digits where the prose uses digits and spelled out where it spells them.
    chk("the split-sign survivor (mean improves, seeds disagree)",
        r"its mean is a \$([\d.]+)\\%\$ \\emph\{improvement\} but (\d+) of its (\d+) seeds are harmful",
        R["split_forg_b"], R["split_pos"], R["split_seeds"])
    chk("mean-reading vs every-seed-reading improver counts",
        r"on the mean-only reading (\w+) of the seven improve and on the every-seed reading (\w+) do",
        R["n_imp_mean"], R["n_imp_unan"])

    # --- the two-cell from-scratch check (Reproducibility Statement). These are the only numbers in
    # the paper that come from a re-run rather than from the recorded runs, so they are the ones most
    # likely to be quietly left behind if the check is ever re-run on different hardware.
    chk("two-cell: TimesFM field count",
        r"all (\w+) fields, including CKA and weight drift, are bit-identical",
        R["rr_tfm_fields"])
    chk("two-cell: Moirai deviations",
        r"zero-shot MSE moves \$(-[\d.]+)\\%\$, held-out MSE \$(\+[\d.]+)\\%\$, "
        r"the selection-window MSE \$(\+[\d.]+)\\%\$, CKA \$(-[\d.]+)\\%\$ "
        r"\(\$([\d.]+)\$ to \$([\d.]+)\$\), and weight drift \$(\+[\d.]+)\\%\$, while "
        r"forg\.\$_\\text\{B\}\$ moves from \$\+([\d.]+)\$ to \$\+([\d.]+)\$, a "
        r"\$([\d.]+)\\%\$ relative change",
        R["rr_zs"], R["rr_test"], R["rr_val"], R["rr_cka"],
        R["rr_cka_stored"], R["rr_cka_fresh"], R["rr_drift"],
        R["rr_forg_stored"], R["rr_forg_fresh"], R["rr_forg"])
    # The mechanism sentence restates three of the same deviations, and it is the one place the paper
    # asserts WHY the ratio moves so far. Checked separately so that correcting a deviation above
    # without correcting the explanation is a failure rather than an internal contradiction.
    chk("two-cell: the amplification explanation",
        r"so a \$(-[\d.]+)\\%\$ move in the denominator against a \$(\+[\d.]+)\\%\$ move in "
        r"the numerator's larger term is amplified into \$\+([\d.]+)\\%\$ in the ratio",
        R["rr_zs"], R["rr_val"], R["rr_forg"])
    chk("two-cell: Moirai's sample count",
        r"Moirai scores the median of (\d+) sampled forecasts", R["rr_moirai_samples"])

    # --- the headline nulls
    chk("degradation count", r"definition of \\S\\ref\{sec:method\} to all (\d+)", R["n_int"])

    # --- claims added by the 20 Sep 2026 clarity pass
    # Figure 1's panel (b) claim, and the only number in the paper that comes from the figure script
    # rather than from a table: the fewest cells any single CKA cut misplaces. It is in the caption
    # only, so without this it would be the one load-bearing figure number with no prose check.
    # Round 7 (B3): the conclusion now quantifies what the shortcut costs with this same number, in the
    # same wording, so the floor rose to 2 -- a caption-only claim had become a body claim.
    # Back to 1 on 22 Sep 2026 (C1/C2): the CAPTION site is the one that went, not the conclusion's, and
    # it went because panel (b) prints the miscut count itself. So the check now guards the body sentence
    # that ASSERTS the cost, which is the site a rewrite could make dishonest; the drawn number is
    # guarded by the figure script's own assert.
    chk("Figure 1: the best CKA cut", r"the best threshold\s+misplaces (\w+) of the (\d+)",
        R["n_miscut"], R["n_int"], min_sites=1)

    # -- the pre-registered LOCO ladder, quoted in S6, in the abstract and in Appendix app:loco. Each
    # number is registered where it is PHRASED, not once per fact: S6 quotes the four rungs in one
    # sentence, the abstract quotes only the two that straddle CKA, and the appendix restates all of
    # them in prose. Three phrasings, three checks -- the alternative is the failure this project has
    # already had ten times, where one restatement drifts and every pattern still matches.
    chk("LOCO ladder (S6, four rungs in one sentence)",
        r"rises from \$([+-][\d.]+)\$ to\s+\$([+-][\d.]+)\$ on backbone identity and "
        r"\$([+-][\d.]+)\$ with horizon and \$\\log n\$, then\s+\\emph\{falls\} to \$([+-][\d.]+)\$",
        R["loco_r2_m0"], R["loco_r2_m1"], R["loco_r2_m2"], R["loco_r2_m3"])
    chk("LOCO dR2 and the share of unhelpful draws",
        r"\$\\Delta R\^2\{=\}\{-\}([\d.]+)\$; no help in\s+\$(\d+)\\%\$ of cluster-bootstrap draws",
        abs(R["loco_dr2"]), R["loco_frac"])
    chk("LOCO interaction rung (the worst of the five)",
        r"backbone-specific-slope variant is\s+worst of all at \$([+-][\d.]+)\$", R["loco_r2_m4"])
    # Both sites -- S6's "cluster interval" and the appendix's "cluster-bootstrap interval" -- in one
    # check, so the two cannot drift apart while each keeps matching its own pattern.
    chk("LOCO b1 and its cluster interval",
        r"\$([+-]\d+)\$~pp per unit CKA with a cluster(?:-bootstrap)? interval of "
        r"\$\[([+-]\d+), ([+-]\d+)\]\$",
        R["loco_b1"], R["loco_b1_lo"], R["loco_b1_hi"], min_sites=3)
    chk("LOCO fold count (S6 and the claims table)",
        r"leave-one-cluster-out \$R\^2\$ over the (\d+) clusters", R["loco_folds"], min_sites=2)
    chk("LOCO two-rung form (abstract and the claims table)",
        r"it \\emph\{lowers\} \$R\^2\$ from \$([+-][\d.]+)\$ to \$([+-][\d.]+)\$",
        R["loco_r2_m2"], R["loco_r2_m3"], min_sites=2)
    chk("LOCO fold count (abstract)", r"across\s+(\d+) leave-one-cluster-out folds", R["loco_folds"])
    chk("LOCO ladder (appendix restatement)",
        r"buys\s+\$R\^2_\\text\{LOCO\}\{=\}\{\+\}([\d.]+)\$; adding horizon and \$\\log n\$ takes it to "
        r"\$([+-][\d.]+)\$;\s+adding CKA \\emph\{lowers\} it to \$([+-][\d.]+)\$",
        R["loco_r2_m1"], R["loco_r2_m2"], R["loco_r2_m3"])

    # The spine sentence, which is the round's whole point: the paper's one claim has to reach the
    # reader in the SAME words in the title, the abstract, the introduction and the section that tests
    # it. Per the project's own experience, coverage here is per PHRASING and not per fact -- the
    # abstract and the intro lead contradicted the corrected counts for ten rounds because each
    # restatement was a site no pattern reached -- so the two phrasings are registered separately and
    # by site count. These two checks capture no numbers: they exist to fail if a rewrite drops a
    # restatement, which is a change no value-based check can see.
    # Four sites: the abstract, the introduction's lead, S6's opening and the conclusion's second
    # paragraph. The conclusion's site was one of the five "reliable ordering" sites until this round,
    # so that count drops to four by exactly the same edit -- the fact did not lose a restatement, one
    # restatement changed into the spine's wording, which is what this round is for. If a later edit
    # converts another ordering site, move the count between these two lines rather than lowering one.
    chk("the spine sentence, verbatim", r"does not reliably predict the latter", min_sites=4)
    # The title no longer restates the spine sentence: it carries the METHODOLOGICAL claim, and the
    # body carries the empirical one. Registered as its own check, and paired with the generalised
    # claim in the conclusion so that a rewrite cannot leave the title asserting something the body
    # never states. If these two ever disagree, the title is the one that is wrong.
    # Round 7 (B1): the title is now the reviewer's own narrower wording. The pattern is updated rather
    # than deleted, and it deliberately spans the line break in main.tex's two-line \title so that
    # dropping either half of the claim -- the subject (CKA) or the qualifier (reliably) -- fails here.
    chk("the methodological claim in the title",
        r"CKA Does Not Reliably Predict the Value of Encoder Adaptation\\*\s*in Time-Series"
        r" Foundation Model Fine-Tuning")
    chk("the methodological claim, generalised",
        r"cannot be assumed to identify the value of a\s+treatment that changed the representation")
    # Figure 1's caption states the same claim about the figure's two axes, so it cannot use
    # former/latter -- nothing in a caption sets those up. Registered separately for that reason:
    # patterns here are case-sensitive, so neither this nor the title check can cover the other.
    chk("the spine sentence in Figure 1's caption",
        r"measured drift does not\s+reliably predict the value of encoder adaptation")
    chk("the ordering claim, verbatim", r"does not provide a reliable ordering", min_sites=4)

    # --- the pre-registered positive control (S7.3 and app:poscontrol), added 21 Sep 2026.
    # Registered per PHRASING, not per fact, for the reason this file exists: the body and the appendix
    # state most of these numbers twice in different words, and a single pattern spanning both would
    # keep matching while one restatement drifted. Where the two DO share wording verbatim -- the task
    # A identifier, the destruction rule -- one check with min_sites=2 is the stronger version of the
    # same guarantee, because it also fails if one of the two sites is deleted.
    chk("positive control: task A, with its gate value",
        r"Moirai-Small/ETTh2, \$h\{=\}(\d+)\$, \$\\Vb\{\\text\{ridge\}\}\{=\}\{\+\}([\d.]+)\$",
        R["pc_horizon"], R["pc_v_ridge"], min_sites=2)
    chk("positive control: task A in the table caption",
        r"Task A is Moirai-Small/ETTh2 \$h\{=\}(\d+)\$", R["pc_horizon"])
    chk("positive control: validity condition (i), as the body phrases it",
        r"learned in \\textbf\{(\d+) of (\d+)\} runs \(minimum \$\+([\d.]+)\\%\$ against a "
        r"registered \$(\d+)\\%\$\)",
        R["pc_n_learned"], R["pc_n_runs"], R["pc_learn_min"], R["pc_learn_thresh"])
    chk("positive control: validity condition (i), as the appendix phrases it",
        r"in \\textbf\{(\d+) of (\d+)\} runs, by at least \$\+([\d.]+)\\%\$ against a registered "
        r"\$(\d+)\\%\$ threshold",
        R["pc_n_learned"], R["pc_n_runs"], R["pc_learn_min"], R["pc_learn_thresh"])
    chk("positive control: the registered validity threshold in the table caption",
        r"registered validity threshold \$(\d+)\\%\$", R["pc_learn_thresh"])
    # The destruction rule, quoted in the same words in both sections. This is the sentence the whole
    # arm turns on -- the rederivation above asserts outcome 3 -- so the threshold is checked where it
    # is stated rather than inferred from the outcome key.
    chk("positive control: the registered destruction rule",
        r"retention\$_A\\!\\geq\\!(\d+)\\%\$ in a majority", R["pc_des_thresh"], min_sites=2)
    chk("positive control: the destruction threshold restated",
        r"destruction threshold (?:is|of) \$\+(\d+)\\%\$", R["pc_des_thresh"], min_sites=2)
    chk("positive control: the worst retention anywhere, and the noise floor it beats",
        r"maximum over all (\d+) runs was \$\+([\d.]+)\\%\$, in one seed, against an evaluator "
        r"noise floor of \$([\d.]+)\$~pp",
        R["pc_n_runs"], R["pc_ret_max"], R["pc_noise_max"])
    chk("positive control: the per-rung retention maxima",
        r"per-rung maxima are \$([+-][\d.]+)\$, \$([+-][\d.]+)\$, \$([+-][\d.]+)\$ and "
        r"\$([+-][\d.]+)\$",
        R["pc_max_r1"], R["pc_max_r2"], R["pc_max_r3"], R["pc_max_r4"])
    # The dissociation the arm buys even though it failed to destroy anything: the top rung against the
    # gentlest one, all four numbers in one sentence so none can move alone.
    chk("positive control: the top rung against the gentlest",
        r"CKA on task A's inputs \$([\d.]+)\$, \$\\ell_2\$ weight drift \$(\d+)\$, against "
        r"\$([\d.]+)\$ and \$([\d.]+)\$ at the gentlest rung",
        R["pc_top_cka"], R["pc_top_drift"], R["pc_bot_cka"], R["pc_bot_drift"])
    chk("positive control: task A improves at the top rung",
        r"\\emph\{better\}, by \$([\d.]+)\\pm([\d.]+)\\%\$ in every seed",
        R["pc_top_ret"], R["pc_top_ret_sem"])
    # The rho and the n it is computed over, in one pattern: a correlation quoted without its n is the
    # one number in this arm a reader cannot sanity-check, and the run count had no site here.
    chk("positive control: CKA-vs-retention rho (body)",
        r"across all (\d+) runs the correlation between CKA\s*and retention is "
        r"\$\\rho\{=\}\{\+\}([\d.]+)\$", R["pc_n_runs"], R["pc_rho"])
    chk("positive control: CKA-vs-retention rho (appendix, with p and n)",
        r"retention is \$\+([\d.]+)\$ \(\$p\{=\}([\d.]+)\$, \$n\{=\}(\d+)\$\)",
        R["pc_rho"], R["pc_rho_p"], R["pc_rho_n"])
    chk("positive control: task B's period equals the horizon",
        r"\$P\{=\}(\d+)\{=\}h\$", R["pc_horizon"])
    chk("positive control: why task B conflicts",
        r"loses to its own negation by \$([\d.]+)\\times\$", R["pc_sn_ratio"])
    chk("positive control: the evaluator noise floor and the pairs it is derived from",
        r"Over the \$(\d+)\$ identical-state pairs that gap is at most \$([\d.]+)\$~pp and "
        r"\$([\d.]+)\$~pp on average",
        R["pc_noise_pairs"], R["pc_noise_max"], R["pc_noise_mean"])
    chk("positive control: the noise floor in the table caption",
        r"against the \$([\d.]+)\$~pp evaluator\s+noise floor", R["pc_noise_max"])
    chk("positive control: validity condition (ii), the three frozen-encoder rungs",
        r"completed rungs \(\$\+([\d.]+)\\pm([\d.]+)\$, \$\+([\d.]+)\\pm([\d.]+)\$, "
        r"\$\+([\d.]+)\\pm([\d.]+)\$\)",
        R["pc_d_r1"], R["pc_d_r1_sem"], R["pc_d_r2"], R["pc_d_r2_sem"],
        R["pc_d_r3"], R["pc_d_r3_sem"])
    chk("positive control: the diverged extension cells",
        r"Condition D diverged in (\d+) of (\d+)", R["pc_n_diverged"], R["pc_ext_seeds"])
    chk("positive control: run count and seeds in the table caption",
        r"positive control: (\d+) runs, no destruction", R["pc_n_runs"])
    chk("positive control: the seed count in the table caption",
        r"\$\\pm\$ is the SEM over (\d+) seeds", R["pc_seeds"])
    # What task A is worth. This is the arm's own stated limit -- it tests DETECTION of a loss, not the
    # value of what is lost. It had two registered phrasings, one per section, until 2026-09-24, when
    # the body sentence was cut for page 10's line budget; the appendix paragraph it duplicated says
    # strictly more, so the claim keeps a site and the body-form pattern is deleted rather than left
    # matching nothing. A pattern that matches zero sites is not a passing check, it is an absent one.
    chk("positive control: task A's margin over the ladder (appendix)",
        r"task A beats seasonal-naive by only\s+\$\+([\d.]+)\$", R["pc_v_rung"])
    # "no cell clears the strongest admissible rung" is now registered per split, beside the rest of
    # the ladder: "no cell clears every admissible rung (selection split)" for the body's 31 and
    # "union count (retrospective)" for the appendix's 32. A single pattern here matched both and
    # could only agree with one.

    # --- Chronos/M4-Monthly (app:chronos_detail, and one clause in S6). Every numeral in that section
    # is registered here or is gone, which is the check whose absence let ~140 lines of untraceable
    # numbers stand for ten rounds. Three deliberate exceptions, each of which is a numeral the section
    # exists to DISOWN rather than to state, and none of which can have a record by construction:
    #   84.5%            the superseded figure itself. There is no record; that is the claim. What is
    #                    registered is every number of the reproduction that fails to recover it.
    #   0.9993--0.9994   the deleted arm's frozen-encoder CKA, quoted twice and attributed to "the old
    #                    version of this appendix" both times, in the course of saying the protocol as
    #                    coded cannot produce it.
    #   n=500, n=10k     the names of two deleted arms, in the list of what was deleted.
    # A fourth category is checked against source rather than against a record: the four
    # finetune_chronos_m4.py line citations and the 80th-percentile split, all anchored below.
    chk("M4: the corrected gate, in the body", r"scores \$\+([\d.]+)\$ once the gate is computed",
        R["m4_gate"])
    chk("M4: the corrected gate, in the appendix",
        r"\\Vb\{\\text\{ridge\}\}\{=\}\{\+\}([\d.]+)\$, which \\emph\{fails\}", R["m4_gate"])
    chk("M4: the superseded trend rung and the constant floor",
        r"trend denominator reads \$\+([\d.]+)\$ on those same windows and is "
        r"\\textbf\{inadmissible\}: at an MSE of \$([\d.]+)\$ it is worse than the constant "
        r"predictor's \$([\d.]+)\$",
        R["m4_trend"], R["m4_trend_mse"], R["m4_const_mse"])
    chk("M4: the best and weakest admissible rungs",
        r"seasonal naive at \$\+([\d.]+)\$.*?GBM's \$\+([\d.]+)\$",
        R["m4_best_rung"], R["m4_worst_rung"])
    chk("M4: the held-out split's fitted and MLP rungs",
        r"the fitted gate reads \$\+([\d.]+)\$ and would pass, but the MLP rung reads \$-([\d.]+)\$",
        R["m4_gate_ho"], abs(R["m4_mlp_ho"]))
    chk("M4: the three splits' window counts",
        r"\(\$([\d]+)\{,\}([\d]+)\$ usable, \$(\d+)\$ dropped",
        R["m4_train_windows"] // 1000, R["m4_train_windows"] % 1000, R["m4_dropped"])
    chk("M4: the selection rule",
        r"first \$(\d+)\$ series in file order with at least \$(\d+)\$ observations of history and "
        r"\$(\d+)\$ test values",
        R["m4_series"], R["m4_minlen"], R["m4_horizon"])
    # The horizon is restated twice more, in the two sentences that describe what each split's target
    # is. Both are registered, because a horizon that disagreed with itself across three sentences is
    # exactly the kind of thing the deleted version of this section did.
    chk("M4: the selection target", r"contain the \$(\d+)\$ values used as the selection target",
        R["m4_horizon"])
    chk("M4: the held-out target", r"test set as the \$(\d+)\$ steps following the history",
        R["m4_horizon"])
    # The operating point itself. Registered globally rather than in the M4 block because the numeral
    # had no check at any of its sites; app:chronos_detail is simply where that was noticed.
    chk("the gate's operating point", r"\$([\d.]+)\$ operating point", R["gate_threshold"],
        min_sites=4)
    chk("the gate's operating point (stated the other way round)",
        r"operating point of \$([\d.]+)\$", R["gate_threshold"])
    # The THIRD and FOURTH phrasings, added 25 Sep 2026 after auditing app:chronos_detail for numerals
    # no chk pattern reaches. The two checks above cover "the $0.20$ operating point" and "operating
    # point of $0.20$" -- 5 sites between them. They reach none of the TWELVE sites that say the gate
    # "clears $0.20$", which is how the paper states the threshold most often. The audit that found this
    # is worth repeating rather than describing: extract every chk pattern from this file with ast, mark
    # the character spans they match in a section's collapsed text, and report the numerals lying
    # outside every span. Grepping this file for the numeral does NOT work -- expectations are derived
    # from records, so the digits are not in the source. Same lesson as the restated-claims one: the
    # checker's coverage is per PHRASING, and a fact with five checked sites can still have a sixth
    # phrasing nobody registered.
    #
    # The negative lookahead is load-bearing, not defensive. app:defsensitivity says of the one
    # positive instance "It clears $0.216$ against a threshold of $0.20$", where "clears" takes the
    # cell's GATE VALUE rather than the threshold; without the lookahead this check would demand
    # 0.216 == 0.20 and fail on a correct sentence. That site's threshold is registered by the check
    # below instead, so the sentence is covered twice over and neither number can drift alone.
    chk("the gate's operating point (what a cell clears)",
        r"clears? (?:the )?\$([\d.]+)\$(?! against a threshold)", R["gate_threshold"],
        min_sites=12)
    chk("the gate's operating point (the gate value stated against it)",
        r"against a threshold of \$([\d.]+)\$", R["gate_threshold"])
    chk("intervention-cell count (what the deleted M4 numbers were outside of)",
        r"outside the (\d+)-cell matrix", R["n_int"])
    chk("M4: where the superseded script cut its pool",
        r"pool at the \$(\d+)\$th percentile of its own index", R["m4_old_split_pct"])
    # The superseded figure, reproduced. The 84.5% itself is deliberately NOT checked against a record:
    # there is none, which is the claim. What is checked is every number of the reproduction.
    chk("M4 legacy: the old selection rule and what its estimator now gives",
        r"first \$(\d+)\$ series with at least \$(\d+)\$ observations, whose histories run from "
        r"\$(\d+)\$ to \$(\d+)\$ points --- gives \$([\d.]+)\\%\$",
        R["m4_series"], R["m4_legacy_minlen"], R["m4_legacy_hist_min"], R["m4_legacy_hist_max"],
        R["m4_legacy_pct"])
    chk("M4 legacy: the training mean beats it, so the defect is not weakness",
        r"the training mean is worse still, at \$([\d.]+)\$ against the legacy estimator's "
        r"\$([\d.]+)\$", R["m4_legacy_const"], R["m4_legacy_mse"])
    chk("M4 legacy: the per-series regressions that fit more coefficients than they have windows",
        r"that \$(\d+)\$ of the \$(\d+)\$ per-series regressions fit \$(\d+)\$ coefficients",
        R["m4_legacy_overparam"], R["m4_series"], R["m4_lookback"])
    chk("M4 legacy: the paper's own denominator on the same windows",
        r"fitted ridge reaches an MSE of \$([\d.]+)\$ on those same windows",
        R["m4_legacy_fitted_mse"])
    chk("M4 legacy: the same numerator under three estimators",
        r"that denominator the identical zero-shot numerator reads \$([\d.]+)\\%\$; against the "
        r"legacy one, \$([\d.]+)\\%\$; and on the split the protocol actually scores, \$\+([\d.]+)\$",
        R["m4_legacy_fitted_pct"], R["m4_legacy_pct"], R["m4_gate"])
    # The six runs.
    chk("M4: both arms' mean forgetting",
        r"full fine-tuning by \$-([\d.]+)\\%\$ and decoder-only adaptation by \$-([\d.]+)\\%\$",
        R["m4_forg_b"], R["m4_forg_d"])
    chk("M4: the paired contrast and its interval",
        r"\\denc\{=\}\{\+\}([\d.]+)\$ with a \$95\\%\$ interval of \$\[-([\d.]+), \+([\d.]+)\]\$",
        R["m4_denc"], abs(R["m4_ci_lo"]), R["m4_ci_hi"])
    chk("M4: the minimum detectable effect at three seeds",
        r"minimum detectable effect here is \$([\d.]+)\$~pp", R["m4_mde"])
    chk("M4: early stopping engaged past epoch 0",
        r"at epochs \$(\d+)\$, \$(\d+)\$ and \$(\d+)\$ under B against patience breaks at \$(\d+)\$ "
        r"to \$(\d+)\$",
        R["m4_best_epoch_1"], R["m4_best_epoch_2"], R["m4_best_epoch_3"],
        R["m4_stop_lo"], R["m4_stop_hi"])
    chk("M4: how far the encoder moved under B",
        r"\$\\ell_2\$ of \$([\d.]+)\$ to \$([\d.]+)\$ at CKA \$([\d.]+)\$ to \$([\d.]+)\$",
        R["m4_l2_lo"], R["m4_l2_hi"], R["m4_cka_b_lo"], R["m4_cka_b_hi"])
    chk("M4: the surviving descriptive observation",
        r"this dataset, at CKA \$\{\\geq\}([\d.]+)\$, where Moirai-Base on ETTh2 restructures to "
        r"CKA \$([\d.]+)\$ at \$h\{=\}(\d+)\$",
        R["m4_cka_b_lo"], R["m4_moirai_cka"], 192)
    # --- the cell's configuration. Each of these was a hand-typed constant in the deleted version of
    # the section, and two of them (n and the horizon) were wrong there by the time it was deleted.
    chk("M4: the cell's shape", r"M4-Monthly, at \$(\d+)\$ series, lookback \$(\d+)\$ and \$h\{=\}(\d+)\$",
        R["m4_series"], R["m4_lookback"], R["m4_horizon"])
    chk("M4: the training configuration",
        r"\$n\{=\}(\d+)\$ training windows, lr \$10\^\{-(\d)\}\$, batch \$(\d+)\$, \$(\d+)\$ epochs "
        r"with patience \$(\d+)\$",
        R["m4_n_train"], R["m4_lr_exp"], R["m4_batch"], R["m4_epochs"], R["m4_patience"])
    chk("M4: the tail held back from the training pool",
        r"sliding training windows over all but the last \$(\d+)\$ observations", R["m4_heldback"])
    chk("M4: the numerator's sample count",
        r"median of \$(\d+)\$ Chronos samples", R["m4_num_samples"])
    chk("M4: the CKA windows",
        r"mean-pooled final encoder output on the \$(\d+)\$ selection contexts", R["m4_cka_windows"])
    # The superseded script's own split, in the two numbers that describe it: stride-1 windows of
    # length `lookback` overlap in lookback-1 steps, so both come from the same recorded constant.
    chk("M4: how far the superseded split's windows overlapped",
        r"windows overlapping in \$(\d+)\$ of \$(\d+)\$ steps sat on both sides",
        R["m4_lookback"] - 1, R["m4_lookback"])
    # The gate-correction count, restated in this caption. The body's site says "moves 17 of Moirai's
    # 21 cells across the threshold" and has its own check; this phrasing is different, and coverage
    # here is per phrasing, not per fact.
    chk("M4: the gate correction's scale, restated in the caption",
        r"moved \$(\d+)\$ of \$(\d+)\$ Moirai cells when it was corrected",
        R["n_flip_moirai"], R["n_moirai_screened"])
    return C


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="print every site, not just failures")
    args = ap.parse_args()

    R = rederive()
    checks = build_checks(R)
    sources = [(p, *collapse(p)) for p in BODY]

    failures, n_sites = [], 0
    for c in checks:
        rx = re.compile(c["pattern"])
        hits = []
        for path, text, lines in sources:
            for m in rx.finditer(text):
                hits.append((path, lines[m.start()], m))
        if len(hits) < c["min_sites"]:
            failures.append(f"{c['name']}: matched {len(hits)} site(s), expected at least "
                            f"{c['min_sites']} -- pattern is stale or the claim was reworded:\n"
                            f"      /{c['pattern']}/")
            continue
        for path, lineno, m in hits:
            n_sites += 1
            groups = m.groups()
            if len(groups) != len(c["expect"]):
                failures.append(f"{c['name']}: pattern captures {len(groups)} group(s) but "
                                f"{len(c['expect'])} expected value(s) are registered")
                break
            for i, (tok, exp) in enumerate(zip(groups, c["expect"]), 1):
                try:
                    ok = agrees(tok, exp)
                except ValueError as e:
                    failures.append(f"{c['name']}: {e}")
                    continue
                where = f"paper_8/{path.relative_to(ROOT / 'paper_8')}:{lineno}"
                if not ok:
                    failures.append(
                        f"{c['name']} [group {i}] at {where}: prose says {tok!r}, "
                        f"records say {float(exp):.6g}\n      ...{m.group(0)[:150]}...")
                elif args.verbose:
                    print(f"  ok  {c['name']} [{i}] {where}: {tok} == {float(exp):.6g}")

    print(f"\n{len(checks)} registered claims, {n_sites} prose sites checked "
          f"against the run records.")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        print("\nFix the prose, or -- if the records moved -- fix the prose to match the records.")
        return 1
    print("All registered claims agree with the run records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
