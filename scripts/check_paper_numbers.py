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

    # -- the two cell counts the paper keeps confusing. n_screened is the gate denominator;
    # n_int is the intervention denominator, one smaller because TimesFM/Electricity has no B/D pair.
    gate_side = json.load(open(ROOT / "results/gate_test_side.json"))
    R["n_screened"] = len(gate_side)
    R["n_int"] = len(rows)
    R["n_gate_pass"] = sum(r["gate"] is not None and r["gate"] >= 0.20 for r in rows)
    R["n_gate_fail"] = R["n_screened"] - R["n_gate_pass"]
    R["n_clause_i_out"] = R["n_int"] - R["n_gate_pass"]
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

    # -- held-out vs selection-window scoring: the sign reversals
    R["n_heldout_rev"] = sum(
        1 for r in rows if r["bd_val"] is not None and r["bd_test"] * r["bd_val"] < 0)

    # -- the strict-freeze control. Recomputed here rather than read from the prose because this arm
    # grew mid-round twice and the prose lagged it both times.
    R["n_sf"] = len(sf)
    R["n_sf_reverse"] = sum(1 for v in sf.values() if v["bd_same"] * v["bh_test"] < 0)
    R["n_sf_agree"] = R["n_sf"] - R["n_sf_reverse"]
    R["n_sf_hneg"] = sum(1 for v in sf.values() if v["forg_h_pos"] == 0)
    R["n_d_only"] = R["n_int"] - R["n_sf"]

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
        src = (ROOT / "scripts/finetune_forecasting.py").read_text()
        R["rr_moirai_samples"] = int(re.search(r"\n *num_samples=(\d+),", src).group(1))
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
    for pred, tag in (("cka", "cka"), ("gate", "gate")):
        x = np.array([r[pred] for r in moirai])
        rho, _, _, _ = ck.cell_bootstrap_spearman(x, y, rng=np.random.default_rng(0))
        _, lo, hi, _, ncl = ck.cluster_bootstrap_spearman(
            x, y, cl, rng=np.random.default_rng(0))
        # magnitudes, because the sign is carried by the prose's own {+}/{-} and the pattern
        # anchors on it -- a flipped sign shows up as a zero-site match, not as a passing check
        R[f"{tag}_rho"], R[f"{tag}_lo"], R[f"{tag}_hi"] = abs(rho), abs(lo), hi
    R["n_moirai_clusters"] = ncl

    # -- the graded value axis and the pooled figure, from the stored artifact
    va = json.load(open(ROOT / "results/value_axis.json"))["correlations"]
    for key, tag in (("cka_vs_denc_all", "pooled"), ("cka_vs_denc_valuecells", "vc"),
                     ("cka_vs_denc_lowvalue", "lv"), ("value_vs_denc_all", "val")):
        c = va[key]
        R[f"{tag}_rho"], R[f"{tag}_lo"], R[f"{tag}_hi"] = c["rho"], abs(c["lo"]), c["hi"]
        R[f"{tag}_n"] = c["n"]
    inter = json.load(open(ROOT / "results/value_axis.json"))["interaction"]["cka_x_value"]
    R["b3"], R["b3_lo"], R["b3_hi"] = abs(inter["b"]), abs(inter["lo"]), inter["hi"]

    # -- paired inference at the value-cell aggregate
    agg = json.load(open(ROOT / "results/paired_inference.json"))["aggregate"]["value"]["d_enc"]
    R["agg_mean"] = abs(agg["mean"])
    R["agg_lo"], R["agg_hi"] = abs(agg["lo"]), agg["hi"]
    R["n_vc"], R["n_vc_pos"] = agg["n_cells"], agg["pos"]

    # -- the eight-rung ladder. "Admissible" = a denominator no worse than the training-mean floor,
    # which is the rule the caption states, so it is the rule reproduced here.
    gb = json.load(open(ROOT / "results/gate_baselines.json"))
    cells, T = gb["cells"], gb["gate_threshold"]
    rungs = ["fitted"] + [b for b in gb["ladder_order"] if b != gb["floor"]]
    counts = []
    for b in rungs:
        counts.append(sum(
            1 for per in cells.values()
            if b in per and per[b].get("r2_task") is not None
            and per[b]["linear_test"] <= per[gb["floor"]]["linear_test"]
            and per[b]["r2_task"] >= T))
    R["ladder_counts"] = counts
    strongest = {}
    for k, per in cells.items():
        adm = [(e["linear_test"], b, e["r2_task"]) for b, e in per.items()
               if b != gb["floor"] and e.get("r2_task") is not None
               and e["linear_test"] <= per[gb["floor"]]["linear_test"]]
        if adm:
            strongest[k] = min(adm)
    R["n_clear_strongest"] = sum(1 for v in strongest.values() if v[2] >= T)
    surv = [v[2] for k, v in strongest.items()
            if gate_side[k]["r2_task"] >= T]
    R["surv_strongest_hi"], R["surv_strongest_lo"] = abs(max(surv)), abs(min(surv))

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
    val_side = json.load(open(ROOT / "results/gate_val_side.json"))
    R["n_gate_pass_val"] = sum(1 for v in val_side.values() if v["r2_task"] >= 0.20)

    # -- the gate values quoted cell by cell. Whole passages of Exp 1 and the correction appendix
    # are lists of these, hand-typed, and they are the least likely thing to be re-checked by eye.
    R["gate_of"] = {k: v["r2_task"] for k, v in gate_side.items()}
    mo_gate = {k: v["r2_task"] for k, v in gate_side.items() if v.get("arm") == "moirai"}
    for ds in ("ETTh1", "Weather", "Electricity7", "ETTm2"):
        vals = [v for k, v in mo_gate.items() if k.split("_")[1] == ds]
        # "spans X to Y" runs from the least negative to the most negative, as the prose reads
        R[f"mo_{ds}_hi"], R[f"mo_{ds}_lo"] = abs(max(vals)), abs(min(vals))
    nonmo = [v["r2_task"] for k, v in gate_side.items()
             if v.get("arm") in ("chronos", "timesfm")]
    R["nonmo_lo"], R["nonmo_hi"] = abs(min(nonmo)), max(nonmo)
    R["ili_gate"] = abs(next(v["r2_task"] for v in gate_side.values() if v.get("arm") == "ili"))

    # -- the five survivors' own numbers, which the body quotes cell by cell
    by = {r["cell"]: r for r in rows}
    improvers = ["Moirai-base/ETTh2 h96 n1000", "Moirai-base/ETTh2 h192 n1000",
                 "Moirai-large/ETTh2 h96 n500"]
    for i, c in enumerate(improvers, 1):
        R[f"imp{i}_b"] = abs(by[c]["forg_b"])
        R[f"imp{i}_d"] = abs(by[c]["forg_d"])
        R[f"imp{i}_cka"] = by[c]["cka"]
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
    return R


# ---------------------------------------------------------------- the registry

def build_checks(R):
    """(name, pattern, expected-per-group). One entry per claim the body makes about a number."""
    C = []

    def chk(name, pattern, *expect, min_sites=1):
        C.append(dict(name=name, pattern=pattern, expect=expect, min_sites=min_sites))

    # --- denominators
    chk("screened-cell count", r"\\textbf\{(\d+) (?:screened|held-out) cells\}",
        R["n_screened"], min_sites=2)
    chk("screened-cell count (prose)", r"(?:Of|across) (\d+) screened cells", R["n_screened"])
    chk("screened-cell count (ladder)", r"of (\d+), so the screen", R["n_screened"])
    chk("screened-cell count (no cell clears)", r"no cell of the (\d+) clears", R["n_screened"])
    chk("intervention-cell count", r"(?:all |the )(\d+) intervention cells",
        R["n_int"], min_sites=2)
    chk("intervention-cell count (paired run)", r"(\d+) cells that carry a paired", R["n_int"])
    chk("intervention-cell count (matrix rows)", r"matrix has (\d+) rows", R["n_int"])
    chk("intervention-cell count (pooled)", r"Pooled across all (\d+) cells", R["n_int"])
    chk("intervention-cell count (conclusion)", r"across the (\d+) cells", R["n_int"])
    chk("gate-passing count", r"only (\d+) of (\d+) held-out cells",
        R["n_gate_pass"], R["n_screened"])
    chk("gate-passing count (conclusion)", r"(\d+) of (\d+) cells beat a lookback",
        R["n_gate_pass"], R["n_screened"])
    chk("gate-passing count (Moirai arm)", r"Moirai: (\d+) of (\d+)\}",
        R["n_gate_pass"], R["n_moirai_screened"])
    chk("gate-failing count", r"(\d+) of (?:the|our) (\d+) cells we screened",
        R["n_gate_fail"], R["n_screened"])
    chk("gate-failing count (disqualifier)", r"(\d+) of our (\d+) fail",
        R["n_gate_fail"], R["n_screened"])
    chk("gate-failing count (limitations)", r"(\d+) of (\d+) cells lose to a fitted",
        R["n_gate_fail"], R["n_screened"])
    chk("clause-(i) exclusions", r"removes (\d+) of the (\d+)",
        R["n_clause_i_out"], R["n_int"])

    # --- the outcome span
    chk("outcome span (abstract/conclusion)",
        r"outcome (?:still )?spans \$-([\d.]+)\\%\$ to \$\{?\+\}?([\d.]+)\\%\$",
        R["forg_b_min"], R["forg_b_max"])
    chk("outcome span (contributions)",
        r"Outcomes from \$-([\d.]+)\\%\$ to \$\{?\+\}?([\d.]+)\\%\$",
        R["forg_b_min"], R["forg_b_max"])
    chk("Chronos span", r"\$-([\d.]+)\\%\$ on ETTm2 to \$\{?\+\}?([\d.]+)\\%\$ on ETTh1",
        R["chronos_min"], R["chronos_max"])
    chk("Chronos span (contributions)",
        r"spans (\d+) points, from \$-([\d.]+)\\%\$ to \$\{?\+\}?([\d.]+)\\%\$",
        R["chronos_span"], R["chronos_min"], R["chronos_max"])
    chk("Chronos span (point count)", r"a (\d+)-point spread", R["chronos_span"])
    chk("TimesFM span", r"all improving \(\$-([\d.]+)\\%\$ to \$-([\d.]+)\\%\$\)",
        R["timesfm_min"], R["timesfm_max"])

    # --- held-out scoring
    chk("held-out sign reversals",
        r"reverses its sign in (\d+) of the (\d+) intervention cells",
        R["n_heldout_rev"], R["n_int"])
    chk("held-out sign reversals (abstract/contributions)",
        r"(\d+) of (\d+) cells (?:otherwise )?reverse",
        R["n_heldout_rev"], R["n_int"], min_sites=2)

    # --- the decomposition behind the reversals
    chk("largest reversal", r"moves from \$\+([\d.]+)\$ to \$-([\d.]+)\$, a (\d+)-point swing",
        R["chron_val"], R["chron_test"], R["chron_swing"])
    chk("reversal mechanism",
        r"\(\$\\Delta\$ frozen \$\+([\d.]+)\$ against \$\\Delta\$ adapted \$\+([\d.]+)\$",
        R["chron_dd"], R["chron_db"])
    chk("selection/held-out difficulty ratio",
        r"ratio spans \$([\d.]+)\$ to \$([\d.]+)\$ across these four cells",
        R["ratio_lo"], R["ratio_hi"])

    # --- strict freeze
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
    chk("strict-freeze D-only remainder", r"and (\d+) rest on D alone", R["n_d_only"])

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
    chk("within-Moirai CKA rho", r"within-backbone (?:Spearman )?\$\\rho\{=\}\{\+\}([\d.]+)",
        R["cka_rho"], min_sites=2)
    chk("within-Moirai CKA rho (body)",
        r"cells to ask, \$?\\rho\{=\}\{\+\}([\d.]+)", R["cka_rho"])
    chk("within-Moirai CKA clustered CI",
        r"clustered CI \$\[-([\d.]+),\{\+\}([\d.]+)\]\$",
        R["cka_lo"], R["cka_hi"], min_sites=3)
    chk("within-Moirai CKA rho (figure caption)",
        r"order them \(\$\\rho\{=\}\{\+\}([\d.]+)", R["cka_rho"])
    chk("Moirai cell count", r"\$n\{=\}(\d+)\$ cells from six series", R["n_moirai"])
    chk("Moirai cell count (rho over)", r"over (\d+) cells from six series", R["n_moirai"])

    # --- correlations: the gate against the intervention
    # Deliberately unanchored: every negative rho the paper prints is this one, and a new site that
    # quotes a different value should fail here rather than pass unnoticed. The one exception is the
    # LoRA arm's two CKA_E correlations, which have their own check above; they are excluded by the
    # words that follow them, not by their value, so a *wrong* value there still fails its own check.
    # The trailing \$ is load-bearing: without it [\d.]+ backtracks a digit at a time until the
    # lookahead is satisfied, so the exclusion silently becomes "match a prefix of any number".
    chk("gate rho within Moirai", r"\\rho\{=\}\{-\}([\d.]+)\$(?! against B\$-\$E)",
        R["gate_rho"], min_sites=3)

    # --- correlations: pooled and graded
    chk("pooled CKA rho and CI",
        r"\\rho\{=\}\{\+\}([\d.]+)\$, clustered CI \$\[\{\+\}([\d.]+),\{\+\}([\d.]+)\]",
        R["pooled_rho"], R["pooled_lo"], R["pooled_hi"])
    chk("value-cell CKA rho and CI",
        r"\\emph\{weaker\} \(\$\\rho\{=\}\{\+\}([\d.]+)\$, CI \$\[-([\d.]+),\{\+\}([\d.]+)\]\$\)",
        R["vc_rho"], R["vc_lo"], R["vc_hi"])
    chk("low-value CKA rho and CI",
        r"than on the (\d+) not \(\$\+([\d.]+)\$, CI \$\[-([\d.]+),\{\+\}([\d.]+)\]\$\)",
        R["lv_n"], R["lv_rho"], R["lv_lo"], R["lv_hi"])
    chk("value-cells clearing the screen", r"on the (\d+) clearing \$0\.20\$", R["vc_n"])
    chk("cells with nothing to lose", r"because (\d+) of the (\d+) cells beat no rung",
        R["lv_n"], R["n_int"])
    chk("interaction coefficient",
        r"\$b_3\{=\}\{-\}([\d.]+)\$~pp per unit CKA per unit value, "
        r"CI \$\[-([\d.]+),\{\+\}([\d.]+)\]",
        R["b3"], R["b3_lo"], R["b3_hi"])
    chk("value score rho and CI",
        r"either \(\$\\rho\{=\}\{\+\}([\d.]+)\$, CI \$\[-([\d.]+),\{\+\}([\d.]+)\]\$\)",
        R["val_rho"], R["val_lo"], R["val_hi"])

    # --- paired inference
    chk("value-cell aggregate Delta_enc",
        r"\\denc\{=\}\{-\}([\d.]+)\$ with CI \$\[-([\d.]+),\{\+\}([\d.]+)\]\$ and (\d+) of (\d+)",
        R["agg_mean"], R["agg_lo"], R["agg_hi"], R["n_vc_pos"], R["n_vc"])
    chk("value-cell count (aggregate)", r"over the (\d+) value-cells", R["n_vc"])

    # --- the ladder
    chk("ladder pass counts",
        r"floor is " + r", ".join([r"(\d+)"] * (len(R["ladder_counts"]) - 1))
        + r" and (\d+) of (\d+)",
        *R["ladder_counts"], R["n_screened"])
    chk("survivor range against the strongest rung",
        r"land\s*between \$-([\d.]+)\$ and \$-([\d.]+)\$",
        R["surv_strongest_hi"], R["surv_strongest_lo"])

    # --- the unfitted-baseline correction
    chk("Moirai gate flips (abstract)",
        r"moves (\d+) of (\d+) Moirai cells", R["n_flip_moirai_p2f"], R["n_moirai_screened"])
    # Both body sites must name the Moirai denominator. 17 is the Moirai flip count; across all
    # arms it is 26 of 32, so an unqualified "17 cells" in a 32-cell context reads as 17 of 32 and
    # understates the correction. The pattern requires the qualifier, so dropping it fails the check.
    chk("gate flips across the threshold",
        r"moves (\d+) of Moirai's (\d+) cells across the threshold",
        R["n_flip_moirai"], R["n_moirai_screened"])
    chk("gate flips (contributions)", r"\((\d+) of (\d+) Moirai cells flip\)",
        R["n_flip_moirai"], R["n_moirai_screened"])
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
        r"\\textbf\{(\d+) of (\d+) cells pass on the test split and (\d+) of (\d+) on validation\}",
        R["n_gate_pass"], R["n_screened"], R["n_gate_pass_val"], R["n_int"])

    # --- the closest candidate cell, quoted number for number
    chk("h192 survivor forgetting",
        r"forg\.\$_\\text\{B\}\{=\}\{\+\}([\d.]+)\{\\pm\}([\d.]+)\$ with (\d+) of (\d+) seeds",
        R["s192_forg_b"], R["s192_sem"], R["s192_pos"], R["s192_seeds"], min_sites=2)
    chk("h96 survivor forgetting",
        r"\(\$\+([\d.]+)\{\\pm\}([\d.]+)\$, (\d+) of (\d+) seeds\)",
        R["s96_forg_b"], R["s96_sem"], R["s96_pos"], R["s96_seeds"])
    chk("seed splits in the introduction",
        r"only (\d+)/(\d+) and (\d+)/(\d+) seeds agreeing",
        R["s96_pos"], R["s96_seeds"], R["s192_pos"], R["s192_seeds"])
    chk("seed splits in the abstract",
        r"(\d+)/(\d+) and (\d+)/(\d+) seed agreement",
        R["s96_pos"], R["s96_seeds"], R["s192_pos"], R["s192_seeds"])
    chk("closest candidate split", r"splits (\d+)/(\d+) on the sign",
        R["s192_pos"], R["s192_seeds"])

    # --- the gate values quoted cell by cell
    g = R["gate_of"]
    chk("the five survivors' gate values",
        r"ETTh2 at Small \$h\{=\}96\$ \(\$\+([\d.]+)\$\) and \$h\{=\}192\$ \(\$\+([\d.]+)\$\), "
        r"Base \$h\{=\}96\$ \(\$\+([\d.]+)\$\) and \$h\{=\}192\$ \(\$\+([\d.]+)\$\), "
        r"and Large \$h\{=\}96\$ \(\$\+([\d.]+)\$\)",
        g["small_ETTh2_h96"], g["small_ETTh2_h192"], g["base_ETTh2_h96"],
        g["base_ETTh2_h192"], g["large_ETTh2_h96"])
    chk("the failing datasets' gate ranges",
        r"ETTh1 spans \$-([\d.]+)\$ to \$-([\d.]+)\$, Weather \$-([\d.]+)\$ to \$-([\d.]+)\$, "
        r"\\texttt\{Electricity7\} \$-([\d.]+)\$ and \$-([\d.]+)\$, "
        r"and ETTm2 \$-([\d.]+)\$ to \$\+([\d.]+)\$",
        R["mo_ETTh1_hi"], R["mo_ETTh1_lo"], R["mo_Weather_hi"], R["mo_Weather_lo"],
        R["mo_Electricity7_lo"], R["mo_Electricity7_hi"],
        R["mo_ETTm2_lo"], R["mo_ETTm2_hi"])
    chk("non-Moirai gate range",
        r"scores run from \$-([\d.]+)\$ \(ETTm2\) to \$\+([\d.]+)\$ \(Electricity\)",
        R["nonmo_lo"], R["nonmo_hi"])
    for arm, label in (("chronos", "Chronos"), ("timesfm", "TimesFM")):
        chk(f"{label} per-cell gate values",
            r"\\textbf\{" + label + r":\}.{0,60}?before: "
            r"ETTh1 \$-([\d.]+)\$, ETTh2 \$-([\d.]+)\$, Weather \$-([\d.]+)\$, "
            r"ETTm2 \$-([\d.]+)\$, Electricity \$\+([\d.]+)\$",
            *(abs(g[f"{arm}_{d}"]) for d in
              ("etth1", "etth2", "weather", "ettm2", "electricity")))
    chk("ILI gate", r"R\^2_\\text\{task\}\(\\text\{PT\}\)\{=\}\{-\}([\d.]+)", R["ili_gate"])
    chk("ILI gate (appendix list)",
        r"\\textbf\{ILI\} \(Moirai-Small\): \$\\mathbf\{-([\d.]+)\}\$", R["ili_gate"])
    chk("ILI gate (appendix, restated)", r"it now carries \$-([\d.]+)\$", R["ili_gate"])

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
    chk("cells clearing the strongest rung",
        r"\\textbf\{no cell of the (\d+) clears", R["n_screened"])
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
