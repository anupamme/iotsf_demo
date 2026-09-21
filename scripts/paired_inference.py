#!/usr/bin/env python3
"""
Paired seed-level inference on the two interventions, per cell and in aggregate.

WHY THIS EXISTS. The paper decides whether a cell "degrades" with a three-clause definition whose
second and third clauses are SIGN COUNTS over seeds: every seed positive, every seed negative. A
reviewer is entitled to object that a unanimity count is not inference -- it reports no interval, no
effect size, and no p-value, so a cell with a large consistent effect and a cell with a tiny
consistent effect are recorded identically. This script reports the conventional quantities BESIDE
the counts, on the same runs, so the definition can be read against them rather than instead of them.

WHICH CONTRASTS, AND WHY THESE. CKA is observational: it is measured after the fact and nothing in
this design manipulates it. The thing the protocol actually manipulates is whether the encoder is
allowed to adapt. So the estimand is

    Delta_encoder = L_full - L_frozen      (condition B - condition D)

and the causal claim the runs support is about encoder adaptation, not about representation drift.
Two further contrasts are reported because they answer different questions:

    Delta_FT      = L_full   - L_zeroshot  (is fine-tuning at all an improvement?)
    Delta_frozen  = L_frozen - L_zeroshot  (is the frozen-encoder arm an improvement?)

All three are in the published units -- percent of the cell's zero-shot reference, so a positive
value means WORSE -- and are read off cell_matrix's per-seed lists, which reproduce bd_test, forg_b
and forg_d exactly. Nothing here recomputes a denominator.

WHAT THE INFERENCE CAN AND CANNOT DO, stated because it bounds every number below. The permutation
test is an exact sign-flip test, enumerated over all 2^N assignments. That makes its p-value
distribution-free, and it also makes its resolution a function of N alone: the smallest attainable
two-sided p is 2/2^N, which is 0.25 at N=3 and 0.0625 at N=5. **No 3-seed cell in this matrix can
reach p < 0.05 no matter how large or how consistent its effect is**, and the paired-N histogram is
{3: 24, 5: 4, 10: 3} -- so only 3 of the 31 cells can resolve anything at alpha=0.05 without a
distributional assumption. That is a real limit on what this matrix can establish, and it is a better
answer to "give me conventional inference" than an interval that looks precise at N=3. Cells are
flagged accordingly. The histogram is printed by main() and the caption's count is computed from the
same N the table's column prints, because a hand-typed seed count in this project has been wrong
before (an earlier draft of this docstring said 21).

Paired-ness is not uniform and is recorded per cell. Delta_encoder is genuinely paired everywhere:
B and D share a seed, an initialisation and a window sample. Delta_FT and Delta_frozen are paired on
the Chronos, TimesFM and ILI arms, whose run files carry each seed's own zero-shot test MSE, but NOT
on the 21 Moirai cells, which divide by a per-cell mean over condition-A seeds. Where they are not
paired, the reference's own measurement error is not propagated into the interval; the zs_sem column
reports how large that omitted term is rather than leaving it implicit.

THREE THINGS THIS SCRIPT ADDS BEYOND THE PER-CELL INTERVAL, each because reporting the intervals
alone invites a reading the intervals do not support.

  1. MULTIPLICITY. Thirty-one cells are tested, so at alpha=0.05 roughly 1.5 uncorrected rejections
     are expected under a complete null. Benjamini-Hochberg q-values are computed across the cells
     WITHIN each contrast family (31 tests per family, not 93 across families: the three contrasts
     answer three different questions and are not a single hypothesis). BH runs on the t-test p, not
     on the exact p: 24 of 31 cells have p_floor = 0.25, so a step-up procedure on the exact column
     would be arithmetic on a column that cannot fall below 0.25 and would reject nothing anywhere.
     That is a limitation of the exact column, stated rather than hidden by correcting the other one
     silently.

  2. A DECISION RULE with three outcomes, not two. On Delta_encoder each cell is called
     "freeze-decisive" (q < 0.05, mean > 0), "adapt-decisive" (q < 0.05, mean < 0) or INCONCLUSIVE.
     The third category is the point: the paper's published criterion recorded a cell with no
     measurable effect and a cell with a large effect the seeds disagreed about identically, as
     "not degrading". Under this rule the second says "inconclusive", which is what it is. The
     published three-clause unanimity definition is still computed, by cell_matrix.degradation_cells,
     and both are reported -- dropping a pre-registered criterion after seeing its outcome is the
     move the registration exists to prevent.

  3. POWER. mde is the smallest |mean| a paired two-sided t would resolve at that cell's own n and
     sd; n_needed is the n at which the OBSERVED effect would resolve. These turn "the interval is
     wide" into a number, and on two cells that number is large enough (69 and 80 seeds) that the
     honest statement is that the cell is unresolvable at any feasible budget rather than that its
     effect is absent. n_needed uses the observed sd, so it is a post-hoc estimate and is labelled
     one; it is not a substitute for a power calculation made in advance.

ZERO-SHOT UNCERTAINTY IS PROPAGATED, and only where it belongs. On the 21 Moirai cells the zero-shot
reference is a per-cell MEAN over condition-A seeds, so treating it as exact makes the Delta_FT and
Delta_frozen intervals narrower than the data warrant. Those two get a widened interval,
se_total = sqrt(se_within^2 + zs_sem^2) -- a first-order propagation that ignores the correlation
induced by the reference appearing in both numerator and denominator, and widens, so it is the
conservative direction. Delta_encoder gets NO widening and must not: the reference cancels from its
numerator and survives only in the denominator as a common scale factor. That asymmetry is why
Delta_encoder is the estimand, and all seven gate-passing cells are cells where it matters, since
every one of them has ft_paired = False.

Regenerate with:  .venv12/bin/python scripts/paired_inference.py
"""
import argparse
import collections
import contextlib
import io
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cell_matrix as cm                                          # noqa: E402
import gate_all_cells as gac                                      # noqa: E402
import gate_baseline_sensitivity as gbs                           # noqa: E402
from cluster_keys import cluster_of                               # noqa: E402

OUT_JSON = ROOT / "results/paired_inference.json"
OUT_TEX = ROOT / "paper_8/tables/paired_inference.tex"
OUT_POWER_TEX = ROOT / "paper_8/tables/power_mde.tex"
# The SELECTION-split ladder, because that is the split the paper's admission decisions are made on:
# a value-cell is admitted by windows disjoint from the held-out ones the aggregate below is computed
# from. results/gate_baselines.json is the retrospective test-side ladder and is reachable with
# --gate; it must not be the default, or this table's value-cell marks would disagree with S4's count.
GATE_BASELINES = ROOT / "results/gate_baselines_val.json"

CONTRASTS = (("d_enc", r"$\Delta_\text{enc}$"),
             ("d_ft", r"$\Delta_\text{FT}$"),
             ("d_frozen", r"$\Delta_\text{frz}$"))

# 2^N enumeration is exact and cheap at these sizes; the largest cell runs 10 seeds (1,024 flips).
MAX_EXACT = 20

# The decision level for the BH-adjusted call. One number, named once, so the table, the JSON and
# every count in the body come from the same constant.
ALPHA = 0.05

# Two different caps, and conflating them is how a power statement turns into a to-do list.
# N_SEARCH_CAP is how far the search looks, so the required budget can be REPORTED even when it is
# absurd. N_FEASIBLE is what this project can actually run: a Moirai-Small cell costs ~25-30 min per
# fine-tune and each paired seed needs two of them (B and D), so 20 paired seeds is ~20 GPU-hours for
# one cell and 71 is a week of one cell. A cell whose n_needed exceeds N_FEASIBLE is reported as
# unresolvable under this design, with the number, rather than as an effect that is absent.
N_SEARCH_CAP = 400
N_FEASIBLE = 20


def exact_sign_flip_p(x):
    """Two-sided p from enumerating all 2^N sign assignments of the paired differences.

    The null is that each paired difference is equally likely to have come out with either sign,
    which is the exchangeability the pairing buys. Returns (p, n_arrangements, p_floor) where
    p_floor = 2/2^N is the smallest p this N can produce -- reported so a non-significant result at
    N=3 is not mistaken for evidence of no effect.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return float("nan"), 0, float("nan")
    obs = abs(x.mean())
    if n > MAX_EXACT:
        raise ValueError(f"{n} seeds exceeds exact enumeration cap {MAX_EXACT}")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    means = np.abs(signs @ x) / n
    # >= with a tolerance: the identity arrangement must count itself, and floating-point means of
    # the same multiset must not be split by rounding.
    p = float((means >= obs - 1e-12).mean())
    return p, len(signs), 2.0 / 2 ** n


def paired_bootstrap_ci(x, n_boot=10000, seed=0):
    """Percentile CI for the mean of the paired differences, resampling SEEDS with replacement.

    NOT the interval this table reports, and the reason is a construction artifact worth naming.
    Every bootstrap resample mean lies inside [min(x), max(x)], so at N=3 with all three observations
    on one side of zero the percentile interval EXCLUDES zero no matter how small the effect --
    "CI excludes zero" is then a restatement of "all seeds agreed in sign", which is the sign count
    the interval was supposed to improve on. Kept in the JSON for the cells with enough seeds for it
    to mean something, and reported beside the t-interval so the gap is visible rather than assumed
    away.
    """
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def t_interval(x):
    """Paired t interval and p-value on the differences: (lo, hi, p).

    This is the interval the table reports. It buys resolution at N=3, where the exact test cannot
    reach p<0.05 and the percentile bootstrap cannot reach zero, and it pays for it with a normality
    assumption on three points -- which is not checkable. So it is reported BESIDE the exact p rather
    than instead of it: agreement means the reading does not rest on the assumption, and the cells
    where they disagree are exactly the cells whose evidence is assumption-dependent.
    """
    from scipy import stats

    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    r = stats.ttest_1samp(np.asarray(x, float), 0.0)
    ci = r.confidence_interval(0.95)
    return float(ci.low), float(ci.high), float(r.pvalue)


def cohens_d(x):
    """Paired Cohen's d: mean difference over its sample sd (ddof=1). None when N < 2 or sd == 0."""
    if len(x) < 2:
        return None
    sd = st.stdev(x)
    return st.mean(x) / sd if sd > 0 else None


def benjamini_hochberg(pvals):
    """BH step-up adjusted p-values (q), aligned with the input order. NaN p's pass through as NaN.

    Written out rather than taken from statsmodels because this project's analysis venv has no
    statsmodels and adding a dependency for eleven lines is a worse trade than the eleven lines. The
    monotonicity pass (cumulative minimum from the largest p downwards) is what makes the output a
    valid adjusted p-value rather than a raw p*m/rank, which is not monotone in p and can rank two
    cells in the opposite order to their evidence.
    """
    idx = [i for i, p in enumerate(pvals) if p is not None and p == p]
    m = len(idx)
    q = [float("nan")] * len(pvals)
    if not m:
        return q
    order = sorted(idx, key=lambda i: pvals[i])
    running = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        running = min(running, pvals[i] * m / rank)
        q[i] = float(min(1.0, running))
    return q


def mde(n, sd, alpha=ALPHA):
    """Smallest |mean| a two-sided paired t at level alpha resolves, given n and sd. None if n < 2.

    t_{1-alpha/2, n-1} * sd / sqrt(n). This is the detection boundary implied by the cell's OWN
    dispersion, not by an assumed effect size, which is why it can be reported for cells that were
    already run: it says what this cell was capable of showing, which is the question a reader asking
    "is a wide interval evidence of no effect?" is actually asking.
    """
    from scipy import stats

    if n is None or n < 2 or sd is None or sd <= 0:
        return None
    return float(stats.t.ppf(1 - alpha / 2, n - 1) * sd / n ** 0.5)


def n_for_effect(effect, sd, alpha=ALPHA, cap=N_SEARCH_CAP):
    """Smallest n in [2, cap] whose mde is no larger than |effect|; None if cap is not enough.

    Post-hoc: sd is the observed sd, so this is an estimate of the budget the observed effect would
    have needed, not a design calculation. Returning None rather than a number past the cap is
    deliberate -- "unresolvable within 200 seeds" is the finding, and printing 340 would invite a
    reader to treat it as a plan.
    """
    if effect is None or sd is None or sd <= 0 or effect == 0:
        return None
    target = abs(effect)
    for n in range(2, cap + 1):
        m = mde(n, sd, alpha)
        if m is not None and m <= target:
            return n
    return None


def propagate_zs(s, zs_sem, alpha=ALPHA):
    """Widen a contrast's interval by the shared zero-shot reference's own SEM. Mutates and returns s.

    Only for Delta_FT and Delta_frozen on cells with ft_paired = False, where the reference is a mean
    over condition-A seeds rather than the seed's own measurement. se_total = sqrt(se_within^2 +
    zs_sem^2) treats the two as independent and ignores the reference's appearance in the denominator;
    both simplifications widen relative to a full delta-method treatment of a ratio whose numerator
    and denominator share a term, so the reported interval is the conservative one. Fields are ADDED
    (lo_zs/hi_zs/p_zs/se_zs) rather than overwriting lo/hi, so the unpropagated interval stays
    auditable and the verification step can assert that the widening actually happened.
    """
    from scipy import stats

    n = s["n"]
    se_within = s["sem"]
    extra = float(zs_sem or 0.0)
    se = (se_within ** 2 + extra ** 2) ** 0.5
    s["se_within"], s["zs_sem_used"], s["se_zs"] = se_within, extra, se
    # A zero here is UNESTIMATED, not zero: eight of the 21 unpaired cells measured their zero-shot
    # reference on a single condition-A seed, which has no SEM. Recording the distinction stops the
    # widening from being read as "we checked and there was nothing to propagate" on those cells.
    s["zs_sem_estimated"] = bool(zs_sem)
    if n is None or n < 2 or se <= 0:
        s["lo_zs"] = s["hi_zs"] = s["p_zs"] = float("nan")
        return s
    half = stats.t.ppf(1 - alpha / 2, n - 1) * se
    s["lo_zs"], s["hi_zs"] = s["mean"] - half, s["mean"] + half
    s["p_zs"] = float(2 * stats.t.sf(abs(s["mean"]) / se, n - 1))
    return s


def summarise(x, n_boot, seed):
    boot_lo, boot_hi = paired_bootstrap_ci(x, n_boot, seed)
    t_lo, t_hi, t_p = t_interval(x)
    p, arr, p_floor = exact_sign_flip_p(x)
    sd, mean = cm._sd(x), st.mean(x) if x else None
    return dict(n=len(x), mean=mean,
                sd=sd, sem=cm._sem(x),
                lo=t_lo, hi=t_hi, t_p=t_p,
                boot_lo=boot_lo, boot_hi=boot_hi,
                p=p, arrangements=arr, p_floor=p_floor, d=cohens_d(x),
                # q is filled in by add_multiplicity once all cells exist: BH is a property of the
                # family, not of the cell, so a per-cell function cannot compute it.
                q=None,
                mde=mde(len(x), sd), n_needed=n_for_effect(mean, sd),
                pos=sum(v > 0 for v in x), neg=sum(v < 0 for v in x))


def add_multiplicity(cells):
    """Fill each contrast's q by running BH across the cells within that contrast family.

    Within, not across: the three contrasts answer three questions (does the encoder intervention
    matter; does fine-tuning at all help; does the frozen arm help), and pooling 93 tests into one
    family would penalise each question for the others having been asked.
    """
    for key, _ in CONTRASTS:
        have = [c for c in cells if c.get(key)]
        m = len(have)
        qs = benjamini_hochberg([c[key]["t_p"] for c in have])
        for c, q in zip(have, qs):
            s = c[key]
            s["q"], s["m_tests"] = q, m
            # The multiplicity-adjusted power boundary, and the reason it has to exist: mde/n_needed
            # above are computed at an UNCORRECTED alpha, so they answer "what would a single test
            # resolve?" while the call answers "what survives BH across 31 tests?". Those differ, and
            # the gap is not academic -- Moirai-Base/ETTh2 h96 has n_needed = 3 at its own n = 3 and
            # still lands at q = 0.071. Reporting only the uncorrected budget would therefore tell a
            # reader that cell is already resolved when the paper's own call says it is not.
            #
            # alpha/m is Bonferroni, which is CONSERVATIVE relative to the BH threshold the call uses
            # (BH rejects at p <= alpha*k/m for rank k, and k >= 1), so n_needed_adj is an upper bound
            # on the budget a decisive call needs, not an estimate of it. An upper bound is the useful
            # direction here: it is what we would have to commit to before running.
            s["alpha_adj"] = ALPHA / m
            s["mde_adj"] = mde(s["n"], s["sd"], s["alpha_adj"])
            s["n_needed_adj"] = n_for_effect(s["mean"], s["sd"], s["alpha_adj"])
    return cells


def call_of(s, alpha=ALPHA):
    """Three-way call on a contrast: 'freeze' / 'adapt' / 'inconclusive'.

    Positive Delta_enc means allowing the encoder to adapt made the cell worse, so a decisive
    positive is a cell where FREEZING is the better choice. 'inconclusive' is a real category and not
    a polite 'no': it covers both a cell with a small effect and a cell with a large effect its seeds
    disagree about, and the mde field is what separates those two.
    """
    if not s or s["q"] is None or s["q"] != s["q"] or s["mean"] is None:
        return "inconclusive"
    if s["q"] >= alpha:
        return "inconclusive"
    return "freeze" if s["mean"] > 0 else "adapt"


def degrades_ci(c, gate_threshold=gac.GATE_THRESHOLD):
    """The published three-clause degradation definition with the unanimity clauses made intervals.

    Clause (i) is unchanged -- the cell must clear the screen. Clauses (ii) and (iii) become
    "the interval lies entirely on the required side of zero" instead of "every seed does", and they
    use the ZS-PROPAGATED interval, because both compare an intervention arm against the shared
    zero-shot reference and so both carry the reference's own error. A cell can fail this and pass the
    unanimity version (three same-sign seeds with a wide interval) or pass this and fail unanimity
    (nine of ten seeds one way with a decisive mean); reporting both is how those two cases stay
    visible.
    """
    ft, fz = c.get("d_ft"), c.get("d_frozen")
    # forg_confounded excluded for the same reason cell_matrix.degradation_cells excludes it: on the
    # Chronos cells the per-condition forgetting terms carry a head/decoder mismatch, so clauses (ii)
    # and (iii) are not measuring what they name. Their Delta_enc is unaffected and is still called.
    if c["forg_confounded"] or c["gate"] is None or c["gate"] < gate_threshold or not ft or not fz:
        return False
    lo_ft = ft.get("lo_zs", ft["lo"])
    hi_fz = fz.get("hi_zs", fz["hi"])
    return bool(lo_ft == lo_ft and hi_fz == hi_fz and lo_ft > 0 and hi_fz < 0)


def degrades_unanimous(c, gate_threshold=gac.GATE_THRESHOLD):
    """The published criterion, restated on the same per-seed lists so the two can be tallied together.

    This must agree cell-for-cell with cell_matrix.degradation_cells, which is the definition of
    record; main() asserts that it does rather than trusting two spellings of one rule to stay in
    step. Unanimity of the per-seed signs implies the mean-sign clauses that function also applies,
    so those are not repeated here.
    """
    ft, fz = c.get("d_ft"), c.get("d_frozen")
    if c["forg_confounded"] or c["gate"] is None or c["gate"] < gate_threshold or not ft or not fz:
        return False
    return bool(ft["pos"] == ft["n"] and fz["neg"] == fz["n"])


def value_cells(rows, gate_json=GATE_BASELINES):
    """Cell refs clearing the gate threshold under at least one ADMISSIBLE baseline.

    Reuses gate_baseline_sensitivity.passes/admissible rather than re-deriving the rule, so "value
    cell" means the same thing here as in the baseline-ladder appendix. A baseline whose denominator
    is worse than the training-mean floor cannot admit a cell: it would be measuring value against a
    predictor that is itself beaten by a constant.
    """
    gb = json.load(open(gate_json))
    results, baselines = gb["cells"], gb["baselines"]
    out = {}
    for r in rows:
        ref = r["ref"]
        if ref not in results:
            continue
        clearing = [b for b in baselines
                    if b != gbs.FLOOR and gbs.passes(results, ref, b, require_admissible=True)]
        if clearing:
            out[ref] = clearing
    return out


def analyse(rows, n_boot=10000, seed=0, gate_json=GATE_BASELINES):
    vc = value_cells(rows, gate_json)
    cells = []
    for r in rows:
        ps = r["per_seed"]
        rec = dict(cell=r["cell"], ref=r["ref"], cluster=cluster_of(r["cell"]),
                   seeds=r["seeds"], gate=r["gate"], cka=r["cka"], n_train=r.get("n_train"),
                   is_value_cell=r["ref"] in vc, clearing=vc.get(r["ref"], []),
                   ft_paired=ps["ft_paired"], zs_seeds=ps["zs_seeds"], zs_sem=ps["zs_sem"],
                   forg_confounded=bool(r.get("forg_confounded")))
        for key, _ in CONTRASTS:
            rec[key] = summarise(ps[key], n_boot, seed) if ps[key] else None
        # d_enc's reference cancels from the numerator, so it is NOT widened; the other two compare an
        # arm against the shared zero-shot mean and are. On ft_paired cells the widening is a no-op by
        # construction (zs_sem is None there), which is the desired behaviour and not a special case.
        for key in ("d_ft", "d_frozen"):
            if rec[key] is not None:
                propagate_zs(rec[key], None if ps["ft_paired"] else ps["zs_sem"])
        if rec["d_enc"] is not None:
            rec["d_enc"]["zs_propagated"] = False
        cells.append(rec)
    assert_zs_propagation(cells)
    add_multiplicity(cells)
    for c in cells:
        c["call"] = call_of(c.get("d_enc"))
        c["degrades_ci"] = degrades_ci(c)
        c["degrades_unanimous"] = degrades_unanimous(c)
    return cells, vc


def assert_zs_propagation(cells):
    """Fail loudly if the widening did not happen where it must, or happened where it must not.

    The claim in S3 and app:pairedinference is a claim about two directions at once: the arm-vs-zero-shot
    intervals are widened by the reference's own SEM, and Delta_enc is NOT, because the reference
    cancels from its numerator. A silent regression in either direction would leave the prose describing
    an analysis the emitter no longer performs, and an interval that is too narrow is exactly the defect
    the reviewer named. So it is asserted rather than documented: at least one unpaired cell must show a
    strictly wider propagated interval, Delta_enc must carry no propagated interval on any cell, and no
    cell may come out narrower.
    """
    widened = 0
    for c in cells:
        d = c.get("d_enc")
        if d is not None and ("lo_zs" in d or "hi_zs" in d):
            raise AssertionError(f"{c['cell']}: d_enc must not carry a zero-shot-propagated interval")
        for key in ("d_ft", "d_frozen"):
            s = c.get(key)
            if s is None or s["lo_zs"] != s["lo_zs"]:      # None, or a NaN interval on n < 2
                continue
            if s["se_zs"] < s["se_within"] - 1e-12:
                raise AssertionError(f"{c['cell']}/{key}: propagation NARROWED the interval "
                                     f"({s['se_zs']} < {s['se_within']})")
            if not c["ft_paired"] and s["zs_sem_used"] > 0:
                if not (s["lo_zs"] < s["lo"] and s["hi_zs"] > s["hi"]):
                    raise AssertionError(
                        f"{c['cell']}/{key}: zs_sem={s['zs_sem_used']} but the interval did not widen "
                        f"([{s['lo']},{s['hi']}] -> [{s['lo_zs']},{s['hi_zs']}])")
                widened += 1
    if widened == 0:
        raise AssertionError("no cell's arm-vs-zero-shot interval was widened; the propagation is dead "
                             "code and the prose describing it is wrong")
    return widened


def cluster_aggregate(cells, key, subset, n_boot=10000, seed=0):
    """Mean of the per-cell means over `subset`, with a CI from resampling (backbone, dataset).

    Cells are not independent: the 31 reuse 3 backbones and 7 series, so a cell-level interval is too
    narrow. Clusters are resampled instead, which is the same level cluster_keys.py defines for every
    other cross-cell interval in the paper.
    """
    sel = [c for c in cells if c["ref"] in subset and c.get(key)]
    if not sel:
        return None
    vals = np.array([c[key]["mean"] for c in sel], float)
    cl = np.array([c["cluster"] for c in sel])
    uniq = np.unique(cl)
    idx_by = {c: np.flatnonzero(cl == c) for c in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        boots.append(vals[np.concatenate([idx_by[c] for c in draw])].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n_cells=len(sel), n_clusters=len(uniq), mean=float(vals.mean()),
                lo=float(lo), hi=float(hi),
                pos=int((vals > 0).sum()), neg=int((vals < 0).sum()))


def fmt(v, places=1, signed=True):
    if v is None or (isinstance(v, float) and v != v):
        return "--"
    s = f"{v:+.{places}f}" if signed else f"{v:.{places}f}"
    return s.replace("+", "$+$").replace("-", "$-$")


def emit_tex(cells, agg, path=OUT_TEX):
    """One row per cell: Delta_enc with its interval, exact p, effect size and sign count."""
    vcells = [c for c in cells if c["is_value_cell"]]
    # Counted off the SAME N the table's column prints, not off row["seeds"]: they agree today, but
    # they are different quantities (seeds run vs seeds surviving the B/D intersection), and a caption
    # that counts one while the column shows the other is the kind of claim that silently goes stale.
    n3 = sum(1 for c in cells if c.get("d_enc") and c["d_enc"]["n"] == 3)
    L = ["% GENERATED by scripts/paired_inference.py -- do not edit by hand.",
         r"\begin{table}[t]", r"\centering",
         r"\caption{\textbf{Paired seed-level inference on encoder adaptation.}",
         r"$\Delta_\text{enc}={L_\text{full}}-{L_\text{frozen}}$ per seed, in percent of the cell's",
         r"zero-shot reference, so a positive value means allowing the encoder to adapt made the cell",
         r"\emph{worse}.  B and D share a seed, an initialisation and a window sample, so this",
         r"contrast is paired by construction.",
         r"$d$ is paired Cohen's $d$.  The SEM column was dropped when $q$ was added---nine columns",
         r"overrun the textwidth and the interval states the same dispersion---so \texttt{sem} lives",
         r"in \texttt{results/paired\_inference.json}, on the body's convention (\S\ref{sec:method}).",
         r"Each row names its own $n$, which is not constant down the column",
         r"(Table~\ref{tab:heldout_all}).",
         rf"\textbf{{$q$ is the decision column, adjusted for the {len(cells)} tests}}:",
         r"Benjamini--Hochberg within this contrast, with the three-way call---freeze-decisive,",
         r"adapt-decisive, inconclusive---read off $q{<}0.05$ and the sign of $\denc$, and bolded",
         rf"where decisive.  BH runs on $p_t$, not $p_\text{{exact}}$, because {n3} cells have an",
         r"exact floor of $0.25$ and a step-up procedure there would reject nothing at any effect",
         r"size.",
         # The exact-test-resolution and no-bootstrap arguments used to be spelled out here in
         # fourteen lines. They are now stated at length in the prose of this appendix section, and
         # carrying both made the float taller than the page ("Float too large for page by 34pt").
         # Compressed to the two facts a reader needs to read the columns, with the argument left
         # to the text that already makes it.
         r"\textbf{Two tests, because neither alone is adequate at these seed counts.}",
         rf"$p_\text{{exact}}$ assumes nothing, but its resolution is fixed by $N$: the {n3} cells at",
         r"$N{=}3$ cannot reach $p<0.05$ at any effect size, their smallest attainable value being",
         r"$0.25$.  $p_t$ and the interval do resolve there, and pay for it with a normality",
         r"assumption on three points that cannot be checked.  Both are shown so that a reading",
         r"resting on that assumption is visible as one; the text states why no bootstrap interval",
         r"is reported.",
         r"In the two summary rows the last column is \emph{not} an effect size: it gives the number",
         r"of cells whose mean $\Delta_\text{enc}$ is positive, out of the cells in that row.",
         r"$\dagger$ marks cells clearing the gate threshold under at least one admissible baseline",
         r"of the ladder (Appendix~\ref{app:baselines}).",
         r"$\ddagger$ marks the Chronos cells, whose per-condition forgetting is confounded by the",
         r"head/decoder mismatch; their $\Delta_\text{enc}$ is unaffected.}",
         # \footnotesize and a tightened tabcolsep, not \small: nine columns with a bracketed CI
         # among them overruns the 5.5in ICLR textwidth by 19pt at \small, and an overfull hbox in a
         # generated table is a defect the emitter should not be able to reintroduce. The q column
         # was added on 22 Sep 2026 and the width was re-checked in the log, not assumed.
         r"\label{tab:paired_inference}", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}l r r r rrr r@{}}", r"\toprule",
         r"Cell & $N$ & $\Delta_\text{enc}$ & $95\%$ CI ($t$) & $p_t$ & $q$ & "
         r"$p_\text{exact}$ & $d$ \\",
         r"\midrule"]
    for c in sorted(cells, key=lambda c: (not c["is_value_cell"], c["cell"])):
        e = c["d_enc"]
        if not e:
            continue
        mark = (r"\textsuperscript{$\dagger$}" if c["is_value_cell"] else "")
        mark += (r"\textsuperscript{$\ddagger$}" if c["forg_confounded"] else "")
        ci = f"[{fmt(e['lo'])}, {fmt(e['hi'])}]"
        d = fmt(e["d"], 2) if e["d"] is not None else "--"
        # cm._display, not c["cell"]: the internal keys carry raw underscores ("Moirai-base_ETTm2_h192")
        # which LaTeX rejects, and reusing the one display helper keeps cell names identical across
        # every generated table rather than inventing a second spelling here.
        if c["seeds"] != e["n"]:
            sys.exit(f"{c['cell']}: {c['seeds']} seeds run but {e['n']} paired differences; the N "
                     f"column must report the paired count, and the caption's tally with it")
        # q is bolded where the call is decisive, so the 2/4/25 split in the body can be counted off
        # the table by eye rather than taken on trust.
        q = f"{e['q']:.3f}" if e["q"] == e["q"] else "--"
        q = rf"\textbf{{{q}}}" if c["call"] != "inconclusive" else q
        L.append(f"{cm._display(c['cell'], c.get('n_train'))}{mark} & {e['n']} & {fmt(e['mean'])} & "
                 f"{ci} & {e['t_p']:.3f} & {q} & {e['p']:.3f} & {d} \\\\")
    L.append(r"\midrule")
    # Eight columns: name, N, mean, CI, p_t, q, p_exact, count. The blanks are positional, and a
    # miscount here slides the cluster aggregate's count under p_exact, which LaTeX renders without
    # complaint -- so the count is derived from the header list rather than typed twice.
    a = agg["value"]["d_enc"]
    L.append(r"\multicolumn{8}{l}{\emph{Mean of cell means, $95\%$ CI from resampling "
             r"(backbone, dataset) clusters:}} \\")
    L.append(rf"\quad {a['n_cells']} value-cells / {a['n_clusters']} clusters & & {fmt(a['mean'])} "
             rf"& [{fmt(a['lo'])}, {fmt(a['hi'])}] & & & & {a['pos']}/{a['n_cells']} \\")
    b = agg["all"]["d_enc"]
    L.append(rf"\quad all {b['n_cells']} cells / {b['n_clusters']} clusters & & {fmt(b['mean'])} & "
             rf"[{fmt(b['lo'])}, {fmt(b['hi'])}] & & & & {b['pos']}/{b['n_cells']} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")
    return len(vcells)


def emit_power_tex(cells, path=OUT_POWER_TEX, gate_threshold=gac.GATE_THRESHOLD):
    """The screen's survivors with what their own seed budget could have detected.

    Restricted to the gate-passing cells on purpose: those are the only cells about which the paper
    makes a preservation recommendation, so those are the cells where "the interval is wide" has to be
    turned into a number rather than left as a hedge. The unresolvable ones are the finding here, not
    an embarrassment to be softened -- a cell needing 71 paired seeds is a cell this design cannot
    settle, and saying so with the number is a stronger statement than a hedge.
    """
    gp = sorted([c for c in cells if c["gate"] is not None and c["gate"] >= gate_threshold
                 and c.get("d_enc")], key=lambda c: -c["gate"])
    # Unresolvability is judged on the ADJUSTED budget, because the adjusted budget is what a decisive
    # call costs; judging it on the uncorrected one would call two more cells resolvable than are.
    unres = [c for c in gp if c["d_enc"]["n_needed_adj"] is None
             or c["d_enc"]["n_needed_adj"] > N_FEASIBLE]
    L = ["% GENERATED by scripts/paired_inference.py -- do not edit by hand.",
         "% Every number is derived from the same per-seed lists as tables/paired_inference.tex.",
         r"\begin{table}[t]", r"\centering",
         r"\caption{\textbf{What each surviving cell's own seed budget could detect.}",
         rf"The {len(gp)} cells clearing $\Vb{{\text{{ridge}}}}{{\geq}}{gate_threshold:.2f}$, with the",
         r"minimum detectable effect implied by their own dispersion: MDE $=",
         r"t_{1-\alpha/2,n-1}\,\text{sd}/\sqrt{n}$, the smallest $|\denc|$ a two-sided paired $t$",
         r"would resolve at that $n$.  $n^\star$ is the paired seed count at which the \emph{observed}",
         r"effect would resolve, holding the observed sd fixed; it is therefore a post-hoc estimate",
         r"and not a design calculation.",
         r"\textbf{Two columns, because a single test and a decisive call cost different budgets.}",
         r"$n^\star$ is computed at an uncorrected $\alpha{=}0.05$; $n^\star_\text{adj}$ is computed",
         rf"at $\alpha/m$ with $m{{=}}{gp[0]['d_enc']['m_tests']}$ tests, which bounds the",
         r"Benjamini--Hochberg threshold the call in Table~\ref{tab:paired_inference} actually uses",
         r"from below and so bounds the budget from above.  Reporting only $n^\star$ would say",
         r"Moirai-Base/ETTh2 $h{=}192$ is already resolved at $n{=}3$, which its own $q{=}0.018$",
         r"happens to agree with---and would say the same of $h{=}96$, whose $q$ is $0.071$.",
         rf"\textbf{{The {len(unres)} starred rows are the point of the table}}: they need more",
         rf"than our feasibility cap of {N_FEASIBLE} paired seeds ($\approx$20 GPU-hours per cell at",
         r"two fine-tunes per seed), so we report them as \emph{unresolvable under this design}, with",
         r"the number, rather than as effects that are absent.}",
         r"\label{tab:power_mde}", r"\small",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{@{}l r r r r r r r@{}}", r"\toprule",
         r"Cell & $\Vb{\text{ridge}}$ & $n$ & $\denc$ & sd & MDE & $n^\star$ & "
         r"$n^\star_\text{adj}$ \\",
         r"\midrule"]

    def nstar(v, star):
        """The star marks the DECISION column only. Starring both would leave a reader counting six
        marks against a caption that says four, and the count in the caption is the claim."""
        s = f"$>${N_SEARCH_CAP}" if v is None else str(v)
        return s + r"$^\star$" if star and (v is None or v > N_FEASIBLE) else s

    for c in gp:
        e = c["d_enc"]
        L.append(f"{cm._display(c['cell'], c.get('n_train'))} & {fmt(c['gate'], 3)} & {e['n']} & "
                 f"{fmt(e['mean'])} & {fmt(e['sd'], 2, signed=False)} & "
                 f"{fmt(e['mde'], 2, signed=False)} & {nstar(e['n_needed'], False)} & "
                 f"{nstar(e['n_needed_adj'], True)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")
    return len(gp), len(unres)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tex", action="store_true")
    ap.add_argument("--gate", default=str(GATE_BASELINES),
                    help="Ladder file defining a value-cell. Default is the selection-split "
                         "ladder; results/gate_baselines.json is the retrospective one.")
    a = ap.parse_args()

    rows = cm.build_rows()
    cells, vc = analyse(rows, a.boot, a.seed, a.gate)
    all_refs = {c["ref"] for c in cells}
    agg = {"value": {k: cluster_aggregate(cells, k, set(vc), a.boot, a.seed) for k, _ in CONTRASTS},
           "all": {k: cluster_aggregate(cells, k, all_refs, a.boot, a.seed) for k, _ in CONTRASTS}}

    OUT_JSON.write_text(json.dumps(dict(cells=cells, aggregate=agg, value_cells=vc,
                                        n_boot=a.boot, seed=a.seed), indent=1) + "\n")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}  ({len(cells)} cells, {len(vc)} value-cells)")

    print(f"\n{'cell':34s} {'N':>2s} {'D_enc':>8s} {'SEM':>6s} {'95% t-CI':>18s} "
          f"{'p_t':>6s} {'p_ex':>6s} {'floor':>6s} {'d':>6s} sign")
    for c in sorted(cells, key=lambda c: (not c["is_value_cell"], c["cell"])):
        e = c["d_enc"]
        if not e:
            continue
        flag = "*" if c["is_value_cell"] else " "
        print(f"{flag}{c['cell']:33s} {c['seeds']:2d} {e['mean']:+8.2f} {e['sem']:6.2f} "
              f"[{e['lo']:+7.2f},{e['hi']:+7.2f}] {e['t_p']:6.3f} {e['p']:6.3f} "
              f"{e['p_floor']:6.3f} "
              f"{(e['d'] if e['d'] is not None else float('nan')):+6.2f} {e['pos']}/{e['n']}")

    hist = collections.Counter(c["d_enc"]["n"] for c in cells if c["d_enc"])
    print(f"\nPaired-N histogram for Delta_enc: {dict(sorted(hist.items()))} "
          f"(total {sum(hist.values())} cells)")

    print("\nWhat can the ASSUMPTION-FREE test resolve at alpha=0.05?")
    resolvable = [c for c in cells if c["d_enc"] and c["d_enc"]["p_floor"] <= 0.05]
    sig = [c for c in resolvable if c["d_enc"]["p"] < 0.05]
    print(f"  {len(resolvable)} of {len(cells)} cells have enough seeds (p_floor <= 0.05); "
          f"{len(sig)} of those reach p < 0.05")
    for c in sig:
        print(f"    {c['cell']:34s} D_enc={c['d_enc']['mean']:+.2f}  p={c['d_enc']['p']:.4f}")

    print("\nWhere do the two tests DISAGREE at alpha=0.05?  (t rejects, exact cannot)")
    dis = [c for c in cells if c["d_enc"] and c["d_enc"]["t_p"] < 0.05 and c["d_enc"]["p"] >= 0.05]
    print(f"  {len(dis)} cells. On these the reading rests on the normality assumption:")
    for c in dis:
        e = c["d_enc"]
        print(f"    {c['cell']:34s} N={c['seeds']} D_enc={e['mean']:+7.2f} "
              f"p_t={e['t_p']:.4f} p_exact={e['p']:.3f} (floor {e['p_floor']:.3f})")
    both = [c for c in cells if c["d_enc"] and c["d_enc"]["t_p"] < 0.05 and c["d_enc"]["p"] < 0.05]
    print(f"  {len(both)} cells are significant under BOTH: "
          f"{', '.join(c['cell'] for c in both) or 'none'}")

    print("\nAggregates (mean of cell means, clustered on (backbone, dataset)):")
    for scope in ("value", "all"):
        for k, lab in CONTRASTS:
            g = agg[scope][k]
            if g:
                print(f"  {scope:6s} {k:9s} n={g['n_cells']:2d} clusters={g['n_clusters']:2d} "
                      f"mean={g['mean']:+7.2f}  CI [{g['lo']:+7.2f},{g['hi']:+7.2f}]  "
                      f"pos={g['pos']} neg={g['neg']}")

    unpaired = [c for c in cells if not c["ft_paired"]]
    print(f"\nDelta_FT/Delta_frozen are NOT seed-paired on {len(unpaired)} cells (shared zero-shot "
          f"reference); their intervals are widened by the reference's own SEM:")
    sems = [c["zs_sem"] for c in unpaired if c["zs_sem"] is not None]
    if sems:
        print(f"  reference SEM median {st.median(sems):.2f}, max {max(sems):.2f} "
              f"(reference measured on {min(c['zs_seeds'] or 0 for c in unpaired)}"
              f"-{max(c['zs_seeds'] or 0 for c in unpaired)} condition-A seeds)")
    one_seed = [c for c in unpaired if c["zs_seeds"] == 1]
    print(f"  on {len(one_seed)} of them the reference ran ONE seed, so its error is UNESTIMATED "
          f"rather than zero and nothing is propagated: "
          f"{', '.join(c['cell'] for c in one_seed) or 'none'}")
    widened = [c for c in cells if c["d_ft"] and c["d_ft"]["se_zs"] > c["d_ft"]["se_within"] + 1e-12]
    print(f"  {len(widened)} cells actually widened (the rest have zs_sem 0 or are seed-paired)")

    print(f"\nBH-ADJUSTED CALL on Delta_enc across all {len(cells)} cells (alpha={ALPHA}):")
    counts = collections.Counter(c["call"] for c in cells)
    print(f"  freezing decisively better: {counts['freeze']}    "
          f"adaptation decisively better: {counts['adapt']}    "
          f"inconclusive: {counts['inconclusive']}")
    for want, lab in (("freeze", "FREEZE decisive"), ("adapt", "ADAPT decisive")):
        for c in sorted((c for c in cells if c["call"] == want), key=lambda c: c["d_enc"]["q"]):
            e = c["d_enc"]
            print(f"    {lab:15s} {c['cell']:34s} gate={c['gate']:+7.3f} "
                  f"{'PASS' if c['gate'] >= gac.GATE_THRESHOLD else 'fail'}  "
                  f"D_enc={e['mean']:+7.2f}  q={e['q']:.4f}")
    # The ordering the reviewer's objection turns on: if every decisively-freeze-better cell is a cell
    # the screen REJECTS, then the screen is not selecting the cells where preservation pays, and that
    # is a statement about the screen rather than about the seeds.
    frz_pass = [c for c in cells if c["call"] == "freeze" and c["gate"] >= gac.GATE_THRESHOLD]
    print(f"  of the {counts['freeze']} freeze-decisive cells, {len(frz_pass)} clear the gate")

    print("\nDEGRADATION under both criteria (the published one is kept, not replaced):")
    u = [c["cell"] for c in cells if c["degrades_unanimous"]]
    ci = [c["cell"] for c in cells if c["degrades_ci"]]
    print(f"  three-clause unanimity (published): {len(u)} cells {u or ''}")
    print(f"  three-clause with interval clauses: {len(ci)} cells {ci or ''}")
    # The published definition is cell_matrix.degradation_cells. Two spellings of one rule drift, so
    # this asserts rather than assumes. Its printing is suppressed: it is a diagnostic of that module,
    # not output of this one.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        of_record = {r["cell"] for r in cm.degradation_cells(rows)}
    if of_record != set(u):
        sys.exit(f"degrades_unanimous disagrees with cell_matrix.degradation_cells, which is the "
                 f"definition of record: {sorted(of_record)} vs {sorted(u)}")
    print(f"  cross-checked against cell_matrix.degradation_cells: agree ({len(of_record)} cells)")

    print(f"\nPOWER on the gate-passing cells (MDE at each cell's own n and sd; "
          f"feasibility cap {N_FEASIBLE} paired seeds):")
    gp = sorted((c for c in cells if c["gate"] >= gac.GATE_THRESHOLD and c["d_enc"]),
                key=lambda c: -c["gate"])
    def shw(v):
        return f">{N_SEARCH_CAP}" if v is None else str(v)

    for c in gp:
        e = c["d_enc"]
        nn = e["n_needed_adj"]
        tag = ("UNRESOLVABLE" if nn is None or nn > N_FEASIBLE else "resolvable")
        print(f"  {c['cell']:34s} n={e['n']:2d} D_enc={e['mean']:+7.2f} sd={e['sd']:6.2f} "
              f"MDE={e['mde']:6.2f}/{e['mde_adj']:6.2f}adj  "
              f"n*={shw(e['n_needed']):>4}  n*adj={shw(nn):>4} {tag}")
    # The top-up list, computed rather than chosen. A cell earns a top-up only if it is (a) not
    # already decisive -- adding seeds to a cell whose call is settled buys nothing -- and (b) inside
    # the cap on the ADJUSTED budget, which is the budget a decisive call costs. Choosing the list any
    # other way is choosing which cells get another chance after seeing their p-values.
    already = [c["cell"] for c in gp if c["d_enc"]["q"] < ALPHA]
    print(f"  {len(already)} already decisive at q<{ALPHA}, so not topped up: "
          f"{', '.join(already) or 'none'}")
    for cap in sorted({N_FEASIBLE, 2 * N_FEASIBLE}):
        topup = [(c["cell"], e["n"], e["n_needed_adj"]) for c in gp
                 if (e := c["d_enc"])["q"] >= ALPHA
                 and e["n_needed_adj"] and cap >= e["n_needed_adj"] > e["n"]]
        extra = sum(nn - n for _, n, nn in topup)
        print(f"  at a cap of {cap:3d} paired seeds: {len(topup)} cell(s) reachable, "
              f"+{extra} paired seeds = {2 * extra} fine-tunes"
              + (("  [" + ", ".join(f"{c} {n}->{nn}" for c, n, nn in topup) + "]") if topup else ""))

    if not a.no_tex:
        n = emit_tex(cells, agg)
        print(f"\nwrote {OUT_TEX.relative_to(ROOT)}  ({n} value-cells marked)")
        ngp, nunres = emit_power_tex(cells)
        print(f"wrote {OUT_POWER_TEX.relative_to(ROOT)}  ({ngp} gate-passing cells, "
              f"{nunres} unresolvable within {N_FEASIBLE} paired seeds)")


if __name__ == "__main__":
    main()
