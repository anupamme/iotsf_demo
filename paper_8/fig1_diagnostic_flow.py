#!/usr/bin/env python3
"""
Fig. 1 -- the screen, and why nothing survives it.

(a) The chain as a FUNNEL, and its boxes are the paper's FOUR LAYERS, numbered as Section 1 numbers
    them so the reader can hold one hierarchy rather than two. Every box carries the count of cells
    still standing, because the paper's result is where the funnel empties, not what the chain looks
    like. 32 cells are screened; 31 of them have a selection-split gate, seven of those clear the
    inclusion criterion, and 0 meet the three-clause degradation definition. The last box is
    highlighted because that zero is the paper's claim.

    LAYER 2 IS DRAWN DASHED BECAUSE IT DOES NOT FILTER. CKA is a measurement taken on every cell, not
    a gate any cell can fail, and an earlier draft of this panel omitted it for exactly that reason --
    at which point the figure no longer contained the quantity the title is about. Drawing it solid
    would be worse than omitting it: the funnel's counts would then have to drop across it, and they
    do not. Its sub-line and its tag both say so, and the count it carries is all 31.

    WATCH THE DENOMINATOR. The gate box divides by 31, not by the 32 of the box above it:
    TimesFM/Electricity is screened but has no paired frozen-encoder run and hence no selection-split
    reference, so it can neither pass nor fail. The two numbers differing by one is deliberate and the
    body says so; 25-of-32 would be the arithmetic of a cell that was never scored.

(b) Every intervention cell as (CKA, held-out B-D) -- the paper's one sentence, drawn.

    THE X-AXIS HAS BEEN CKA, THEN THE GATE, AND IS NOW CKA AGAIN, and the round trip is worth
    recording because each move was right at the time. It was CKA while the claim was "degradation
    cells are the least drifted" and the panel drew a degradation class. The corrected gate emptied
    that class, so drawing it became indefensible and the axis moved to the gate, whose point estimate
    runs the WRONG WAY within Moirai (rho(gate, Delta_encoder) < 0: higher measured pre-trained value
    goes with encoder adaptation helping MORE). It is CKA again because the paper's title and abstract
    now state the CKA-to-outcome relation as the finding, and the figure a reader meets first has to be
    the figure of the claim. What is NOT drawn either time is a degradation class: there is none, panel
    (a)'s terminal box says so, and no marker here encodes one.
    Both correlations are still printed, the gate one included, so nothing is lost by the axis choice;
    the gate-versus-outcome scatter itself is Appendix Figure \ref{fig:gatescatter}.

    THE GATE IS SCORED ON THE SELECTION SPLIT, the y-axis on the held-out split. That is the whole
    reason the fill convention here (filled = clears the gate) is admissible: an earlier version scored
    both on the same windows, which made cell inclusion a decision taken after seeing the outcome.
    cell_matrix.GATE_SPLIT controls it, and flipping it back changes which markers are filled but not
    the sign of either correlation.

    THE FOUR LABELLED POINTS ARE THE PAPER'S RUNNING EXAMPLE, introduced in S1 and returned to in S6:
    Moirai-Base drifts LEAST on ETTh1 and is the arm's only degrading pair, and drifts MOST on ETTh2
    and improves most. They are selected by cell ref and asserted below, not eyeballed off the plot.

    THE INTERVALS ARE CLUSTERED AND THEY ARE WIDE, and the panel prints them rather than a p-value.
    An earlier version of this docstring asserted "rho = -0.52 (CI [-0.84, -0.03], excludes zero)" and
    the panel printed a matching cell-level p. Both were cell-level, and the body replaced cell-level
    intervals with intervals clustered on (backbone, dataset) in bb718d4 -- 22 Moirai cells reuse only
    6 series, so treating them as 22 independent observations understates the uncertainty. Clustered,
    neither interval excludes zero. No number is quoted in this docstring any more: the values are
    printed by the script at draw time, and a docstring that repeats them is one more place for them
    to go stale, which is exactly how the -0.52 survived the correction that invalidated it.

    History, so the change is auditable: at 23 cells this panel read "degradation cells are the least
    drifted"; the prospective arm broke that; the corrected gate then removed the degradation set
    entirely. The claim has weakened three times and the figure has followed it each time.

An earlier version of this figure was a 15-bar chart with hand-typed values, which is how it came to
silently drop a cell and aggregate four others without saying so. Every point below is read from
cell_matrix.build_rows(), the same function behind the tables and the statistics.

SIZED FOR 1:1 PLACEMENT. The ICLR textwidth is 5.5in, so this figure is authored at 5.5in and
included at width=\\linewidth. Do not shrink it in LaTeX -- every fontsize below is the size it
will actually print at.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The palette, the markers and the row loader are shared with the two appendix figures so that green
# means the same thing in all three; figstyle.py records why they are not defined here any more.
from figstyle import (  # noqa: E402
    ADAPT, ADAPT_F, DAMAGE, DAMAGE_F, GATE, GRID, HELDOUT, HELDOUT_F,
    INK_MUTED, INK_PRIMARY, INK_SECONDARY, MARKER, N_SCORED, N_SCREENED, SURFACE,
    backbone, intervention_rows, save,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import cluster_keys  # noqa: E402

# ================================================================ data
ROWS = intervention_rows()


def is_degradation(r):
    """The paper's own definition, applied here so the figure cannot disagree with the text."""
    return (r["gate"] is not None and r["gate"] >= GATE
            and r.get("forg_b") is not None and r["forg_b"] > 0 and r["forg_d"] < 0
            and r["forg_b_pos"] == r["seeds"] and r["forg_d_neg"] == r["seeds"])


# The paper's central negative result, asserted rather than trusted. If a future run produces a
# degradation cell this fails, and it SHOULD -- panel (a)'s terminal box and most of the body would
# then be wrong. The earlier count of 8 was an artefact of an unfitted linear denominator.
assert sum(is_degradation(r) for r in ROWS) == 0, \
    "a cell now meets the degradation definition -- panel (a) and the body's claim both need redoing"

# The gate follows cell_matrix.GATE_SPLIT, which is the SELECTION split: the windows the gate is
# scored on are disjoint from the held-out windows bd_test is scored on, so panel (a)'s funnel is a
# genuine screen rather than a decision made after seeing the outcome. Seven pass, not the five of
# the retrospective variant, and the two extra are ETTm2 -- which is why panel (b) is no longer
# describable as an ETTh2-only story.
PASSING = [r for r in ROWS if r["gate"] >= GATE]
assert len(PASSING) == 7, f"expected 7 gate-passing intervention cells, got {len(PASSING)}"
assert len({r["cell"].split("/")[-1].split()[0] for r in PASSING}) >= 1
# Of the seven, the ones full fine-tuning IMPROVES -- clause (ii) fails outright for these.
# UNANIMITY, NOT THE MEAN. Five of the seven have a negative mean forg_B, but one of those five
# (Moirai-Small/ETTm2 h192, -1.10%) has 3 of 5 seeds pointing the other way, so its mean improvement
# is outlier-driven and it is not a cell where "fine-tuning improves the checkpoint" is a safe thing
# to say. The body's clause-(ii) count is the unanimous one, and this figure has to use the same rule
# or the funnel and the text disagree by one cell.
IMPROVED = [r for r in PASSING if r["forg_b"] < 0 and r["forg_b_pos"] == 0]
assert len(IMPROVED) == 4, f"expected 4 unanimously improved survivors, got {len(IMPROVED)}"
assert len([r for r in PASSING if r["forg_b"] < 0]) == 5, \
    "the mean-improvement count moved; the body distinguishes it from the unanimous one"
_imp = sorted(abs(r["forg_b"]) for r in IMPROVED)

# Sign reversals between the selection split and the disjoint test split -- a measurement
# precondition, so panel (a) carries it. Counted here rather than typed, and the one-directionality
# that licenses the word "bias" is asserted, not assumed.
REVERSALS = [r for r in ROWS
             if r.get("bd_val") is not None and (r["bd_val"] > 0) != (r["bd_test"] > 0)]
assert len(REVERSALS) == 4, f"expected 4 sign reversals, got {len(REVERSALS)}"
assert all(r["bd_val"] > 0 > r["bd_test"] for r in REVERSALS), \
    "reversals are no longer all in the direction that flatters the frozen encoder"


# Within-backbone correlations. Moirai is the only arm with enough cells to ask, and pooling across
# backbones is what S5.2 forbids, so the within-Moirai number is the one this panel leads with.
#
# ROUND 8 REVERSES HALF OF THAT, AND THE PREDECESSOR'S REASONING IS LEFT STANDING ABOVE BECAUSE IT WAS
# RIGHT ABOUT THE THING IT WAS ABOUT. Printing ONLY the within-Moirai rho was defensible as long as the
# question was which number the paper reports. It is not defensible against the way the panel is
# actually read: a reviewer who knows the pooled figure is +0.567 and sees only +0.168 here reads the
# within-backbone null as special pleading, and told us so in those terms. The pooled figure is not
# suppressible by omission -- S6 states it -- so omitting it from the figure costs the credibility of
# the panel without costing the reader the number. What the panel can do, and prose cannot, is show WHY
# the two differ: the Chronos squares sit bottom-left (most drifted, adaptation helps most) and the
# Moirai circles upper-right, so the pooled association IS the gap between two backbones and the eye
# gets that in one look. So both are printed, the pooled one labelled as a between-backbone contrast
# rather than as an estimand. This is the Simpson's-paradox resolution drawn instead of asserted.
#
# INTERVALS, NOT p-VALUES, AND CLUSTERED ONES. Until 2026-09-18 this panel printed the cell-level
# Spearman p-value ("p=0.014"), which contradicted the body twice over: the body reports the
# DIRECTION and not the interval on this axis, and it reports cross-cell uncertainty clustered on
# (backbone, dataset) because the 22 Moirai cells reuse 6 series and a cell-level interval is too
# narrow. A figure quoting a cell-level p beside a body quoting a clustered CI is the paper
# disagreeing with itself in the one place a reader looks first. The clustered interval is wide, and
# printing it wide is the point.
_M = [r for r in ROWS if backbone(r) == "Moirai"]
RHO_GATE = stats.spearmanr([r["gate"] for r in _M], [r["bd_test"] for r in _M])
RHO_CKA = stats.spearmanr([r["cka"] for r in _M], [r["bd_test"] for r in _M])
_MCL = cluster_keys.clusters_for([r["cell"] for r in _M])
CI_GATE = cluster_keys.cluster_bootstrap_spearman([r["gate"] for r in _M],
                                                 [r["bd_test"] for r in _M], _MCL)
CI_CKA = cluster_keys.cluster_bootstrap_spearman([r["cka"] for r in _M],
                                                [r["bd_test"] for r in _M], _MCL)
assert RHO_GATE.statistic < 0, \
    "the gate is no longer anti-predictive; the docstring's (b) block and S5.2 both say it is"

# The pooled figure, READ FROM value_axis.json rather than recomputed for printing, and recomputed
# anyway to check it. Two directions of failure, two guards:
#   - the figure printing a number S6 does not quote. Avoided by taking the printed value from the same
#     artifact check_paper_numbers.py registers S6's numerals against (:817-823), so the panel, the
#     prose and the checker cannot hold three versions of one correlation.
#   - value_axis.json going stale against the rows this panel actually draws. Avoided by recomputing
#     rho here from ROWS and requiring agreement, which is the cross-emitter assert the checker already
#     makes between value_axis.json and clustered_inference.json (:845). The bootstrap is seeded
#     (cluster_keys.cluster_bootstrap_spearman defaults to default_rng(0)), so the interval is
#     reproducible too and is checked on the same footing as the point estimate.
_VA = json.loads((Path(__file__).resolve().parent.parent
                  / "results/value_axis.json").read_text())["correlations"]["cka_vs_denc_all"]
POOLED = (_VA["rho"], _VA["lo"], _VA["hi"], _VA["n"], _VA["clusters"])
_pooled_recomputed = cluster_keys.cluster_bootstrap_spearman(
    [r["cka"] for r in ROWS], [r["bd_test"] for r in ROWS],
    cluster_keys.clusters_for([r["cell"] for r in ROWS]))
assert _VA["n"] == len(ROWS) and _VA["clusters"] == _pooled_recomputed[4], (
    f"value_axis.json pools {_VA['n']} cells in {_VA['clusters']} clusters; this panel draws "
    f"{len(ROWS)} in {_pooled_recomputed[4]}")
assert all(abs(_VA[k] - _pooled_recomputed[i]) < 1e-12 for i, k in enumerate(("rho", "lo", "hi"))), (
    f"the pooled CKA correlation in value_axis.json ({_VA['rho']:+.4f} "
    f"[{_VA['lo']:+.4f}, {_VA['hi']:+.4f}]) disagrees with the same statistic recomputed from the "
    f"rows this panel draws ({_pooled_recomputed[0]:+.4f} [{_pooled_recomputed[1]:+.4f}, "
    f"{_pooled_recomputed[2]:+.4f}]); S6 quotes the artifact and the figure would draw the rows")
assert POOLED[0] > RHO_CKA.statistic, (
    "the pooled correlation is no longer larger than the within-Moirai one, so the panel's "
    "'between backbones, not within one' label is describing the opposite of the data")

# ---- panel (b)'s claim, computed rather than eyeballed off the plot ----------------------------
# The panel title says no cut on CKA separates the cells encoder adaptation helped from the cells it
# hurt. That is a statement about the BEST such cut, so find it: over every threshold, the best rule
# mislabels at least MISCUT of the 31 cells. Drawing a claim this cheap to check and not checking it is
# how the "-0.52 excludes zero" of the docstring above survived.
#
# BOTH ORIENTATIONS, and that is not pedantry. "Below t, adaptation helps" is the direction a
# drift-is-damage heuristic implies, but the positive rho points the other way, so scoring only one
# direction would leave the caption's claim resting on a choice a reader could reasonably reverse. The
# minimum over both is the best case ANY single-threshold rule on CKA achieves. check_paper_numbers.py
# recomputes this same minimum for the caption's number, so the two cannot drift apart.
_S = sorted(ROWS, key=lambda r: r["cka"])
_sg = [1 if r["bd_test"] > 0 else 0 for r in _S]
MISCUT = min(min(sum(_sg[:k]) + sum(1 - s for s in _sg[k:]),
                 sum(1 - s for s in _sg[:k]) + sum(_sg[k:]))
             for k in range(len(_sg) + 1))
assert MISCUT >= 4, f"a CKA threshold now separates the two regimes to within {MISCUT} cells"

# ---- the running example of S1, returned to in S6 -------------------------------------------------
# Selected by ref, never by the display string: cell_matrix's display strings are formatted two
# different ways across the arms, so a substring match on them silently drops cells.
_E1 = ("base_ETTh1_h96", "base_ETTh1_h192")
_E2 = ("base_ETTh2_h96", "base_ETTh2_h192")
RUN = {r["ref"]: r for r in ROWS if r["ref"] in _E1 + _E2}
assert len(RUN) == 4, f"expected the 4 Moirai-Base running-example cells, got {sorted(RUN)}"
_BASE = [r for r in ROWS if r["ref"].startswith("base_")]
# The three things S1 paragraph 2 asserts about this pair, in the order it asserts them.
assert sorted(_BASE, key=lambda r: -r["cka"])[:2] == [RUN[_E1[0]], RUN[_E1[1]]], \
    "the ETTh1 pair is no longer Moirai-Base's two LEAST drifted cells"
assert sorted(r["ref"] for r in _BASE
              if r["forg_b"] > 0 and r["forg_b_pos"] == r["seeds"]) == sorted(_E1), \
    "the ETTh1 pair is no longer Moirai-Base's only unanimously degrading pair"
assert sorted(_BASE, key=lambda r: r["forg_b"])[:2] == [RUN[_E2[1]], RUN[_E2[0]]], \
    "the ETTh2 pair no longer improves most within the Moirai-Base arm"
# The clustered rho must be the same point estimate the body quotes; cluster_bootstrap_spearman
# returns it from the unresampled data, so a mismatch here means the two are reading different rows.
assert abs(CI_GATE[0] - RHO_GATE.statistic) < 1e-12 and abs(CI_CKA[0] - RHO_CKA.statistic) < 1e-12, \
    "clustered and cell-level point estimates disagree -- the two are not reading the same cells"

fig = plt.figure(figsize=(5.5, 1.62), dpi=150)
fig.patch.set_facecolor(SURFACE)
gs = GridSpec(1, 2, width_ratios=[0.86, 1.14], left=0.005, right=0.985,
              top=0.99, bottom=0.005, wspace=0.30)

# ================================================================ (a) the funnel
axl = fig.add_subplot(gs[0])
axl.set_facecolor(SURFACE)
axl.set_xlim(0, 100); axl.set_ylim(0, 100); axl.axis("off")

axl.text(0, 99.5, f"(a) {N_SCREENED} cells enter, none meets the definition",
         fontsize=6.4, color=INK_PRIMARY, fontweight="bold", va="top")


def box(cy, h, title, sub, fc, ec, bold_sub=False, ls="solid"):
    axl.add_patch(FancyBboxPatch((2, cy - h / 2), 62, h,
                                 boxstyle="round,pad=0.6,rounding_size=2", linestyle=ls,
                                 linewidth=0.8, edgecolor=ec, facecolor=fc, zorder=3))
    axl.text(33, cy + (2.2 if sub else 0), title, fontsize=5.9, color=INK_PRIMARY,
             ha="center", va="center", fontweight="bold", zorder=4)
    if sub:
        axl.text(33, cy - 4.1, sub, fontsize=5.1,
                 color=ec if bold_sub else INK_SECONDARY,
                 ha="center", va="center", zorder=4,
                 fontweight="bold" if bold_sub else "normal")


def arrow(y0, y1):
    axl.add_patch(FancyArrowPatch((33, y0), (33, y1), arrowstyle="-|>", mutation_scale=5,
                                  linewidth=0.7, color=INK_MUTED, zorder=2))


def tag(cy, text, colour):
    axl.text(67, cy, text, fontsize=4.8, color=colour, ha="left", va="center",
             style="italic", linespacing=1.12, zorder=4)


box(86, 12, f"{N_SCREENED} screened cells", "6 datasets, 3 backbones", "#f2f1ec", INK_MUTED)
arrow(79.5, 72)
box(65, 13, "1. worth preserving?",
    rf"{len(PASSING)} of {N_SCORED} clear $V_{{\rm ridge}}\geq 0.20$", "#f2f1ec", INK_MUTED)
arrow(58.2, 52.2)
box(45.5, 13, "2. representation changed?",
    f"CKA, measured on all {N_SCORED}", SURFACE, INK_MUTED, ls=(0, (2.2, 1.6)))
arrow(38.8, 32.8)
box(26, 13, "3. did adaptation matter?", "fine-tune (B) vs freeze (D)", ADAPT_F, ADAPT)
arrow(19.3, 13)
# "all 3 clauses", not "every definition": Appendix \ref{app:defsensitivity} finds ONE cell under a
# weaker rung crossed with the lower threshold, so a box claiming the zero is definition-free would
# overstate what the sensitivity analysis found. The limitations section grants that in words.
box(6.5, 12, "4. capability destroyed?",
    f"{sum(is_degradation(r) for r in ROWS)} cells meet all 3 clauses",
    DAMAGE_F, DAMAGE, bold_sub=True)

# The tags carry what the box labels cannot: WHY 24 cells drop out (no advantage over a baseline fit on
# their own data, which is the screen's whole content), that layer 2 filters nothing, that layer 3 is
# scored off the selection windows, and that the survivors fail in the strongest possible direction --
# the thing that was supposed to damage them improves them.
tag(65, "measured against a\nbaseline fit on the\ncell's own data", INK_SECONDARY)
tag(45.5, "a measurement, not a\nfilter: every cell\ncontinues", INK_SECONDARY)
tag(26, f"held-out windows, not\nthe selection windows:\n"
        f"{len(REVERSALS)} of {len(ROWS)} reverse sign", HELDOUT)
tag(6.5, f"{len(IMPROVED)} of the {len(PASSING)} survivors\nare "
         f"{_imp[0]:.0f}$-${_imp[-1]:.0f}% better\nfine-tuned", DAMAGE)

# ================================================================ (b) the scatter
axr = fig.add_subplot(gs[1])
axr.set_facecolor(SURFACE)
for sp in ("top", "right"):
    axr.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    axr.spines[sp].set_color(INK_MUTED); axr.spines[sp].set_linewidth(0.6)

axr.axhline(0, color=INK_MUTED, lw=0.7, zorder=1)
axr.grid(axis="y", color=GRID, lw=0.5, zorder=0)
axr.set_axisbelow(True)

# NO SHADED REGION AND NO VERTICAL RULE. The gate axis had both, because 0.20 is a real boundary there.
# On the CKA axis there is no boundary to draw -- that absence IS the panel's claim -- and a decorative
# band would read as one. The untrained-encoder CKA calibration is architecture-specific (S5.2) and so
# cannot be a single rule across three backbones either.
for r in ROWS:
    passes = r["gate"] >= GATE
    helps = r["bd_test"] < 0
    ec = ADAPT if helps else DAMAGE
    fc = (ADAPT_F if helps else DAMAGE_F) if passes else "none"
    axr.scatter(r["cka"], r["bd_test"], s=26, marker=MARKER[backbone(r)],
                facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=4)

axr.set_xlim(0.03, 1.03)
axr.set_ylim(-46, 69)
axr.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
# "1 = unchanged" rather than "higher = less drift": the reader has to know which end of this axis the
# forgetting literature would call catastrophic, and the number 1 says it without a second clause.
axr.set_xlabel("CKA, pre-trained vs fine-tuned encoder   (1 = representation unchanged)",
               fontsize=5.9, color=INK_SECONDARY, labelpad=1.5)
# Review S18: the panel has to be readable without S3, so the y axis carries the estimand itself
# rather than the shorthand "B-D". Written out as in Equation 1 -- per-seed mean of the normalised
# difference -- and the sign is glossed in the panel by the two labels below rather than in the caption.
axr.set_ylabel(r"$\Delta_{\mathrm{enc}} = 100\,(L_{\mathrm{B}}-L_{\mathrm{D}})/L_{\mathrm{ZS}}$  (pp)",
               fontsize=5.9, color=INK_SECONDARY, labelpad=1.5)
axr.tick_params(labelsize=5.4, colors=INK_SECONDARY, length=2, width=0.6, pad=1.2)

axr.text(-0.10, 1.03, f"(b) no cut on CKA separates them (best misplaces {MISCUT})",
         transform=axr.transAxes, fontsize=6.4, color=INK_PRIMARY, fontweight="bold", va="bottom")
axr.text(0.01, 0.955, "positive: freezing better, encoder adaptation hurt",
         transform=axr.transAxes,
         fontsize=5.3, color=DAMAGE, ha="left", va="center")
# Not the bottom-left corner, where it lands on the Chronos cells: the four most-drifted cells in the
# matrix are all down there, which is the same crowding that makes the panel's point.
axr.text(0.22, -44, "adaptation helps", fontsize=5.3, color=ADAPT, ha="left", va="center")
# The gate rho stays printed even though the gate is off the axis: it is the number S5.2 reports, and
# dropping it with the axis would make this panel quieter than the body it illustrates.
axr.text(0.055, 14,
         f"Moirai only ($n$={len(_M)}, {CI_CKA[4]} clusters):\n"
         rf"$\rho_{{\rm CKA}}={CI_CKA[0]:+.2f}$ [{CI_CKA[1]:+.2f}, {CI_CKA[2]:+.2f}]" "\n"
         rf"$\rho_{{\rm gate}}={CI_GATE[0]:.2f}$ [{CI_GATE[1]:+.2f}, {CI_GATE[2]:+.2f}]",
         fontsize=4.8, color=INK_SECONDARY, ha="left", va="center", linespacing=1.25)
# The pooled figure, ABOVE the within-backbone block and labelled with what it is. The two rhos have to
# be read against each other, so they cannot be separated into figure and caption.
#
# THREE LINES, NOT TWO, AND THAT IS THE WHOLE PLACEMENT CONSTRAINT. The pocket this text sits in is
# y in [+36, +48] at x < 0.52: bounded below by the lone cell at (0.36, +28), above by nothing until the
# red axis label at y = +63, and on the RIGHT by the ETTh1 annotation, whose third line starts at
# x = 0.55 at exactly this height with its leader descending through x in [0.62, 0.76] just below.
# Measured at this font: 0.0158 x-units per character from x = 0.055, so 29 characters is the budget for
# the top line and ~33 for the others. The first draft put the count and the interval on ONE line -- 57
# characters, running to x = 0.68 -- and the closing bracket came out struck through by that leader; the
# second wrapped to two and the top line still touched the red text. Hence three lines, and the first
# says "pooled:" rather than "pooled," with a trailing colon to buy the two characters that clear it.
# DO NOT rejoin these lines. If a records change moves the lone cell or the annotation group this text
# will collide with one of them, which is a thing to LOOK at the .png for, not something an assert
# can catch.
axr.text(0.055, 42,
         rf"pooled: {POOLED[3]} cells, {POOLED[4]} clusters" "\n"
         rf"$\rho={POOLED[0]:+.2f}$ [{POOLED[1]:+.2f}, {POOLED[2]:+.2f}]" "\n"
         r"$\it{between}$ backbones, not within one",
         fontsize=4.8, color=INK_MUTED, ha="left", va="center", linespacing=1.25)

# Section 1's running example, drawn as two leader-line groups rather than four point labels: the pair
# is the unit of the argument -- same backbone, two datasets, opposite lessons -- and four separate
# labels at this size would read as four unrelated cells.
def group(refs, tx, ty, text, colour, ha="center"):
    axr.text(tx, ty, text, fontsize=4.7, color=colour, ha=ha, va="center", linespacing=1.15,
             zorder=5)
    for k in refs:
        axr.annotate("", xy=(RUN[k]["cka"], RUN[k]["bd_test"]), xytext=(tx, ty),
                     arrowprops=dict(arrowstyle="-", color=colour, lw=0.45, alpha=0.75,
                                     shrinkA=7, shrinkB=3), zorder=3)


group(_E1, 0.50, 52, "Moirai-Base / ETTh1\nleast drifted of its arm,\nand its only degraders", DAMAGE)
group(_E2, 0.60, -31, "Moirai-Base / ETTh2:\nmost drifted, improves most", ADAPT, ha="left")

handles = [plt.Line2D([], [], marker=m, linestyle="none", markersize=3.6,
                      markerfacecolor="none", markeredgecolor=INK_SECONDARY,
                      markeredgewidth=0.9, label=b) for b, m in MARKER.items()]
# ONE ROW, BOTTOM RIGHT. A stacked legend is ~27 pp of this y-axis tall and there is no column of empty
# plane that deep left on the CKA axis; laid out in three columns it needs only the strip below
# y = -35, where the lowest high-CKA cell sits 30 pp above it. The old upper-left pocket is now the
# rho block.
leg = axr.legend(handles=handles, fontsize=5.1, loc="center right", ncol=3,
                 bbox_to_anchor=(1.0, 0.045), frameon=False, columnspacing=0.9,
                 handletextpad=0.35, borderpad=0.1, labelspacing=0.25)
for t in leg.get_texts():
    t.set_color(INK_SECONDARY)

out = save(fig, Path(__file__).with_name("fig1_diagnostic_flow.png"))
print(f"wrote {out} and .pdf  ({len(ROWS)} cells, {len(PASSING)} gate-passing, "
      f"{sum(is_degradation(r) for r in ROWS)} degradation, {len(REVERSALS)} reversals)")
