#!/usr/bin/env python3
"""
Fig. 1 -- the screen, and why nothing survives it.

(a) The chain as a FUNNEL. Every box carries the count of cells still standing, because the paper's
    result is where the funnel empties, not what the chain looks like. 32 screened cells enter; 5
    clear the inclusion criterion; 0 meet the three-clause degradation definition. The last box is
    highlighted because that zero is the paper's claim. Each per-box tag asks ONE short question --
    the CKA-vs-B-D distinction is carried by the boxed display in Section 1, and repeating it here is
    what made this panel unreadable at a glance.

(b) Every intervention cell as (gate, held-out B-D). This panel used to plot CKA on the x-axis and
    argue that no cut separated the regimes. It now plots the GATE, for two reasons. First, there are
    no regimes left to separate: with the gate computed against the fitted linear baseline the
    degradation set is empty, so a panel whose legend said "8 degradation cells" was drawing a class
    that does not exist. Second, the gate is the axis the paper is actually about, and on it the
    relationship runs the WRONG WAY: within Moirai, rho = -0.52 (CI [-0.84, -0.03], excludes zero),
    i.e. higher measured pre-trained value predicts encoder adaptation helping MORE. Only 5 of 32
    cells sit right of the 0.20 threshold at all, and 3 of those 5 sit below zero, where adaptation
    helps. The dissociation reading survives too, in weaker form and off this axis: within Moirai
    rho(CKA, B-D) = +0.17 with CI [-0.33, +0.60]. Both correlations are computed here, not typed.

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
import io
import contextlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cell_matrix  # noqa: E402

# ---- ink / chrome (dataviz reference, light surface) ----
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#7f7d78"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
DAMAGE, DAMAGE_F = "#c0392b", "#f6dcd9"
ADAPT, ADAPT_F = "#1a7a3c", "#dcefe1"
NEUTRAL, NEUTRAL_F = "#7f7d78", "#ececec"
HELDOUT, HELDOUT_F = "#1f5fa8", "#dde7f4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})

GATE = 0.20
N_SCREENED = 32  # screened cells; 31 of them have a paired B/D run (TimesFM/Electricity does not)

# ================================================================ data
with contextlib.redirect_stdout(io.StringIO()):
    ROWS = [r for r in cell_matrix.build_rows() if r["bd_test"] is not None and r["cka"] is not None]
assert len(ROWS) == 31, f"expected 31 intervention cells, got {len(ROWS)}"


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

PASSING = [r for r in ROWS if r["gate"] >= GATE]
assert len(PASSING) == 5, f"expected 5 gate-passing intervention cells, got {len(PASSING)}"
# Of the five, the ones full fine-tuning IMPROVES -- clause (ii) fails outright for these.
IMPROVED = [r for r in PASSING if r["forg_b"] < 0]
assert len(IMPROVED) == 3, f"expected 3 improved survivors, got {len(IMPROVED)}"
_imp = sorted(abs(r["forg_b"]) for r in IMPROVED)

# Sign reversals between the selection split and the disjoint test split -- a measurement
# precondition, so panel (a) carries it. Counted here rather than typed, and the one-directionality
# that licenses the word "bias" is asserted, not assumed.
REVERSALS = [r for r in ROWS
             if r.get("bd_val") is not None and (r["bd_val"] > 0) != (r["bd_test"] > 0)]
assert len(REVERSALS) == 4, f"expected 4 sign reversals, got {len(REVERSALS)}"
assert all(r["bd_val"] > 0 > r["bd_test"] for r in REVERSALS), \
    "reversals are no longer all in the direction that flatters the frozen encoder"


def backbone(r):
    c = r["cell"]
    return "TimesFM" if c.startswith("TimesFM") else ("Chronos" if c.startswith("Chronos") else "Moirai")


MARKER = {"Moirai": "o", "Chronos": "s", "TimesFM": "^"}

# Within-backbone correlations. Moirai is the only arm with enough cells to ask, and pooling across
# backbones is what S5.2 forbids, so the number the panel prints is the within-Moirai one.
_M = [r for r in ROWS if backbone(r) == "Moirai"]
RHO_GATE = stats.spearmanr([r["gate"] for r in _M], [r["bd_test"] for r in _M])
RHO_CKA = stats.spearmanr([r["cka"] for r in _M], [r["bd_test"] for r in _M])
assert RHO_GATE.statistic < 0, "the gate is no longer anti-predictive; panel (b)'s title is wrong"

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


def box(cy, h, title, sub, fc, ec, bold_sub=False):
    axl.add_patch(FancyBboxPatch((2, cy - h / 2), 62, h,
                                 boxstyle="round,pad=0.6,rounding_size=2",
                                 linewidth=0.8, edgecolor=ec, facecolor=fc, zorder=3))
    axl.text(33, cy + (2.4 if sub else 0), title, fontsize=5.9, color=INK_PRIMARY,
             ha="center", va="center", fontweight="bold", zorder=4)
    if sub:
        axl.text(33, cy - 4.6, sub, fontsize=5.1,
                 color=ec if bold_sub else INK_SECONDARY,
                 ha="center", va="center", zorder=4,
                 fontweight="bold" if bold_sub else "normal")


def arrow(y0, y1):
    axl.add_patch(FancyArrowPatch((33, y0), (33, y1), arrowstyle="-|>", mutation_scale=5,
                                  linewidth=0.7, color=INK_MUTED, zorder=2))


def tag(cy, text, colour):
    axl.text(67, cy, text, fontsize=4.8, color=colour, ha="left", va="center",
             style="italic", linespacing=1.12, zorder=4)


box(79, 14, f"{N_SCREENED} screened cells", "6 datasets, 3 backbones", "#f2f1ec", INK_MUTED)
arrow(71.5, 65)
box(56, 15, r"$R^2_{\rm task}(\rm PT)\geq 0.20$?",
    f"{len(PASSING)} pass, {N_SCREENED - len(PASSING)} fail", "#f2f1ec", INK_MUTED)
arrow(48, 41)
box(32, 15, "fine-tune (B)  vs", "freeze encoder (D)", ADAPT_F, ADAPT)
arrow(24, 17)
box(8, 14, "degradation, every seed",
    f"{sum(is_degradation(r) for r in ROWS)} cells", DAMAGE_F, DAMAGE, bold_sub=True)

# One short question per step, matching the boxed display in the paper's Section 2. The tags carry
# the two facts that make the funnel's zero informative rather than merely small: WHY 27 cells drop
# out (no demonstrated advantage to lose), and that the survivors fail in the strongest possible
# direction (fine-tuning improves them).
tag(56, f"worth preserving?\n{N_SCREENED - len(PASSING)} of {N_SCREENED} cells: no", INK_SECONDARY)
tag(32, f"was allowing it worth it?\nheld-out, not selection:\n"
        f"{len(REVERSALS)} of {len(ROWS)} cells reverse", HELDOUT)
tag(9.5, f"{len(IMPROVED)} of the {len(PASSING)} survivors\nare "
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

# The threshold, and the sliver of the plane where a degradation cell could possibly live.
axr.axvline(GATE, color=HELDOUT, lw=0.7, ls=(0, (3, 2)), zorder=2)
axr.axvspan(GATE, 0.58, color=HELDOUT_F, alpha=0.55, zorder=0)

for r in ROWS:
    passes = r["gate"] >= GATE
    helps = r["bd_test"] < 0
    ec = ADAPT if helps else DAMAGE
    fc = (ADAPT_F if helps else DAMAGE_F) if passes else "none"
    axr.scatter(r["gate"], r["bd_test"], s=26, marker=MARKER[backbone(r)],
                facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=4)

axr.set_xlim(-1.78, 0.58)
axr.set_ylim(-46, 69)
axr.set_xlabel(r"gate  $R^2_{\rm task}(\rm PT)$   (0.20 = threshold; higher = more to lose)",
               fontsize=5.9, color=INK_SECONDARY, labelpad=1.5)
axr.set_ylabel("held-out B$-$D (pp)", fontsize=5.9, color=INK_SECONDARY, labelpad=1.5)
axr.tick_params(labelsize=5.4, colors=INK_SECONDARY, length=2, width=0.6, pad=1.2)

axr.text(-0.10, 1.03, "(b) the gate points the wrong way",
         transform=axr.transAxes, fontsize=6.4, color=INK_PRIMARY, fontweight="bold", va="bottom")
axr.text(-1.70, 56, "freezing better", fontsize=5.3, color=DAMAGE, ha="left", va="center")
axr.text(-0.55, -43, "adaptation helps", fontsize=5.3, color=ADAPT, ha="center", va="bottom")
axr.text(-1.70, 30,
         f"Moirai only ($n$={len(_M)}):\n"
         rf"$\rho_{{\rm gate}}={RHO_GATE.statistic:.2f}$, $p$={RHO_GATE.pvalue:.3f}" "\n"
         rf"$\rho_{{\rm CKA}}={RHO_CKA.statistic:+.2f}$, $p$={RHO_CKA.pvalue:.2f}",
         fontsize=4.8, color=INK_SECONDARY, ha="left", va="center", linespacing=1.25)

# The bracket spanning the survivors, drawn because it is the whole argument in five points: the only
# cells with a capability to protect disagree about the intervention by 28 pp, and the majority of
# them are improved by the thing that was supposed to damage them. Coordinates come from the rows.
_hi = max(PASSING, key=lambda r: r["bd_test"])
_lo = min(PASSING, key=lambda r: r["bd_test"])
axr.annotate("", xy=(0.52, _hi["bd_test"]), xytext=(0.52, _lo["bd_test"]),
             arrowprops=dict(arrowstyle="<->", color=INK_SECONDARY, lw=0.6,
                             shrinkA=1.5, shrinkB=1.5), zorder=3)
# Above the band, not beside it: the band is only ~28 pt wide in print, so a label to the left of the
# bracket lands on the survivors themselves and one to the right runs off the axes.
axr.text(0.40, 45,
         f"the {len(PASSING)} that\nclear the gate:\n"
         f"{_hi['bd_test'] - _lo['bd_test']:.0f} pp apart",
         fontsize=4.8, color=INK_SECONDARY, ha="center", va="center", linespacing=1.15)

handles = [plt.Line2D([], [], marker=m, linestyle="none", markersize=3.6,
                      markerfacecolor="none", markeredgecolor=INK_SECONDARY,
                      markeredgewidth=0.9, label=b) for b, m in MARKER.items()]
# Not "lower right": that corner is inside the shaded band, on top of the survivors and the bracket.
# This pocket (gate < -1.0, B-D in [-13,-27]) is the only empty region large enough.
leg = axr.legend(handles=handles, fontsize=5.1, loc="center left",
                 bbox_to_anchor=(0.005, 0.24), frameon=False,
                 handletextpad=0.35, borderpad=0.1, labelspacing=0.25)
for t in leg.get_texts():
    t.set_color(INK_SECONDARY)

out = str(Path(__file__).with_name("fig1_diagnostic_flow.png"))
fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.01, facecolor=SURFACE)
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01,
            facecolor=SURFACE)
print(f"wrote {out} and .pdf  ({len(ROWS)} cells, {len(PASSING)} gate-passing, "
      f"{sum(is_degradation(r) for r in ROWS)} degradation, {len(REVERSALS)} reversals)")
