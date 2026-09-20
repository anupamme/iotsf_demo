#!/usr/bin/env python3
"""
Appendix figure -- the gate against the outcome it was supposed to predict.

WHERE THIS CAME FROM. This was panel (b) of Figure 1 until 20 Sep 2026, when the main figure's
x-axis went back to CKA because the title and abstract now state the CKA-to-outcome relation as the
paper's finding. The gate-to-outcome relation is a separate claim -- the pre-registered predictor of
Section 7 -- and it did not stop being true when it stopped being the headline, so it is drawn here
in full rather than reduced to a rho in a caption.

WHAT IT SHOWS. The gate is scored on the SELECTION windows and the outcome on the disjoint held-out
windows, so the horizontal position of a cell is not a function of its vertical position: this is a
screen, not a re-description of the outcome. Within Moirai the point estimate runs the WRONG WAY --
higher measured pre-trained value goes with encoder adaptation helping MORE -- and the shaded strip
right of 0.20 is the only region a cell could occupy and still meet clause (i) of the degradation
definition. Most of the cells in it sit BELOW zero, where fine-tuning the encoder helped.

ILI IS OFF SCALE AND SAID SO. Its gate is about -10, a dozen times further left than any other cell,
and an axis wide enough to include it compresses the other 30 into a fifth of the plot. The retired
panel had the same xlim problem and simply clipped the point, with nothing on the figure saying a
cell was missing; here the count in the corner is the number drawn, the arrow names the one that is
not, and the assert below fails if a second cell ever leaves the frame.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (  # noqa: E402
    ADAPT, ADAPT_F, DAMAGE, DAMAGE_F, GATE, GRID, HELDOUT, HELDOUT_F,
    INK_MUTED, INK_PRIMARY, INK_SECONDARY, MARKER, SURFACE,
    backbone, intervention_rows, save,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import cluster_keys  # noqa: E402

ROWS = intervention_rows()
PASSING = [r for r in ROWS if r["gate"] >= GATE]
assert len(PASSING) == 7, f"expected 7 gate-passing cells, got {len(PASSING)}"

XLO, XHI = -1.0, 1.02
OFF = [r for r in ROWS if r["gate"] < XLO]
assert [r["ref"] for r in OFF] == ["ili"], \
    f"cells other than ILI now fall outside the frame: {[r['ref'] for r in OFF]}"
DRAWN = [r for r in ROWS if r["gate"] >= XLO]

_M = [r for r in ROWS if backbone(r) == "Moirai"]
_MCL = cluster_keys.clusters_for([r["cell"] for r in _M])
CI_GATE = cluster_keys.cluster_bootstrap_spearman([r["gate"] for r in _M],
                                                  [r["bd_test"] for r in _M], _MCL)
RHO_ALL = stats.spearmanr([r["gate"] for r in ROWS], [r["bd_test"] for r in ROWS])
assert CI_GATE[0] < 0, "the gate is no longer anti-predictive within Moirai; the caption says it is"

fig, ax = plt.subplots(figsize=(5.5, 2.7), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(INK_MUTED); ax.spines[sp].set_linewidth(0.6)

ax.axhline(0, color=INK_MUTED, lw=0.7, zorder=1)
ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
ax.axvline(GATE, color=HELDOUT, lw=0.7, ls=(0, (3, 2)), zorder=2)
ax.axvspan(GATE, XHI, color=HELDOUT_F, alpha=0.55, zorder=0)

for r in DRAWN:
    helps = r["bd_test"] < 0
    ec = ADAPT if helps else DAMAGE
    fc = (ADAPT_F if helps else DAMAGE_F) if r["gate"] >= GATE else "none"
    ax.scatter(r["gate"], r["bd_test"], s=30, marker=MARKER[backbone(r)],
               facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=4)

ax.set_xlim(XLO, XHI)
# Headroom above the tallest cell (62 pp) so the two correlations sit in empty plane instead of on
# top of the Moirai-Small/Weather points, which is where they landed in the retired panel.
ax.set_ylim(-48, 92)
ax.set_xlabel(r"baseline-relative pre-trained advantage  $V_{\rm ridge}$, selection windows"
              "   (0.20 = threshold; higher = more to lose)",
              fontsize=7.0, color=INK_SECONDARY, labelpad=2.0)
ax.set_ylabel(r"held-out $\Delta_{\rm enc}$ (pp)", fontsize=7.0, color=INK_SECONDARY, labelpad=2.0)
ax.tick_params(labelsize=6.4, colors=INK_SECONDARY, length=2.4, width=0.6, pad=1.4)

ax.text(0.0, 1.02, "The screen's own axis: measured pre-trained value does not predict "
                   "whether encoder adaptation helped",
        transform=ax.transAxes, fontsize=7.4, color=INK_PRIMARY, fontweight="bold", va="bottom")
ax.text(XLO + 0.03, 58, "freezing better", fontsize=6.4, color=DAMAGE, ha="left", va="center")
ax.text(XLO + 0.03, -44, "adaptation helps", fontsize=6.4, color=ADAPT, ha="left", va="center")
ax.text(XLO + 0.03, 81,
        f"Moirai only ($n$={len(_M)}, {CI_GATE[4]} clusters): "
        rf"$\rho={CI_GATE[0]:+.2f}$ [{CI_GATE[1]:+.2f}, {CI_GATE[2]:+.2f}]" "\n"
        rf"all {len(ROWS)} cells pooled: $\rho={RHO_ALL.statistic:+.2f}$"
        "  (pooling is what our own clustering rule forbids)",
        fontsize=6.0, color=INK_SECONDARY, ha="left", va="center", linespacing=1.3)

# The survivors' spread, which is the argument in seven points: the only cells with a demonstrated
# advantage to protect disagree about the intervention by tens of points, and most are IMPROVED by it.
_hi = max(PASSING, key=lambda r: r["bd_test"])
_lo = min(PASSING, key=lambda r: r["bd_test"])
ax.annotate("", xy=(0.99, _hi["bd_test"]), xytext=(0.99, _lo["bd_test"]),
            arrowprops=dict(arrowstyle="<->", color=INK_SECONDARY, lw=0.6,
                            shrinkA=1.5, shrinkB=1.5), zorder=3)
ax.text(0.96, (_hi["bd_test"] + _lo["bd_test"]) / 2,
        f"the {len(PASSING)} that clear\nthe gate: {_hi['bd_test'] - _lo['bd_test']:.0f} pp apart",
        fontsize=6.0, color=INK_SECONDARY, ha="right", va="center", linespacing=1.2)

# The cell that is not in the frame, named on the frame.
_ili = OFF[0]
ax.annotate(f"ILI, $V_{{\\rm ridge}}{{=}}{_ili['gate']:.2f}$,\noff scale "
            f"($\\Delta_{{\\rm enc}}{{=}}{_ili['bd_test']:+.0f}$)",
            xy=(XLO, _ili["bd_test"]), xytext=(XLO + 0.10, _ili["bd_test"] + 13),
            fontsize=6.0, color=INK_SECONDARY, ha="left", va="center", linespacing=1.2,
            arrowprops=dict(arrowstyle="-|>", mutation_scale=6, color=INK_SECONDARY, lw=0.6,
                            shrinkA=2, shrinkB=1))

handles = [plt.Line2D([], [], marker=m, linestyle="none", markersize=4.2,
                      markerfacecolor="none", markeredgecolor=INK_SECONDARY,
                      markeredgewidth=0.9, label=b) for b, m in MARKER.items()]
leg = ax.legend(handles=handles, fontsize=6.2, loc="lower right", ncol=3,
                bbox_to_anchor=(1.0, -0.02), frameon=False, columnspacing=1.0,
                handletextpad=0.35, borderpad=0.1)
for t in leg.get_texts():
    t.set_color(INK_SECONDARY)

out = save(fig, Path(__file__).with_name("figA_gate_scatter.png"))
print(f"wrote {out} and .pdf  ({len(DRAWN)} of {len(ROWS)} cells drawn, "
      f"{len(PASSING)} gate-passing, rho_Moirai={CI_GATE[0]:+.3f})")
