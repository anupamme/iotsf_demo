#!/usr/bin/env python3
"""
Fig. 1 (workshop version) -- the diagnostic chain, and the scatter that motivates it.

(a) The chain, which exists to keep three quantities apart: CKA drift is OBSERVATIONAL, B-D is
    INTERVENTIONAL, forecast error is the OUTCOME. Conflating them is the reflex the paper argues
    against.
(b) Every intervention cell as (CKA, held-out B-D). The message is that NO CUT ON THIS AXIS
    separates the two regimes: the eight pretrained-capability-degradation cells span CKA
    0.36-0.89, and the two most-preserved cells in the whole matrix (Electricity7, CKA 0.94/0.95)
    are cells where adaptation HELPS. The reflex -- large drift means damage -- is not merely
    inverted here, it is unordered. Note the axis is not a licensed decision boundary either: CKA
    is only comparable within a backbone, so cutting across marker shapes is what S2 forbids.

    The earlier "degradation cells are the least drifted" reading held at 23 cells and broke when
    the prospective arm added large/ETTh1 (CKA 0.36) and large/Weather (CKA 0.62) to the
    degradation set. The weaker per-cell claim is the stronger thesis, and the one the title makes.

An earlier version of this figure was a 15-bar chart with hand-typed values, which is how it came
to silently drop a cell and aggregate four others without saying so. Every point below is read from
cell_matrix.build_rows(), the same function behind the tables and the statistics.

SIZED FOR 1:1 PLACEMENT. The NeurIPS textwidth is 5.5in, so this figure is authored at 5.5in and
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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})

# ================================================================ data
with contextlib.redirect_stdout(io.StringIO()):
    ROWS = [r for r in cell_matrix.build_rows() if r["bd_test"] is not None and r["cka"] is not None]
assert len(ROWS) == 31, f"expected 31 intervention cells, got {len(ROWS)}"


def is_degradation(r):
    """The paper's own definition, applied here so the figure cannot disagree with the text."""
    return (r["gate"] is not None and r["gate"] >= 0.20
            and r.get("forg_b") is not None and r["forg_b"] > 0 and r["forg_d"] < 0
            and r["forg_b_pos"] == r["seeds"] and r["forg_d_neg"] == r["seeds"])


assert sum(is_degradation(r) for r in ROWS) == 8, "the eight degradation cells must be the eight"


def backbone(r):
    c = r["cell"]
    return "TimesFM" if c.startswith("TimesFM") else ("Chronos" if c.startswith("Chronos") else "Moirai")


MARKER = {"Moirai": "o", "Chronos": "s", "TimesFM": "^"}

fig = plt.figure(figsize=(5.5, 1.62), dpi=150)
fig.patch.set_facecolor(SURFACE)
gs = GridSpec(1, 2, width_ratios=[0.86, 1.14], left=0.005, right=0.985,
              top=0.99, bottom=0.005, wspace=0.30)

# ================================================================ (a) the chain
axl = fig.add_subplot(gs[0])
axl.set_facecolor(SURFACE)
axl.set_xlim(0, 100); axl.set_ylim(0, 100); axl.axis("off")

axl.text(0, 99.5, "(a) the chain, and what each step measures",
         fontsize=6.4, color=INK_PRIMARY, fontweight="bold", va="top")


def box(cy, h, title, sub, fc, ec):
    axl.add_patch(FancyBboxPatch((2, cy - h / 2), 62, h,
                                 boxstyle="round,pad=0.6,rounding_size=2",
                                 linewidth=0.8, edgecolor=ec, facecolor=fc, zorder=3))
    axl.text(33, cy + (2.4 if sub else 0), title, fontsize=5.9, color=INK_PRIMARY,
             ha="center", va="center", fontweight="bold", zorder=4)
    if sub:
        axl.text(33, cy - 4.6, sub, fontsize=5.1, color=INK_SECONDARY,
                 ha="center", va="center", zorder=4)


def arrow(y0, y1):
    axl.add_patch(FancyArrowPatch((33, y0), (33, y1), arrowstyle="-|>", mutation_scale=5,
                                  linewidth=0.7, color=INK_MUTED, zorder=2))


def tag(cy, text, colour):
    axl.text(67, cy, text, fontsize=5.0, color=colour, ha="left", va="center",
             style="italic", zorder=4)


box(79, 14, "released checkpoint", "+ target series", "#f2f1ec", INK_MUTED)
arrow(71.5, 65)
box(56, 15, "inclusion criterion", r"$R^2_{\rm task}\geq 0.20$", "#f2f1ec", INK_MUTED)
arrow(48, 41)
box(32, 15, "fine-tune (B)  vs", "freeze encoder (D)", ADAPT_F, ADAPT)
arrow(24, 17)
box(8, 14, "held-out evaluation", "test split", "#f2f1ec", INK_MUTED)

tag(56, "is there capability\nworth preserving?", INK_SECONDARY)
tag(32, "CKA: observational\nB$-$D: interventional", DAMAGE)
tag(8, "forecast MSE:\nthe outcome", INK_SECONDARY)

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

for r in ROWS:
    dmg = is_degradation(r)
    helps = r["bd_test"] < 0
    ec = DAMAGE if dmg else (ADAPT if helps else NEUTRAL)
    fc = DAMAGE_F if dmg else ("none" if helps else NEUTRAL_F)
    axr.scatter(r["cka"], r["bd_test"], s=26, marker=MARKER[backbone(r)],
                facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=4)

axr.set_xlim(0.02, 1.02)
axr.set_ylim(-46, 69)
axr.set_xlabel("CKA drift  (1.0 = unchanged representation)", fontsize=5.9, color=INK_SECONDARY,
               labelpad=1.5)
axr.set_ylabel("held-out B$-$D (pp)", fontsize=5.9, color=INK_SECONDARY, labelpad=1.5)
axr.tick_params(labelsize=5.4, colors=INK_SECONDARY, length=2, width=0.6, pad=1.2)

axr.text(-0.10, 1.03, "(b) no cut on CKA separates the two regimes",
         transform=axr.transAxes, fontsize=6.4, color=INK_PRIMARY, fontweight="bold", va="bottom")
axr.text(0.28, 52, "freezing better\n8 degradation cells,\nspanning CKA 0.36$-$0.89",
         fontsize=5.1, color=DAMAGE, ha="left", va="center")
axr.text(0.62, -35, "adaptation helps", fontsize=5.3, color=ADAPT, ha="left", va="bottom")

# The near-tie, drawn because it is the whole argument in two points: same backbone (Moirai-Large),
# same horizon (h=96), CKA 0.004 apart, opposite sides of the regime. Coordinates are asserted
# against the rows rather than typed, so the annotation cannot drift from the data.
_tie = {r["cell"]: r for r in ROWS
        if r["cell"] in ("Moirai-large_Weather_h96", "Moirai-large/ETTh2 h96 n500")}
assert len(_tie) == 2, f"near-tie cells not found: {sorted(_tie)}"
_hi, _lo = _tie["Moirai-large_Weather_h96"], _tie["Moirai-large/ETTh2 h96 n500"]
assert abs(_hi["cka"] - _lo["cka"]) < 0.01, "the near-tie is no longer near"
_x = (_hi["cka"] + _lo["cka"]) / 2
axr.annotate("", xy=(_x, _hi["bd_test"]), xytext=(_x, _lo["bd_test"]),
             arrowprops=dict(arrowstyle="<->", color=INK_SECONDARY, lw=0.6,
                             shrinkA=2.5, shrinkB=2.5), zorder=3)
axr.text(_x + 0.022, (_hi["bd_test"] + _lo["bd_test"]) / 2,
         f"{abs(_hi['bd_test'] - _lo['bd_test']):.0f} pp apart\nat equal CKA",
         fontsize=4.8, color=INK_SECONDARY, ha="left", va="center", linespacing=1.15)

handles = [plt.Line2D([], [], marker=m, linestyle="none", markersize=3.6,
                      markerfacecolor="none", markeredgecolor=INK_SECONDARY,
                      markeredgewidth=0.9, label=b) for b, m in MARKER.items()]
leg = axr.legend(handles=handles, fontsize=5.1, loc="lower right", frameon=False,
                 handletextpad=0.35, borderpad=0.1, labelspacing=0.25)
for t in leg.get_texts():
    t.set_color(INK_SECONDARY)

out = str(Path(__file__).with_name("fig1_diagnostic_flow.png"))
fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.01, facecolor=SURFACE)
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01,
            facecolor=SURFACE)
print(f"wrote {out} and .pdf  ({len(ROWS)} cells, "
      f"{sum(is_degradation(r) for r in ROWS)} degradation)")
