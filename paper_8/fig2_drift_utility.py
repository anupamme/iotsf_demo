#!/usr/bin/env python3
"""
Fig. 2 (workshop version) — the drift-utility dissociation.

Scatter of encoder representational drift (CKA, post- vs pre-fine-tuning) against
task improvement after fine-tuning, across four time-series foundation models,
colored + shaped by diagnostic regime.

NOTE: point values are REPRESENTATIVE, assembled from the reported per-run ranges
(this is a mock-up to evaluate the figure). Swap in exact per-seed (CKA, utility)
pairs from the results JSON before using in the paper.

Palette: dataviz reference categorical slots 1-4 (validated for scatter/all-pairs).
Secondary encoding (marker shape + direct labels + dark edges) satisfies the
CVD-floor-band and light-surface contrast relief rules.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- ink / chrome (dataviz reference, light surface) ----
INK_PRIMARY   = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED     = "#898781"
GRID          = "#e1e0d9"
BASELINE      = "#c3c2b7"
SURFACE       = "#fcfcfb"

# ---- regimes: (color slot, marker, label, points as (CKA, task_improvement_%)) ----
REGIMES = {
    "Beneficial specialization\n(Moirai-Base, MoE enc.)": dict(
        color="#2a78d6", marker="o",
        pts=[(0.25, 56.9), (0.29, 54.2), (0.31, 55.5),
             (0.27, 57.0), (0.33, 53.8), (0.28, 56.0)],
    ),
    "Incidental restructuring\n(Chronos / ETT)": dict(
        color="#008300", marker="s",
        pts=[(0.886, 19.0), (0.905, 14.0), (0.92, 11.0),
             (0.949, 6.0), (0.91, 15.0), (0.90, 17.0)],
    ),
    "Value-driven stability\n(Chronos / M4)": dict(
        color="#eda100", marker="D",
        pts=[(0.97, 6.0), (0.98, 4.0), (0.985, 5.0), (0.99, 7.0), (0.975, 5.5)],
    ),
    "Capacity-driven stability\n(TimesFM / ETT)": dict(
        color="#e87ba4", marker="^",
        pts=[(0.995, 2.5), (0.998, 1.2), (0.999, 0.8),
             (0.996, 2.0), (0.997, 1.5), (0.999, 1.0)],
    ),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})

fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# feared "high drift -> harm" zone (empty): the crux of the argument
ax.axhspan(-8, 0, xmin=0, xmax=(0.55-0.18)/(1.04-0.18), color=GRID, alpha=0.45, zorder=0)
ax.text(0.205, -6.2, "region practitioners fear:\nhigh drift causes harm  (no cases here)",
        fontsize=8, color=INK_MUTED, va="center", ha="left", style="italic")

# baseline
ax.axhline(0, color=BASELINE, lw=1.0, ls="--", zorder=1)

# grid (recessive, y only)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)

# scatter
for label, d in REGIMES.items():
    xs = [p[0] for p in d["pts"]]
    ys = [p[1] for p in d["pts"]]
    ax.scatter(xs, ys, s=95, marker=d["marker"], c=d["color"],
               edgecolors=INK_PRIMARY, linewidths=0.7, alpha=0.95,
               label=label.replace("\n", " "), zorder=3)

# direct labels placed in open space with leader lines (no legend; shape+color+label
# together satisfy the CVD-floor-band and light-surface contrast relief rules)
def centroid(pts):
    return np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])

# (label text, text-xy, target cluster key)
LABELS = [
    ("Beneficial specialization\n(Moirai-Base, MoE encoder)\n~70% drift, yet the largest gain",
     (0.345, 41.5), "Beneficial specialization\n(Moirai-Base, MoE enc.)", INK_PRIMARY, True),
    ("Incidental restructuring\n(Chronos / ETT)",
     (0.52, 25.0), "Incidental restructuring\n(Chronos / ETT)", INK_SECONDARY, False),
    ("Value-driven stability\n(Chronos / M4)",
     (0.55, 12.5), "Value-driven stability\n(Chronos / M4)", INK_SECONDARY, False),
    ("Capacity-driven stability\n(TimesFM / ETT)",
     (0.60, -5.0), "Capacity-driven stability\n(TimesFM / ETT)", INK_SECONDARY, False),
]
for text, txy, key, col, bold in LABELS:
    cx, cy = centroid(REGIMES[key]["pts"])
    ax.annotate(text, (cx, cy), txy, fontsize=8.2, color=col,
                ha="left", va="center", linespacing=1.1,
                fontweight="bold" if bold else "normal",
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.6,
                                shrinkA=2, shrinkB=8))

# axes
ax.set_xlim(0.18, 1.04)
ax.set_ylim(-8, 62)
ax.set_xlabel("Encoder representational drift  —  CKA (1.0 = unchanged, lower = more drift)",
              fontsize=9.5, color=INK_SECONDARY)
ax.set_ylabel("Task improvement after fine-tuning (%)",
              fontsize=9.5, color=INK_SECONDARY)
ax.set_title("Drift ≠ damage: representational drift does not predict the fine-tuning outcome",
             fontsize=11, color=INK_PRIMARY, fontweight="bold", pad=10)

# "more drift" direction hint
ax.annotate("more drift", (0.30, 61), (0.30, 61), fontsize=8, color=INK_MUTED,
            ha="center", va="top")
ax.annotate("", (0.19, 58.5), (0.42, 58.5),
            arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.8))

ax.tick_params(colors=INK_MUTED, labelsize=8.5)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(BASELINE)

fig.tight_layout()
out = "/Users/mediratta/code/paper_writing/iotsf_demo/paper_8/fig2_drift_utility.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor=SURFACE)
print("saved:", out)
