#!/usr/bin/env python3
"""
Fig. 2 (workshop version) — the drift-utility dissociation, within a single model.

Two stacked panels sharing the x-axis (number of fine-tuning samples):
  (top)    encoder drift: CKA falls MONOTONICALLY, with tight error bars, as more
           data drives more drift.
  (bottom) task outcome: forgetting is NON-MONOTONIC (sign flips several times)
           and noisy.

A clean monotonic driver producing a sign-flipping, high-variance outcome is the
dissociation: drift magnitude does not determine whether fine-tuning helps or hurts.

DATA: exact values from the paper's Table tab:sample_sweep (ETTh2, h=96, condition B,
Moirai-Small). n=500-5,000 are 3-seed; n=10,000 is the 10-seed CUDA-deterministic
early-stopped run (mean best epoch 4.2). Means +/- 1 SD.

No dual axis (dataviz rule): two panels, one measure each.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---- ink / chrome (dataviz reference, light surface) ----
INK_PRIMARY   = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED     = "#898781"
GRID          = "#e1e0d9"
BASELINE      = "#c3c2b7"
SURFACE       = "#fcfcfb"
BLUE          = "#2a78d6"   # slot 1 — drift
ORANGE        = "#eb6834"   # slot 6 — outcome

# ---- paper Table tab:sample_sweep (ETTh2, h=96, cond B, Moirai-Small) ----
n       = np.array([500, 1000, 2000, 5000, 10000])
cka     = np.array([0.949, 0.936, 0.814, 0.546, 0.518])
cka_sd  = np.array([0.010, 0.025, 0.042, 0.082, 0.188])
forget  = np.array([14.0, 13.4, -2.9, 6.6, -5.3])   # % ; + worse, - better
fg_sd   = np.array([6.7,  9.1,  5.9, 10.0,  6.2])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})

fig, (axA, axB) = plt.subplots(2, 1, figsize=(6.2, 3.1), dpi=150, sharex=True,
                               constrained_layout=True,
                               gridspec_kw=dict(height_ratios=[1, 1]))
fig.patch.set_facecolor(SURFACE)
for ax in (axA, axB):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)

# ---------- Panel A: drift (CKA) ----------
axA.errorbar(n, cka, yerr=cka_sd, color=BLUE, lw=2.0, marker="o", ms=8,
             mec=INK_PRIMARY, mew=0.7, ecolor=INK_MUTED, elinewidth=1.0,
             capsize=3, zorder=3)
axA.set_ylabel("Encoder drift\nCKA (lower = more drift)", fontsize=9.5, color=INK_SECONDARY)
axA.set_ylim(0.25, 1.0)
axA.annotate("clean & monotonic:\nmore data, more drift\n(CKA 0.95 to 0.52)",
             (2000, 0.814), (2600, 0.90), fontsize=8.5, color=INK_SECONDARY,
             ha="left", va="center", linespacing=1.1,
             arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.8))
axA.set_title("Within one model, drift is monotonic in $n$; task outcome is not",
              fontsize=11, color=INK_PRIMARY, fontweight="bold", pad=6)

# ---------- Panel B: outcome (forgetting) ----------
axB.axhspan(0, 26, color=GRID, alpha=0.40, zorder=0)          # "worse" half-plane
axB.axhline(0, color=BASELINE, lw=1.0, ls="--", zorder=1)
axB.errorbar(n, forget, yerr=fg_sd, color=ORANGE, lw=2.0, marker="s", ms=7.5,
             mec=INK_PRIMARY, mew=0.7, ecolor=INK_MUTED, elinewidth=1.0,
             capsize=3, zorder=3)
axB.set_ylabel("Task outcome\nforgetting (%)", fontsize=9.5, color=INK_SECONDARY)
axB.set_ylim(-13, 26)
axB.set_xlabel("Fine-tuning samples   (Moirai-Small / ETTh2, cond. B)",
               fontsize=9.5, color=INK_SECONDARY)

axB.text(430, 22.5, "worse  (forgetting)", fontsize=8, color=INK_MUTED, va="center", ha="left")
axB.text(430, -10.5, "better  (improved)", fontsize=8, color=INK_MUTED, va="center", ha="left")
# Two lines, not three: at three the block overflows the axes and touches the x tick labels.
# The dropped line ("not determined by the drift curve") is already made in the caption.
axB.annotate("outcome sign flips (+14, -3, +7, -5) —\nat maximum drift it still improves",
             (10000, -5.3), (1150, -9.6), fontsize=8.5, color=INK_PRIMARY,
             ha="left", va="center", fontweight="bold", linespacing=1.1,
             arrowprops=dict(arrowstyle="->", color=INK_PRIMARY, lw=0.9))

# shared log x with real n ticks
axB.set_xscale("log")
axB.set_xlim(430, 12500)
axB.set_xticks(n)
axB.set_xticklabels([f"{v//1000}k" if v >= 1000 else str(v) for v in n])
axB.minorticks_off()

out = "/Users/mediratta/code/paper_writing/iotsf_demo/paper_8/fig2_dissociation_sweep.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor=SURFACE)
print("saved:", out)
