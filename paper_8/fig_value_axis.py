#!/usr/bin/env python3
"""The graded value axis: neither drift nor pre-trained value orders the encoder-adaptation outcome.

Two panels, one x-variable each, sharing the y-axis (held-out Delta_enc = B - D in percentage points,
positive = freezing the encoder did better):

  (left)  x = CKA between the pre-trained and fine-tuned encoder -- the diagnostic under test.
  (right) x = R2_task against the STRONGEST admissible rung of the eight-baseline ladder -- how much
          pre-trained value the cell demonstrably has. A minimum over rungs, not a maximum: R2_task
          = 1 - MSE_zs/MSE_base, so a weaker rival inflates it, and value that survives the best
          rival is the smallest R2_task.

Point AREA encodes the same value score in both panels, so the left panel answers the reviewer's
question visually: if drift predicted the outcome wherever there was something to lose, the large
points on the left would trend and the small ones would scatter. Backbone is encoded by colour AND
marker, because CKA is not comparable across architectures (Chronos 0.09-0.23, Moirai 0.36-0.97,
TimesFM 0.25-0.56) and a reader must be able to see that the pooled left-panel trend is largely
between-backbone.

NO HARD-CODED NUMBERS. Everything is read from results/value_axis.json, written by
scripts/value_axis.py --json, which in turn reads the graded ladder from results/gate_baselines.json.
The assertions below are the point of that: this script is a data-integrity check that happens to
draw, and it fails loudly rather than drawing a figure that disagrees with the text.

Needs matplotlib, which neither venv has:
  /opt/homebrew/bin/python3 paper_8/fig_value_axis.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
J = json.load(open(ROOT / "results/value_axis.json"))

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
STYLE = {                          # colour and marker, so the arms separate in greyscale too
    "Moirai":  ("#2a78d6", "o"),
    "Chronos": ("#eb6834", "s"),
    "TimesFM": ("#4c9a6b", "^"),
}


def backbone_of(cell):
    for bb in ("Chronos", "TimesFM"):
        if cell.startswith(bb):
            return bb
    return "Moirai"


cells = list(J["cells"].values())
cka = np.array([c["cka"] for c in cells])
val = np.array([c["value"] for c in cells])
dnc = np.array([c["d_enc"] for c in cells])
bb = np.array([backbone_of(c["cell"]) for c in cells])
anyv = np.array([c["clears_any"] for c in cells])
thr = J["gate_threshold"]

# Data-integrity assertions. Each mirrors a sentence in the paper; a failure here means the figure
# and the text have diverged, which is the failure mode this project has hit most often.
assert len(cells) == J["n_scored"] == 31, f"expected 31 scored cells, got {len(cells)}"
assert anyv.sum() == J["n_value_cells"] == 15, f"expected 15 value-cells, got {anyv.sum()}"
assert J["n_clearing_all"] == 0, f"body claims no cell clears every rung; json says {J['n_clearing_all']}"
assert (val < thr)[~anyv].all(), "a cell marked no-value scores at or above the threshold"
assert len(J["baselines"]) == 9, f"body claims an eight-rung ladder plus the floor; got {J['baselines']}"

# Point area proportional to value, floored so that the most negative cells stay visible: a cell
# that loses badly to every baseline is still a data point, and an area-zero marker would drop it.
sz = 18 + 150 * np.clip(val - val.min(), 0, None) / max(np.ptp(val), 1e-9)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.4, 2.45), dpi=150, sharey=True,
                               constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1]))
fig.patch.set_facecolor(SURFACE)
for ax in (axA, axB):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, lw=0.6)
    ax.axhline(0, color=BASELINE, lw=0.9, zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8)

for panel, xv in ((axA, cka), (axB, val)):
    for name, (col, mk) in STYLE.items():
        m = bb == name
        if not m.any():
            continue
        panel.scatter(xv[m], dnc[m], s=sz[m], marker=mk, facecolor=col, alpha=0.75,
                      edgecolor="white", linewidth=0.6, zorder=3,
                      label=f"{name} (n={m.sum()})")

axB.axvline(thr, color=INK_MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
axB.annotate(f"gate {thr:g}", xy=(thr, axB.get_ylim()[1]), xytext=(2, -3),
             textcoords="offset points", ha="left", va="top", fontsize=7, color=INK_MUTED)

axA.set_ylabel(r"held-out $\Delta_{\mathrm{enc}}$ (pp)", fontsize=8.5, color=INK_PRIMARY)
axA.set_xlabel("CKA (pre-trained vs fine-tuned encoder)", fontsize=8.5, color=INK_PRIMARY)
axB.set_xlabel(r"$R^2_{\mathrm{task}}$ vs strongest admissible rung", fontsize=8.5, color=INK_PRIMARY)

# Statistics printed from the JSON, so the caption, the body and the panel cannot drift apart.
def label(pre, key):
    c = J["correlations"][key]
    return (rf"{pre} $\rho={c['rho']:+.3f}$ [{c['lo']:+.2f}, {c['hi']:+.2f}] "
            rf"$n{{=}}{c['n']}$")


# TWO intervals per panel, not one. On the left, the pooled rho is the figure the body calls a
# backbone artefact, so printing it alone would contradict the caption; the within-Moirai row is the
# one the claim rests on. On the right, restricting to value-cells is the reviewer's question, and the
# point of showing both is that the restriction does not sharpen anything.
for ax, rows_ in ((axA, [("pooled", "cka_vs_denc_all"), ("Moirai", "cka_vs_denc_moirai")]),
                  (axB, [("all", "value_vs_denc_all"), ("value-cells", "value_vs_denc_valuecells")])):
    for i, (pre, key) in enumerate(rows_):
        ax.annotate(label(pre, key), xy=(0.5, 0.02 + 0.075 * (len(rows_) - 1 - i)),
                    xycoords="axes fraction", ha="center", va="bottom",
                    fontsize=6.5, color=INK_MUTED)

axA.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=INK_MUTED,
           handletextpad=0.2, borderpad=0.1, labelspacing=0.15)
fig.suptitle("Point area $\\propto$ measured pre-trained value; Spearman intervals cluster on "
             "(backbone, dataset)", fontsize=7.5, color=INK_MUTED, y=1.02)

out = ROOT / "paper_8/fig_value_axis"
fig.savefig(f"{out}.pdf", facecolor=SURFACE, bbox_inches="tight")
fig.savefig(f"{out}.png", facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {out}.pdf / .png   {len(cells)} cells, {anyv.sum()} value-cells, "
      f"{J['n_clearing_all']} clearing every rung")
