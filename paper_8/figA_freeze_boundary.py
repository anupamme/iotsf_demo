#!/usr/bin/env python3
"""
Appendix figure -- where the freeze boundary is drawn, module by module.

WHY IT EXISTS. Two of the paper's seven survivors change sign between the ordinary frozen encoder
and the strict one (Appendix \\ref{app:strictfreeze}), so "we froze the encoder" is not a complete
description of a control: which modules were held fixed decides the answer. A reviewer asked for the
boundary as a diagram rather than as a sentence, and the sentence it replaces had to name four
PyTorch module attributes to say the same thing.

THE GRID IS READ OFF THE TRAINING CODE, NOT DRAWN FROM MEMORY. scripts/finetune_forecasting.py sets
requires_grad in exactly three patterns: B trains everything; D calls
`for param in encoder.parameters(): param.requires_grad = False`; H additionally freezes `in_proj`
and `mask_encoding`, after which its own log line reads "only param_proj trains". TRAINABLE below is
that code transcribed, and its keys are the attribute names so the transcription can be checked
against the source by grep.

ONE NUMBER, AND IT IS DERIVED. Condition H's CKA is exactly 1.000 on every cell it ran on, which is
the consequence the diagram exists to explain: with everything upstream frozen the encoder's output
is a fixed function of its input. That is asserted from the run records rather than typed, because
it is the claim that distinguishes H from D and the whole figure is wrong without it.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (  # noqa: E402
    ADAPT, ADAPT_F, GRID, INK_MUTED, INK_PRIMARY, INK_SECONDARY,
    NEUTRAL, NEUTRAL_F, SURFACE, intervention_rows, save,
)

# Module rows, in forward order: what the model does to a window on its way to a forecast.
MODULES = [
    ("in_proj", "input projection", "patch $\\rightarrow$ model width"),
    ("mask_encoding", "mask encoding", "learned mask token"),
    ("encoder", "transformer encoder stack", "self-attention $+$ FFN, $\\times L$"),
    ("param_proj", "distribution head", "$\\rightarrow$ mixture parameters"),
]
# Transcribed from finetune_forecasting.py's requires_grad blocks. True = the optimiser may move it.
TRAINABLE = {
    "B": {"in_proj": True,  "mask_encoding": True,  "encoder": True,  "param_proj": True},
    "D": {"in_proj": True,  "mask_encoding": True,  "encoder": False, "param_proj": True},
    "H": {"in_proj": False, "mask_encoding": False, "encoder": False, "param_proj": True},
}
CONDITIONS = [("B", "full fine-tune"), ("D", "frozen encoder"), ("H", "strict frozen encoder")]

_H = [r for r in intervention_rows() if r.get("cka_h") is not None]
assert _H, "no condition-H cells in the run records; this figure's whole point is unsupported"
assert {r["cka_h"] for r in _H} == {1.0}, \
    f"condition H's CKA is no longer exactly 1 ({sorted({r['cka_h'] for r in _H})})"

fig, ax = plt.subplots(figsize=(5.5, 2.15), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

COLX = {"B": 44.5, "D": 64.0, "H": 83.5}
COLW = 17.5
ROWY = [82.0, 65.5, 47.0, 28.5]
ROWH = [11.0, 11.0, 14.0, 11.0]

# Column headers.
for key, name in CONDITIONS:
    ax.text(COLX[key], 97.0, key, fontsize=7.4, color=INK_PRIMARY, fontweight="bold",
            ha="center", va="center")
    ax.text(COLX[key], 91.5, name, fontsize=5.9, color=INK_SECONDARY, ha="center", va="center")

# The forward-pass spine, so the row order reads as a model rather than as a list.
ax.add_patch(FancyArrowPatch((3.0, ROWY[0] + 4), (3.0, ROWY[-1] - 4), arrowstyle="-|>",
                             mutation_scale=6, linewidth=0.7, color=INK_MUTED, zorder=2))
ax.text(1.2, (ROWY[0] + ROWY[-1]) / 2, "forward pass", fontsize=5.4, color=INK_MUTED,
        ha="center", va="center", rotation=90)

for (attr, label, detail), cy, h in zip(MODULES, ROWY, ROWH):
    # The PyTorch attribute name is on the figure, in monospace, because it is what a reader has to
    # grep for to check this grid against finetune_forecasting.py.
    ax.text(32.0, cy + 3.4, label, fontsize=6.4, color=INK_PRIMARY, ha="right", va="center")
    ax.text(32.0, cy - 1.0, attr, fontsize=5.4, color=INK_SECONDARY, ha="right", va="center",
            family="monospace")
    ax.text(32.0, cy - 5.6, detail, fontsize=5.4, color=INK_MUTED, ha="right", va="center")
    for key, _ in CONDITIONS:
        trains = TRAINABLE[key][attr]
        ec, fc = (ADAPT, ADAPT_F) if trains else (NEUTRAL, NEUTRAL_F)
        ax.add_patch(FancyBboxPatch((COLX[key] - COLW / 2, cy - h / 2), COLW, h,
                                    boxstyle="round,pad=0.5,rounding_size=1.6",
                                    linewidth=0.8, edgecolor=ec, facecolor=fc,
                                    hatch=None if trains else "////", zorder=3))
        # The frozen label gets a knockout panel: at this size the hatch runs straight through the
        # word and "frozen" was the harder of the two states to read, which is backwards.
        ax.text(COLX[key], cy, "trains" if trains else "frozen", fontsize=6.0,
                color=ec, ha="center", va="center", fontweight="bold" if trains else "normal",
                zorder=4,
                bbox=None if trains else dict(facecolor=NEUTRAL_F, edgecolor="none", pad=1.4))

# The two contrasts the grid licenses, named on the figure so the reader does not have to subtract
# columns in their head.
ax.plot([34.5, 34.5], [20.0, 90.0], color=GRID, lw=0.8, zorder=1)
ax.text(8.0, 8.0,
        # "below 1", not "near 1": D's CKA is just short of 1 on Moirai but falls to 0.38 on
        # TimesFM (S3), so the only thing true of every arm is that it is not 1.
        "$\\Delta_{\\rm enc}$ = B $-$ D isolates ENCODER WEIGHT UPDATES, but the encoder's input "
        "still moves under D, so its\noutput is not a fixed function of the window and its CKA is "
        f"below 1.   B $-$ H also holds the input\nfixed: H's CKA is exactly 1.000 on all {len(_H)} "
        "cells it ran on, which is what makes it strict.",
        fontsize=5.7, color=INK_SECONDARY, ha="left", va="center", linespacing=1.35)

out = save(fig, Path(__file__).with_name("figA_freeze_boundary.png"))
print(f"wrote {out} and .pdf  (3 conditions x {len(MODULES)} modules, "
      f"H verified at CKA 1.000 on {len(_H)} cells)")
