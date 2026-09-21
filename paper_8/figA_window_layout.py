#!/usr/bin/env python3
"""
Appendix figure -- which timesteps Moirai is given, and which it is asked to predict.

WHY IT EXISTS. A reviewer read the appendix phrase "extended lookback sequences (context_length +
prediction_length timesteps)" as a description of target leakage, and said the point had to be made
unambiguous before acceptance. It is not leakage -- the evaluation input is 96+h steps of HISTORY,
and the target is the h steps strictly after it -- but a sentence asserting that is exactly as
trustworthy as the person who wrote it. So this figure is drawn from the index arrays the window
builders actually return, and asserts disjointness before it draws anything.

HOW THE CODE IS READ, AND WHY NOT BY IMPORT. The two builders are module-level in
scripts/finetune_forecasting.py precisely so they can be reused, but importing that module pulls in
torch, loguru and the dataset loaders, none of which the TIER A figure interpreter has and none of
which a clean clone needs in order to rebuild the paper. So their source is extracted from the file
by AST and executed with numpy alone. That keeps the figure derived from the real code -- rename
either function, change either slice, or push them back inside main(), and this script fails rather
than drawing a stale diagram.

THE TRICK THAT MAKES IT EXACT. The builders are fed `data = arange(T)`, so every value they return
IS the timestep index it came from. The bars below are those index arrays' first and last entries,
not a transcription of the slice expressions.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (  # noqa: E402
    ADAPT, ADAPT_F, GRID, HELDOUT, HELDOUT_F, INK_MUTED, INK_PRIMARY,
    INK_SECONDARY, NEUTRAL, NEUTRAL_F, SURFACE, save,
)

RUNNER = Path(__file__).resolve().parent.parent / "scripts" / "finetune_forecasting.py"
WANTED = ("make_train_sequences", "make_eval_sequences")

# ---------------------------------------------------------------------------
# Lift the two builders out of the runner's source, by AST, at module scope only.
# ---------------------------------------------------------------------------
_tree = ast.parse(RUNNER.read_text())
_defs = {n.name: n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name in WANTED}
missing = [w for w in WANTED if w not in _defs]
assert not missing, (
    f"{missing} are not module-level functions in {RUNNER.name}. They were nested inside main() "
    "until 21 Sep 2026; if they have moved back, this figure can no longer be derived from the "
    "code that produced the numbers and must not be drawn."
)
_ns = {"np": np}
exec(compile(ast.Module(body=list(_defs.values()), type_ignores=[]), str(RUNNER), "exec"), _ns)
make_train_sequences = _ns["make_train_sequences"]
make_eval_sequences = _ns["make_eval_sequences"]

# ---------------------------------------------------------------------------
# Run them on the identity series, so the returned values are timestep indices.
# ---------------------------------------------------------------------------
CTX = 96          # context_length, fixed for every cell in the paper
H = 96            # the horizon drawn; h=192 is the same layout with h substituted
T = CTX + 2 * H   # exactly long enough for one evaluation window
series = np.arange(T).reshape(T, 1)

Xtr, ytr = make_train_sequences(series, CTX, H)
Xev, yev = make_eval_sequences(series, CTX + H, H)

# The claim the figure exists to support, checked on both window sets before drawing.
for name, X, y, n_in in (("train", Xtr, ytr, CTX), ("eval", Xev, yev, CTX + H)):
    assert X.shape[1] == n_in, f"{name} input is {X.shape[1]} steps, expected {n_in}"
    assert y.shape[1] == H, f"{name} target is {y.shape[1]} steps, expected {H}"
    assert y[0][0, 0] == X[0][-1, 0] + 1, (
        f"{name}: target starts at {y[0][0, 0]} but the input ends at {X[0][-1, 0]}; the windows are "
        "no longer disjoint and consecutive, which is the whole content of this figure"
    )
    assert not (set(X[0].ravel().tolist()) & set(y[0].ravel().tolist())), \
        f"{name}: a target timestep appears in the model input"

# Index arithmetic, read off the arrays rather than typed.
tr_in = (int(Xtr[0][0, 0]), int(Xtr[0][-1, 0]))
tr_tg = (int(ytr[0][0, 0]), int(ytr[0][-1, 0]))
ev_in = (int(Xev[0][0, 0]), int(Xev[0][-1, 0]))
ev_tg = (int(yev[0][0, 0]), int(yev[0][-1, 0]))
SPAN = ev_tg[1] + 1  # the widest timeline either row needs

fig, ax = plt.subplots(figsize=(5.5, 2.35), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

X0, XW = 24.0, 73.0          # the timeline occupies [X0, X0+XW]; labels live to its left
BARH = 12.5


def xt(t):
    """Timestep -> figure x. One shared mapping, so the two rows are literally to scale."""
    return X0 + XW * t / SPAN


def bar(t0, t1, cy, ec, fc, label, sub, hatch=None):
    x0, x1 = xt(t0), xt(t1 + 1)
    ax.add_patch(FancyBboxPatch((x0, cy - BARH / 2), x1 - x0, BARH,
                                boxstyle="round,pad=0.35,rounding_size=1.4",
                                linewidth=0.85, edgecolor=ec, facecolor=fc,
                                hatch=hatch, zorder=3))
    ax.text((x0 + x1) / 2, cy + 2.3, label, fontsize=6.0, color=ec, ha="center", va="center",
            fontweight="bold", zorder=4,
            bbox=dict(facecolor=fc, edgecolor="none", pad=1.2) if hatch else None)
    ax.text((x0 + x1) / 2, cy - 3.1, sub, fontsize=5.3, color=INK_SECONDARY, ha="center",
            va="center", zorder=4,
            bbox=dict(facecolor=fc, edgecolor="none", pad=1.2) if hatch else None)


ROW = {"train": 86.0, "eval": 62.0}

# Row captions: the function name, in monospace, because that is what a reader greps for.
for key, title, fn in (("train", "training (NLL)", "make_train_sequences"),
                       ("eval", "evaluation (forward)", "make_eval_sequences")):
    ax.text(22.0, ROW[key] + 3.0, title, fontsize=6.6, color=INK_PRIMARY, ha="right", va="center")
    ax.text(22.0, ROW[key] - 2.4, fn, fontsize=5.2, color=INK_MUTED, ha="right", va="center",
            family="monospace")

def span(a, b):
    return f"$x_{{i}}\\,{{\\ldots}}\\,x_{{i+{b}}}$" if a == 0 else \
           f"$x_{{i+{a}}}\\,{{\\ldots}}\\,x_{{i+{b}}}$"


# The index range goes on the SECOND line of each bar, not the first: a target bar is h/(96+2h) of the
# timeline wide, and "forecast target x_{i+192}...x_{i+287}" on one line overruns it in both directions.
bar(*tr_in, ROW["train"], NEUTRAL, NEUTRAL_F,
    f"context, {CTX} steps", span(*tr_in))
# Condition-B training hands _val_loss one concatenated tensor, and the target half of it is where the
# leakage question actually bites -- so the masking is on the figure, not only in the prose.
bar(*tr_tg, ROW["train"], ADAPT, ADAPT_F,
    "target, masked", span(*tr_tg), hatch="////")

bar(*ev_in, ROW["eval"], NEUTRAL, NEUTRAL_F,
    f"past_target: {CTX}$+h$ = {ev_in[1] + 1} steps of HISTORY", span(*ev_in))
bar(*ev_tg, ROW["eval"], HELDOUT, HELDOUT_F,
    "forecast target", span(*ev_tg))

# The boundary that answers the question. The leader stops at the bar's lower edge: drawn through the
# bar it lands on top of the words "forecast target", which is the one label that must stay legible.
bx = xt(ev_tg[0])
ax.plot([bx, bx], [ROW["eval"] - BARH / 2 - 4.2, ROW["eval"] - BARH / 2],
        color=INK_PRIMARY, lw=0.9, zorder=5)
ax.text(bx, ROW["eval"] - BARH / 2 - 6.9,
        f"input ends at $i{{+}}{ev_in[1]}$, target begins at $i{{+}}{ev_tg[0]}$: disjoint and "
        "consecutive",
        fontsize=5.5, color=INK_PRIMARY, ha="center", va="center")

# The shared timeline, so "extended lookback" is visibly a longer history rather than a longer window.
ax.add_patch(FancyArrowPatch((X0, 35.0), (X0 + XW + 1.5, 35.0), arrowstyle="-|>",
                             mutation_scale=6, linewidth=0.7, color=INK_MUTED, zorder=2))
for t in (0, CTX, ev_tg[0], SPAN - 1):
    ax.plot([xt(t), xt(t)], [34.0, 36.0], color=INK_MUTED, lw=0.7, zorder=2)
    ax.text(xt(t), 31.2, f"$i{{+}}{t}$" if t else "$i$", fontsize=5.2, color=INK_MUTED,
            ha="center", va="center")
ax.text(X0 + XW + 2.5, 35.0, "time", fontsize=5.4, color=INK_MUTED, ha="left", va="center")

ax.plot([23.0, 23.0], [39.0, 94.0], color=GRID, lw=0.8, zorder=1)
ax.text(1.0, 15.0,
        "Both rows are the index arrays the two builders return for window $i$, drawn to one shared "
        "scale.  At evaluation the whole\ninput is passed as past_target with past_observed_target "
        "all ones, and no target value is anywhere in it.  Training's two\nhalves are concatenated "
        "into a single $96{+}h$ tensor, but _val_loss splits it at context_length and the module "
        "substitutes\nits learned mask_encoding at every target position, so the target is scored by "
        "the likelihood and never conditioned on.",
        fontsize=5.5, color=INK_SECONDARY, ha="left", va="center", linespacing=1.35)

out = save(fig, Path(__file__).with_name("figA_window_layout.png"))
print(f"wrote {out} and .pdf  (h={H}: eval input i..i+{ev_in[1]}, target i+{ev_tg[0]}..i+{ev_tg[1]}, "
      f"disjointness asserted on both window sets)")
