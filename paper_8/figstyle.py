#!/usr/bin/env python3
"""
Shared ink, markers and row loading for this paper's figures.

WHY THIS FILE EXISTS. Figure 1 defined the palette inline, and the two appendix figures added on
20 Sep 2026 need the same palette to mean the same thing: green is "encoder adaptation helped",
red is "freezing did better", blue is "held-out rather than selection". A reader who learns that
code on page 2 and meets a different one in the appendix has been taught something false. Copying
the hex values into three files would have made that divergence a matter of time, so they live here
and every figure imports them.

Importing this module has no side effect other than the rcParams update, which is the same update
all three figures made separately before. fig1_diagnostic_flow.pdf is byte-identical across the
refactor; if it ever is not, the constants below have drifted from what that figure was drawn with.

SIZED FOR 1:1 PLACEMENT. The ICLR textwidth is 5.5in, so every figure here is authored at 5.5in
and included at width=\\linewidth. Do not shrink them in LaTeX -- the fontsizes are print sizes.
"""
import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt

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
N_SCORED = 31    # cells with a selection-split gate -- the denominator of every pass/fail count

MARKER = {"Moirai": "o", "Chronos": "s", "TimesFM": "^"}


def backbone(r):
    c = r["cell"]
    return "TimesFM" if c.startswith("TimesFM") else ("Chronos" if c.startswith("Chronos") else "Moirai")


def intervention_rows():
    """The 31 cells with both an intervention outcome and a CKA -- the denominator of every figure.

    build_rows() prints a progress log to stdout, which would land in the middle of a figure
    script's own output; redirected here rather than at each call site.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        rows = [r for r in cell_matrix.build_rows()
                if r["bd_test"] is not None and r["cka"] is not None]
    assert len(rows) == N_SCORED, f"expected {N_SCORED} intervention cells, got {len(rows)}"
    return rows


def save(fig, path):
    """Write the .png a human looks at and the .pdf LaTeX includes, from one figure."""
    out = str(path)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.01, facecolor=SURFACE)
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01,
                facecolor=SURFACE)
    return out
