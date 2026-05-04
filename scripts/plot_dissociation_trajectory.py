"""Three-panel trajectory figure for the dissociation story.

CKA, Ridge Delta-R^2, and forgetting % vs. sample size n (log-x).
Values are the headline numbers from Table 3 (sample_sweep.tex).
V17 ES-restored n=10k point is plotted with a distinct marker so the
protocol distinction is visible.

Output: paper_8/figures/dissociation_trajectory.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "paper_8" / "figures" / "dissociation_trajectory.pdf"

# Headline numbers from tables/sample_sweep.tex + appendix per-seed breakdown.
# n, CKA mean, CKA std, forg mean, forg std, dR2 mean, dR2 std, R2_FT mean
ROWS = [
    (500,   0.949, 0.010, +14.0, 6.7,  +0.12, 0.02, -6.79),
    (1000,  0.936, 0.025, +13.4, 9.1,  +0.13, 0.00, -6.77),
    (2000,  0.814, 0.042, -2.9,  5.9,  +0.39, 0.35, -6.51),
    (5000,  0.546, 0.082, +6.6,  10.0, +0.50, 0.26, -6.41),
]
# n=10k reported twice: V16 final-epoch vs V17 ES-restored.
V16_FE_N10K = dict(n=10000, cka=0.407, cka_sd=0.088, forg=+7.5,  forg_sd=8.2,
                    dR2=+0.78, dR2_sd=0.14, R2_FT=-6.12)
V17_ES_N10K = dict(n=10000, cka=0.461, cka_sd=0.232, forg=-11.7, forg_sd=5.3,
                    dR2=+0.56, dR2_sd=0.14, R2_FT=-6.34)


def build_axes():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0), dpi=150)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Fine-tuning samples $n$")
        ax.set_xticks([500, 1000, 2000, 5000, 10000])
        ax.set_xticklabels(["500", "1k", "2k", "5k", "10k"])
        ax.grid(True, which="both", ls=":", alpha=0.3)
    return fig, axes


def main():
    ns = [r[0] for r in ROWS]
    ckas = [r[1] for r in ROWS]
    cka_sds = [r[2] for r in ROWS]
    forgs = [r[3] for r in ROWS]
    forg_sds = [r[4] for r in ROWS]
    dR2s = [r[5] for r in ROWS]
    dR2_sds = [r[6] for r in ROWS]

    fig, (ax_cka, ax_dR2, ax_forg) = build_axes()

    # --- CKA panel ---
    ax_cka.errorbar(ns, ckas, yerr=cka_sds, fmt="o-", color="#1f77b4",
                    capsize=3, label="final-epoch protocol")
    ax_cka.errorbar([V16_FE_N10K["n"]], [V16_FE_N10K["cka"]],
                    yerr=[V16_FE_N10K["cka_sd"]],
                    fmt="o", color="#1f77b4", capsize=3)
    ax_cka.errorbar([V17_ES_N10K["n"]], [V17_ES_N10K["cka"]],
                    yerr=[V17_ES_N10K["cka_sd"]],
                    fmt="s", color="#d62728", capsize=3,
                    markersize=8, label="early-stopped (n=10k)")
    ax_cka.axhline(1.0, color="grey", lw=0.5, ls="--")
    ax_cka.set_ylabel("CKA (vs.\\ pre-trained)")
    ax_cka.set_title("Geometric drift: CKA $\\downarrow$")
    ax_cka.set_ylim(0.0, 1.05)
    ax_cka.legend(fontsize=8, loc="lower left")

    # --- Delta R^2 panel ---
    ax_dR2.errorbar(ns, dR2s, yerr=dR2_sds, fmt="o-", color="#1f77b4",
                    capsize=3, label="Ridge, final-epoch")
    ax_dR2.errorbar([V16_FE_N10K["n"]], [V16_FE_N10K["dR2"]],
                    yerr=[V16_FE_N10K["dR2_sd"]],
                    fmt="o", color="#1f77b4", capsize=3)
    ax_dR2.errorbar([V17_ES_N10K["n"]], [V17_ES_N10K["dR2"]],
                    yerr=[V17_ES_N10K["dR2_sd"]],
                    fmt="s", color="#d62728", capsize=3,
                    markersize=8, label="Ridge, early-stopped (n=10k)")
    ax_dR2.axhline(0.0, color="grey", lw=0.5, ls="--")
    ax_dR2.set_ylabel("$\\Delta R^2$")
    ax_dR2.set_title("Relative linear-decodability: $\\Delta R^2 \\uparrow$")
    ax_dR2.legend(fontsize=8, loc="lower right")

    # --- Forgetting panel ---
    ax_forg.errorbar(ns, forgs, yerr=forg_sds, fmt="o-", color="#1f77b4",
                     capsize=3, label="final-epoch protocol")
    ax_forg.errorbar([V16_FE_N10K["n"]], [V16_FE_N10K["forg"]],
                     yerr=[V16_FE_N10K["forg_sd"]],
                     fmt="o", color="#1f77b4", capsize=3)
    ax_forg.errorbar([V17_ES_N10K["n"]], [V17_ES_N10K["forg"]],
                     yerr=[V17_ES_N10K["forg_sd"]],
                     fmt="s", color="#d62728", capsize=3,
                     markersize=8, label="early-stopped (best-val-MSE)")
    ax_forg.axhline(0.0, color="grey", lw=0.5, ls="--")
    ax_forg.set_ylabel("Forgetting \\%")
    ax_forg.set_title("Task forgetting: non-monotonic")
    ax_forg.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
