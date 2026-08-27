"""Four-panel trajectory figure for the drift-utility case study.

Panel 1: CKA decay vs n (geometric drift)
Panel 2: Trained-head Ridge ΔR² vs n (rising)
Panel 3: Probe asymmetry at n=10k (trained head vs orthogonal probes)
Panel 4: Forgetting % vs n (non-monotonic)

Output: paper_8/figures/dissociation_trajectory.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "paper_8" / "figures" / "dissociation_trajectory.pdf"

# Headline numbers from tables/sample_sweep.tex + appendix per-seed breakdown.
ROWS = [
    (500,   0.949, 0.010, +14.0, 6.7,  +0.12, 0.02),
    (1000,  0.936, 0.025, +13.4, 9.1,  +0.13, 0.00),
    (2000,  0.814, 0.042, -2.9,  5.9,  +0.39, 0.35),
    (5000,  0.546, 0.082, +6.6,  10.0, +0.50, 0.26),
]
V16_FE_N10K = dict(n=10000, cka=0.407, cka_sd=0.088, forg=+7.5,  forg_sd=8.2,
                    dR2=+0.78, dR2_sd=0.14)
V17_ES_N10K = dict(n=10000, cka=0.461, cka_sd=0.232, forg=-11.7, forg_sd=5.3,
                    dR2=+0.56, dR2_sd=0.14)

# Orthogonal probe data (Appendix J.1)
ORTH_ETTH2 = {"trained": (+0.67, 0.23), "lag1": (-0.052, 0.041),
              "mean": (-0.033, 0.019), "var": (-0.048, 0.030)}
ORTH_ETTM2 = {"trained": (+6.43, 3.08), "lag1": (-0.052, 0.038),
              "mean": (-0.022, 0.014), "var": (-0.040, 0.029)}


def main():
    ns = [r[0] for r in ROWS]
    ckas = [r[1] for r in ROWS]
    cka_sds = [r[2] for r in ROWS]
    forgs = [r[3] for r in ROWS]
    forg_sds = [r[4] for r in ROWS]
    dR2s = [r[5] for r in ROWS]
    dR2_sds = [r[6] for r in ROWS]

    fig, (ax_cka, ax_dR2, ax_asym, ax_forg) = plt.subplots(
        1, 4, figsize=(14.0, 3.0), dpi=150)

    for ax in [ax_cka, ax_dR2, ax_forg]:
        ax.set_xscale("log")
        ax.set_xlabel("Fine-tuning samples $n$")
        ax.set_xticks([500, 1000, 2000, 5000, 10000])
        ax.set_xticklabels(["500", "1k", "2k", "5k", "10k"])
        ax.grid(True, which="both", ls=":", alpha=0.3)

    # --- Panel 1: CKA ---
    ax_cka.errorbar(ns, ckas, yerr=cka_sds, fmt="o-", color="#1f77b4",
                    capsize=3, label="final-epoch")
    ax_cka.errorbar([V16_FE_N10K["n"]], [V16_FE_N10K["cka"]],
                    yerr=[V16_FE_N10K["cka_sd"]],
                    fmt="o", color="#1f77b4", capsize=3)
    ax_cka.errorbar([V17_ES_N10K["n"]], [V17_ES_N10K["cka"]],
                    yerr=[V17_ES_N10K["cka_sd"]],
                    fmt="s", color="#d62728", capsize=3,
                    markersize=8, label="early-stopped")
    ax_cka.axhline(1.0, color="grey", lw=0.5, ls="--")
    ax_cka.set_ylabel("CKA (vs.\\ pre-trained)")
    ax_cka.set_title("(a) Geometric drift")
    ax_cka.set_ylim(0.0, 1.05)
    ax_cka.legend(fontsize=7, loc="lower left")

    # --- Panel 2: Trained-head ΔR² ---
    ax_dR2.errorbar(ns, dR2s, yerr=dR2_sds, fmt="o-", color="#1f77b4",
                    capsize=3, label="final-epoch")
    ax_dR2.errorbar([V16_FE_N10K["n"]], [V16_FE_N10K["dR2"]],
                    yerr=[V16_FE_N10K["dR2_sd"]],
                    fmt="o", color="#1f77b4", capsize=3)
    ax_dR2.errorbar([V17_ES_N10K["n"]], [V17_ES_N10K["dR2"]],
                    yerr=[V17_ES_N10K["dR2_sd"]],
                    fmt="s", color="#d62728", capsize=3,
                    markersize=8, label="early-stopped")
    ax_dR2.axhline(0.0, color="grey", lw=0.5, ls="--")
    ax_dR2.set_ylabel("$\\Delta R^2$ (trained head)")
    ax_dR2.set_title("(b) Trained-head $\\Delta R^2 \\uparrow$")
    ax_dR2.legend(fontsize=7, loc="lower right")

    # --- Panel 3: Probe asymmetry (dot plot) ---
    ax_asym.grid(True, which="both", ls=":", alpha=0.3)
    # ETTh2 probes
    x_h2 = [0, 1, 2, 3]
    vals_h2 = [ORTH_ETTH2["trained"][0], ORTH_ETTH2["lag1"][0],
               ORTH_ETTH2["mean"][0], ORTH_ETTH2["var"][0]]
    errs_h2 = [ORTH_ETTH2["trained"][1], ORTH_ETTH2["lag1"][1],
               ORTH_ETTH2["mean"][1], ORTH_ETTH2["var"][1]]
    colors_h2 = ["#1f77b4", "#2ca02c", "#2ca02c", "#2ca02c"]

    # ETTm2 probes (normalised: divide trained head by 10 for visual scale)
    vals_m2 = [ORTH_ETTM2["lag1"][0], ORTH_ETTM2["mean"][0], ORTH_ETTM2["var"][0]]
    errs_m2 = [ORTH_ETTM2["lag1"][1], ORTH_ETTM2["mean"][1], ORTH_ETTM2["var"][1]]

    for i, (v, e, c) in enumerate(zip(vals_h2, errs_h2, colors_h2)):
        ax_asym.errorbar([i - 0.12], [v], yerr=[e], fmt="o", color=c,
                         capsize=3, markersize=8,
                         label="ETTh2" if i == 0 else None)
    # ETTm2 orthogonal probes (offset slightly)
    for i, (v, e) in enumerate(zip(vals_m2, errs_m2)):
        ax_asym.errorbar([i + 1 + 0.12], [v], yerr=[e], fmt="^", color="#9467bd",
                         capsize=3, markersize=7,
                         label="ETTm2" if i == 0 else None)

    ax_asym.axhline(0.0, color="grey", lw=0.5, ls="--")
    ax_asym.set_xticks([0, 1, 2, 3])
    ax_asym.set_xticklabels(["96-step\n(trained)", "lag-1", "mean", "var"],
                            fontsize=7)
    ax_asym.set_ylabel("$\\Delta R^2$")
    ax_asym.set_title("(c) Probe asymmetry ($n{=}10$k)")
    ax_asym.set_xlabel("Probe target")
    ax_asym.legend(fontsize=7, loc="upper right")

    # --- Panel 4: Forgetting ---
    ax_forg.errorbar(ns, forgs, yerr=forg_sds, fmt="o-", color="#1f77b4",
                     capsize=3, label="final-epoch")
    ax_forg.errorbar([V16_FE_N10K["n"]], [V16_FE_N10K["forg"]],
                     yerr=[V16_FE_N10K["forg_sd"]],
                     fmt="o", color="#1f77b4", capsize=3)
    ax_forg.errorbar([V17_ES_N10K["n"]], [V17_ES_N10K["forg"]],
                     yerr=[V17_ES_N10K["forg_sd"]],
                     fmt="s", color="#d62728", capsize=3,
                     markersize=8, label="early-stopped")
    ax_forg.axhline(0.0, color="grey", lw=0.5, ls="--")
    ax_forg.set_ylabel("Forgetting \\%")
    ax_forg.set_title("(d) Task forgetting")
    ax_forg.legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
