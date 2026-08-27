"""Analyse per-epoch trajectories at n=5000 to characterise the bimodality at the crossover regime.

Reads all condition B, h=96, n=5000 JSONs across results/, classifies seeds by final forgetting%,
and plots per-epoch val_mse and weight_drift trajectories colored by mode.
Saves figure to paper_8/figures/n5k_trajectories.pdf and prints a mode-separator statistic.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULT_GLOB = "/Users/mediratta/code/iotsf_demo/results/*/condition_B_h96_s*.json"
OUT_PDF = "/Users/mediratta/code/iotsf_demo/paper_8/figures/n5k_trajectories.pdf"


def load_n5k_seeds():
    runs = []
    for p in glob.glob(RESULT_GLOB):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if d.get("max_train_samples") == 5000 and d.get("condition") == "B" and d.get("horizon") == 96:
            runs.append(d)
    runs.sort(key=lambda d: d["seed"])
    return runs


def classify(forg):
    # Split at median-ish boundary; the distribution is long-tailed rather than strictly bimodal.
    return "low" if forg <= 5.0 else "high"


def main():
    runs = load_n5k_seeds()
    if not runs:
        sys.exit("no n=5000 condition B h=96 runs found")

    print(f"{'seed':>6} {'forg%':>8} {'cka':>6} {'wd':>6} mode")
    for d in runs:
        print(f"{d['seed']:>6} {d['forgetting_pct']:+8.2f} {d['final_cka']:6.3f} {d['final_weight_drift']:6.2f} {classify(d['forgetting_pct']):>4}")

    zs_mse = runs[0]["zeroshot_mse"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
    color_low = "#1f77b4"
    color_high = "#d62728"

    # Panel 1: val_mse trajectories
    ax1 = axes[0]
    for d in runs:
        hist = d["history"]
        epochs = list(range(len(hist["val_mse"])))
        c = color_low if classify(d["forgetting_pct"]) == "low" else color_high
        ax1.plot(epochs, hist["val_mse"], color=c, alpha=0.8, linewidth=1.2,
                 label=f"s{d['seed']} ({d['forgetting_pct']:+.1f}%)")
    ax1.axhline(zs_mse, color="gray", linestyle="--", linewidth=1.0, label=f"ZS MSE ({zs_mse:.3f})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val MSE")
    ax1.set_title(r"Val MSE trajectory, $n{=}5{,}000$")
    ax1.legend(fontsize=6, loc="upper right", ncol=2)

    # Panel 2: weight_drift trajectories — labelled per seed to match reviewer request
    ax2 = axes[1]
    for d in runs:
        hist = d["history"]
        wd = hist.get("weight_drift", [])
        epochs = list(range(len(wd)))
        c = color_low if classify(d["forgetting_pct"]) == "low" else color_high
        ax2.plot(epochs, wd, color=c, alpha=0.8, linewidth=1.2,
                 label=f"s{d['seed']} ({d['forgetting_pct']:+.1f}%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"Weight drift $\|\theta_t - \theta_0\|_2$")
    ax2.set_title(r"Weight drift trajectory, $n{=}5{,}000$")
    ax2.legend(fontsize=6, loc="lower right", ncol=2)

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nsaved: {OUT_PDF}")

    # Separator statistics
    print("\n--- mode separator diagnostics ---")
    for stat_name in ("early_val_mse", "mid_weight_drift", "val_mse_min_epoch"):
        vals_low = []
        vals_high = []
        for d in runs:
            hist = d["history"]
            if stat_name == "early_val_mse":
                v = float(np.mean(hist["val_mse"][1:4]))  # epochs 1–3
            elif stat_name == "mid_weight_drift":
                wd = hist.get("weight_drift", [])
                v = float(np.mean(wd[5:11])) if len(wd) >= 11 else float("nan")
            elif stat_name == "val_mse_min_epoch":
                v = int(np.argmin(hist["val_mse"]))
            mode = classify(d["forgetting_pct"])
            (vals_low if mode == "low" else vals_high).append(v)
        print(f"{stat_name:>25}: low={np.mean(vals_low):.4f}±{np.std(vals_low):.4f}  high={np.mean(vals_high):.4f}±{np.std(vals_high):.4f}")


if __name__ == "__main__":
    main()
