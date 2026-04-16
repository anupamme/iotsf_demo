#!/usr/bin/env python3
"""
Generate t-SNE embedding visualization for Figure 5 of the NeurIPS paper.

Trains conditions C and D from scratch, extracts projection-head embeddings
from the stealth-95 test set, and produces a side-by-side t-SNE comparison
showing SupCon-collapse (C) vs. hard-negative rescue (D).

Usage:
    cd /Users/admin/code/iotsf_demo
    source .venv/bin/activate
    python scripts/generate_tsne.py
"""

import sys
import json
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from loguru import logger

# --- Config ---
SYNTHETIC_DIR = ROOT_DIR / "data" / "synthetic"
OUTPUT_DIR = ROOT_DIR / "results" / "figures"
PAPER_FIG_DIR = ROOT_DIR / "paper" / "figures"
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-4
CONTRASTIVE_WEIGHT = 0.5
MAX_EVAL_SAMPLES = 50
SEED = 42
TSNE_PERPLEXITY = 20
TSNE_SEED = 42


def load_test_data(rng):
    """Load stealth-95 test data: benign + all four attack types."""
    benign = np.load(SYNTHETIC_DIR / "benign_samples.npy")
    attack_types = [
        "slow_exfiltration", "lotl_mimicry", "beacon", "protocol_anomaly"
    ]
    attack_chunks = []
    attack_type_labels = []
    for at in attack_types:
        fp = SYNTHETIC_DIR / f"{at}_stealth_95.npy"
        if fp.exists():
            arr = np.load(fp)
            if len(arr) > MAX_EVAL_SAMPLES:
                idx = rng.choice(len(arr), size=MAX_EVAL_SAMPLES, replace=False)
                arr = arr[idx]
            attack_chunks.append(arr)
            attack_type_labels.extend([at] * len(arr))

    attacks = np.concatenate(attack_chunks)
    # Cap benign to match attack count
    n_benign = min(len(benign), len(attacks))
    b_idx = rng.choice(len(benign), size=n_benign, replace=False)
    benign_subset = benign[b_idx]

    X = np.concatenate([benign_subset, attacks])
    y = np.array([0] * n_benign + [1] * len(attacks))
    # Detailed label: 0=benign, 1-4=attack types
    type_map = {at: i + 1 for i, at in enumerate(attack_types)}
    detail_labels = (
        ["benign"] * n_benign + attack_type_labels
    )
    return X, y, detail_labels


def train_and_extract(condition_name: str, use_hard_negatives: bool, rng):
    """
    Train the detector for the given condition and return (embeddings, labels).
    """
    from src.models import MoiraiAnomalyDetector

    benign = np.load(SYNTHETIC_DIR / "benign_samples.npy")

    if use_hard_negatives:
        # Condition D: use all synthetic hard-negative files
        chunks = list((SYNTHETIC_DIR).glob("*_stealth_*.npy"))
        attacks = np.concatenate([np.load(p) for p in chunks])
    else:
        # Condition C: use noisy benign copies (Gaussian-noise negatives)
        rng_c = np.random.default_rng(1)
        attacks = benign + rng_c.normal(0, 0.3, benign.shape)

    all_X = np.concatenate([benign, attacks])
    all_y = np.array([0] * len(benign) + [1] * len(attacks))
    perm = rng.permutation(len(all_X))
    n_val = max(4, int(0.15 * len(all_X)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    logger.info(f"Training condition {condition_name} ({len(train_idx)} train, {n_val} val)...")

    det = MoiraiAnomalyDetector(
        model_size="small",
        context_length=96,
        prediction_length=32,
        confidence_level=0.95,
    )
    det.initialize()

    ft_method = getattr(det, "fine_tune_supervised", None)
    if ft_method is None:
        logger.error("fine_tune_supervised not available — is uni2ts installed?")
        raise RuntimeError("uni2ts not available")

    ft_method(
        train_data=all_X[train_idx],
        train_labels=all_y[train_idx],
        val_data=all_X[val_idx],
        val_labels=all_y[val_idx],
        n_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        contrastive_weight=CONTRASTIVE_WEIGHT,
    )

    logger.info(f"Extracting embeddings for condition {condition_name}...")
    X_test, y_test, detail_labels = load_test_data(np.random.default_rng(SEED))
    embeddings = det.get_embeddings(X_test)
    return embeddings, y_test, detail_labels


def generate_tsne_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(SEED)

    logger.info("=== Condition C: SupCon on Gaussian-noise negatives ===")
    emb_c, y_c, labels_c = train_and_extract("C", use_hard_negatives=False, rng=rng)

    rng = np.random.default_rng(SEED)  # re-seed for reproducibility
    logger.info("=== Condition D: SupCon on hard-negative attacks ===")
    emb_d, y_d, labels_d = train_and_extract("D", use_hard_negatives=True, rng=rng)

    # t-SNE projection
    logger.info("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=TSNE_SEED,
                max_iter=1000)
    proj_c = tsne.fit_transform(emb_c)

    tsne2 = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=TSNE_SEED,
                 max_iter=1000)
    proj_d = tsne2.fit_transform(emb_d)

    # Colour coding
    COLOR_MAP = {
        "benign": "#2ecc71",          # green
        "slow_exfiltration": "#e74c3c",  # red
        "lotl_mimicry": "#e67e22",    # orange
        "beacon": "#9b59b6",          # purple
        "protocol_anomaly": "#3498db", # blue
    }
    LABEL_MAP = {
        "benign": "Benign",
        "slow_exfiltration": "Slow Exfil",
        "lotl_mimicry": "LotL Mimicry",
        "beacon": "C2 Beacon",
        "protocol_anomaly": "Protocol Anomaly",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, proj, labels, title in [
        (axes[0], proj_c, labels_c, r"Condition C: SupCon on Gaussian-noise neg."),
        (axes[1], proj_d, labels_d, r"Condition D: $\mathbf{HNIDS}$ — SupCon on hard negatives"),
    ]:
        unique_labels = list(dict.fromkeys(labels))  # preserve order
        for lbl in unique_labels:
            mask = np.array([l == lbl for l in labels])
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                c=COLOR_MAP.get(lbl, "#7f8c8d"),
                label=LABEL_MAP.get(lbl, lbl),
                alpha=0.65,
                s=18,
                linewidths=0,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t-SNE dim 1", fontsize=9)
        ax.set_ylabel("t-SNE dim 2", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared legend below both panels
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=False)

    plt.suptitle(
        "t-SNE projection of projection-head embeddings (stealth-95 test data)",
        fontsize=11, y=1.01
    )
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "fig5_tsne_embeddings.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    logger.success(f"Saved: {out_path}")

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(out_path, PAPER_FIG_DIR / "fig5_tsne_embeddings.pdf")
    logger.success(f"Copied to: {PAPER_FIG_DIR / 'fig5_tsne_embeddings.pdf'}")

    plt.close(fig)


if __name__ == "__main__":
    generate_tsne_figure()
