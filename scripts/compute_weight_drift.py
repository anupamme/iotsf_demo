#!/usr/bin/env python3
"""
Compute encoder weight drift (cosine distance, Frobenius norm, CKA) between
pre-trained and fine-tuned Moirai at different training scales and conditions.

Addresses reviewer concern MC3: direct probes of forgetting rather than
inferring it from encoder freezing.

Output: results/weight_drift.json
"""

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.moirai_detector import MoiraiAnomalyDetector


def cosine_distance_per_layer(state_init, state_final):
    """Per-layer cosine distance: 1 - cos(w_init, w_final)."""
    results = {}
    for name in state_init:
        if name not in state_final:
            continue
        w0 = state_init[name].flatten().float()
        w1 = state_final[name].flatten().float()
        cos_sim = F.cosine_similarity(w0.unsqueeze(0), w1.unsqueeze(0)).item()
        results[name] = {
            "cosine_distance": 1.0 - cos_sim,
            "frobenius_ratio": torch.norm(w1 - w0).item() / (torch.norm(w0).item() + 1e-12),
        }
    return results


def aggregate_drift(per_layer):
    """Aggregate per-layer drift into summary statistics."""
    cos_dists = [v["cosine_distance"] for v in per_layer.values()]
    frob_ratios = [v["frobenius_ratio"] for v in per_layer.values()]
    return {
        "mean_cosine_distance": float(np.mean(cos_dists)),
        "max_cosine_distance": float(np.max(cos_dists)),
        "mean_frobenius_ratio": float(np.mean(frob_ratios)),
        "max_frobenius_ratio": float(np.max(frob_ratios)),
        "n_layers": len(cos_dists),
    }


def linear_cka(X, Y):
    """
    Linear CKA between two representation matrices.
    X, Y: (n_samples, d) numpy arrays
    Returns scalar CKA similarity in [0, 1].
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, ord="fro")
    hsic_yy = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(hsic_xy / (hsic_xx * hsic_yy + 1e-12))


def get_raw_encoder_representations(detector, probe_data):
    """Extract raw encoder output (no projection head) via forward hook.

    Uses the same MoiraiSupervisedDataset + _val_loss pattern as get_embeddings(),
    but skips the projection head — works even pre-fine-tuning.
    """
    from src.data.torch_dataset import MoiraiSupervisedDataset
    from torch.utils.data import DataLoader

    encoder = detector._get_encoder()
    captured = {}

    def hook_fn(module, inp, out):
        captured["output"] = out.detach()

    handle = encoder.register_forward_hook(hook_fn)

    dummy_labels = np.zeros(len(probe_data), dtype=np.int64)
    dataset = MoiraiSupervisedDataset(probe_data, dummy_labels,
                                       context_length=detector.context_length)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    patch_size = detector.patch_size if detector.patch_size != 'auto' else 32
    detector.model.eval()
    all_repr = []

    try:
        with torch.no_grad():
            for batch in loader:
                context = batch['context'].to(detector.device)
                target = batch['target'].to(detector.device)
                full_target = torch.cat([context, target], dim=1)
                B, seq_len, n_features = full_target.shape
                observed_target = torch.ones(B, seq_len, n_features,
                                             dtype=torch.bool, device=detector.device)
                is_pad = torch.zeros(B, seq_len, dtype=torch.bool, device=detector.device)

                _ = detector.model._val_loss(
                    patch_size=patch_size,
                    target=full_target,
                    observed_target=observed_target,
                    is_pad=is_pad,
                )
                if "output" in captured:
                    pooled = captured["output"].mean(dim=1).cpu().numpy()
                    all_repr.append(pooled)
    finally:
        handle.remove()

    if not all_repr:
        raise RuntimeError("Forward hook did not capture encoder output")
    return np.concatenate(all_repr, axis=0)


def load_condition_data(condition, synthetic_dir, max_samples, seed=42):
    """Load training data for a given condition."""
    from src.models.hard_negative_generator import HardNegativeGenerator

    benign = np.load(Path(synthetic_dir) / "benign_samples.npy")

    attack_files = sorted(Path(synthetic_dir).glob("*_stealth_*.npy"))
    attacks = []
    for f in attack_files:
        attacks.append(np.load(f))

    if not attacks:
        logger.warning(f"No attack files found in {synthetic_dir}")
        return benign[:max_samples], np.zeros(min(len(benign), max_samples))

    all_attacks = np.concatenate(attacks)

    if condition == "c":
        rng = np.random.default_rng(seed)
        noise_neg = benign + rng.normal(0, 0.3, benign.shape)
        X = np.concatenate([benign, noise_neg])
        y = np.array([0] * len(benign) + [1] * len(noise_neg))
    else:
        X = np.concatenate([benign, all_attacks])
        y = np.array([0] * len(benign) + [1] * len(all_attacks))

    if max_samples and len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    return X, y


def run_single(condition, scale, epochs, synthetic_dir, probe_data, pretrained_state, seed=42):
    """Fine-tune and measure drift for one (condition, scale) pair."""
    logger.info(f"Running {condition} at {scale}/{epochs}")
    t0 = time.time()

    X, y = load_condition_data(condition, synthetic_dir, max_samples=scale, seed=seed)

    rng = np.random.default_rng(seed + 1000)
    perm = rng.permutation(len(X))
    n_val = max(4, int(0.15 * len(X)))
    train_X, train_y = X[perm[n_val:]], y[perm[n_val:]]
    val_X, val_y = X[perm[:n_val]], y[perm[:n_val]]

    detector = MoiraiAnomalyDetector(model_size="small", context_length=96, prediction_length=32)
    detector.initialize()

    if detector._mock_mode:
        logger.error("Running in mock mode — results will be meaningless")
        return None

    history = detector.fine_tune_supervised(
        train_data=train_X, train_labels=train_y,
        val_data=val_X, val_labels=val_y,
        n_epochs=epochs, batch_size=32, learning_rate=1e-4,
        contrastive_weight=0.5,
    )

    encoder = detector._get_encoder()
    finetuned_state = {name: param.clone().detach().cpu()
                       for name, param in encoder.named_parameters()}

    per_layer = cosine_distance_per_layer(pretrained_state, finetuned_state)
    summary = aggregate_drift(per_layer)

    repr_ft = None
    try:
        repr_ft = get_raw_encoder_representations(detector, probe_data)
    except Exception as e:
        logger.warning(f"Could not extract representations: {e}")

    best_epoch = history.get("best_epoch", None) if history else None
    stopped_epoch = history.get("stopped_epoch", None) if history else None

    elapsed = time.time() - t0
    logger.info(f"  {condition}/{scale}: mean_cos_dist={summary['mean_cosine_distance']:.6f}, "
                f"max_cos_dist={summary['max_cosine_distance']:.6f}, "
                f"best_epoch={best_epoch}, stopped_epoch={stopped_epoch}, elapsed={elapsed:.0f}s")

    return {
        "condition": condition,
        "scale": scale,
        "epochs": epochs,
        "summary": summary,
        "per_layer": {k: v for k, v in per_layer.items()},
        "elapsed_s": elapsed,
        "representations": repr_ft,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure encoder weight drift during fine-tuning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/weight_drift.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Loading pre-trained Moirai (no fine-tuning) for baseline weights...")
    detector_pre = MoiraiAnomalyDetector(model_size="small", context_length=96, prediction_length=32)
    detector_pre.initialize()

    if detector_pre._mock_mode:
        logger.error("uni2ts not available — cannot run weight drift analysis in mock mode")
        sys.exit(1)

    encoder_pre = detector_pre._get_encoder()
    pretrained_state = {name: param.clone().detach().cpu()
                        for name, param in encoder_pre.named_parameters()}
    logger.info(f"Captured pre-trained state: {len(pretrained_state)} parameter tensors")

    probe_benign = np.load(ROOT / "data" / "synthetic" / "benign_samples.npy")[:50]
    logger.info(f"Probe data: {probe_benign.shape}")

    pretrained_repr = None
    try:
        pretrained_repr = get_raw_encoder_representations(detector_pre, probe_benign)
        logger.info(f"Pre-trained representations: {pretrained_repr.shape}")
    except Exception as e:
        logger.warning(f"Could not get pre-trained representations: {e}")

    conditions_configs = [
        # (condition, scale, epochs, synthetic_dir)
        ("c", 200, 5, "data/synthetic"),
        ("d", 200, 5, "data/synthetic"),
        ("d", 200, 5, "data/synthetic_diffts"),  # D-DiffTS at 200/5

        ("c", 500, 10, "data/synthetic_500"),
        ("d", 500, 10, "data/synthetic_500"),
        ("d", 500, 10, "data/synthetic_diffts_500"),  # D-DiffTS at 500/10

        ("c", 1000, 20, "data/synthetic_1k"),
        ("d", 1000, 20, "data/synthetic_1k"),
        ("d", 1000, 20, "data/synthetic_diffts_1k"),  # D-DiffTS at 1000/20
    ]

    all_results = []
    cka_results = []

    for cond, scale, epochs, syn_dir in conditions_configs:
        syn_path = ROOT / syn_dir
        if not syn_path.exists():
            logger.warning(f"Synthetic dir not found: {syn_path}, skipping")
            continue

        label = f"{cond}" if "diffts" not in syn_dir else "d-diffts"
        result = run_single(cond, scale, epochs, str(syn_path), probe_benign, pretrained_state, seed=args.seed)
        if result is None:
            continue

        result["label"] = label
        repr_ft = result.pop("representations", None)

        all_results.append(result)

        if pretrained_repr is not None and repr_ft is not None:
            cka_val = linear_cka(pretrained_repr, repr_ft)
            cka_results.append({
                "label": label, "scale": scale,
                "cka": cka_val,
            })
            logger.info(f"  CKA({label}/{scale}) = {cka_val:.4f}")

    # Clean per-layer for JSON serialization
    for r in all_results:
        r["per_layer"] = {k: {kk: float(vv) for kk, vv in v.items()}
                          for k, v in r["per_layer"].items()}

    output = {
        "seed": args.seed,
        "n_pretrained_params": len(pretrained_state),
        "probe_size": len(probe_benign),
        "drift_measurements": all_results,
        "cka_measurements": cka_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.success(f"Weight drift results saved to {out_path}")

    # Print summary table
    print("\n--- Weight Drift Summary ---")
    print(f"{'Condition':<12} {'Scale':<8} {'Mean Cos Dist':<14} {'Max Cos Dist':<14} {'Mean Frob':<12}")
    for r in all_results:
        s = r["summary"]
        print(f"{r['label']:<12} {r['scale']:<8} {s['mean_cosine_distance']:<14.6f} "
              f"{s['max_cosine_distance']:<14.6f} {s['mean_frobenius_ratio']:<12.6f}")

    if cka_results:
        print("\n--- CKA Similarity (pre-trained → fine-tuned) ---")
        for c in cka_results:
            print(f"  {c['label']}/{c['scale']}: CKA = {c['cka']:.4f}")


if __name__ == "__main__":
    main()
