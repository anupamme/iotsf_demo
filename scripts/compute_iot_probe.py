#!/usr/bin/env python3
"""
Linear probing on IoT representations (V9 rebuttal experiment 1B).

For each seed:
  1) Load pre-trained Moirai, extract raw encoder reps on benign+attack eval set.
  2) Fine-tune with condition-C config (NLL + SupCon, Gaussian-noise negatives).
  3) Extract post-FT encoder reps on the same eval set.
  4) Fit sklearn LogisticRegression on 50% (probe-train), evaluate ROC-AUC on 50% (probe-val).
  5) Report pre/post probe AUC, delta, and eval CKA between pre-FT and post-FT reps.

This is the IoT analogue of `linear_probe_r2` in finetune_forecasting.py and tests
whether the CKA/probe dissociation observed on ETTh2 also holds on IoT.
"""
import argparse
import json
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.moirai_detector import MoiraiAnomalyDetector
from src.data.torch_dataset import MoiraiSupervisedDataset
from torch.utils.data import DataLoader


def extract_encoder_reps(detector, X, batch_size=16):
    """Extract mean-pooled raw encoder reps for every sample in X.

    Returns:
        reps: (n, d_model) numpy array.
    """
    detector.model.eval()
    captured = {}

    def hook(module, input, output):
        captured['encoder'] = output

    handle = detector._get_encoder().register_forward_hook(hook)
    dummy_labels = np.zeros(len(X), dtype=np.int64)
    dataset = MoiraiSupervisedDataset(X, dummy_labels, context_length=detector.context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    patch_size = detector.patch_size if detector.patch_size != 'auto' else 32
    all_reps = []
    try:
        with torch.no_grad():
            for batch in loader:
                context = batch['context'].to(detector.device)
                target = batch['target'].to(detector.device)
                full = torch.cat([context, target], dim=1)
                B, seq_len, n_feat = full.shape
                observed = torch.ones(B, seq_len, n_feat, dtype=torch.bool, device=detector.device)
                is_pad = torch.zeros(B, seq_len, dtype=torch.bool, device=detector.device)
                try:
                    _ = detector._safe_val_loss(
                        patch_size=patch_size, target=full,
                        observed_target=observed, is_pad=is_pad,
                    )
                except Exception as e:
                    logger.warning(f"val_loss error (batch): {e}")
                    continue
                rep = captured.get('encoder')
                if rep is None:
                    continue
                if isinstance(rep, tuple):
                    rep = rep[0]
                pooled = rep.mean(dim=1).cpu().numpy()  # (B, d_model)
                all_reps.append(pooled)
    finally:
        handle.remove()

    if not all_reps:
        raise RuntimeError("No encoder reps captured")
    return np.concatenate(all_reps, axis=0)


def linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    num = float((X.T @ Y).astype(np.float64).__pow__(2).sum())
    den = float(np.sqrt(((X.T @ X) ** 2).sum() * ((Y.T @ Y) ** 2).sum()))
    return num / den if den > 1e-12 else 0.0


def probe_auc(reps, labels, rng):
    """50/50 split, fit LogReg, return ROC-AUC on held-out half."""
    n = len(labels)
    idx = rng.permutation(n)
    half = n // 2
    tr, va = idx[:half], idx[half:]
    X_tr, y_tr = reps[tr], labels[tr]
    X_va, y_va = reps[va], labels[va]
    # Standardise (logistic regression is scale-sensitive)
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd
    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    if len(np.unique(y_tr)) < 2:
        return 0.5
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_va)[:, 1]
    if len(np.unique(y_va)) < 2:
        return 0.5
    return float(roc_auc_score(y_va, proba))


def load_probe_eval_set(synthetic_dir, max_per_stealth=50, rng=None):
    """Build a balanced probe eval set: benign_eval + attacks (all stealth levels)."""
    synth = Path(synthetic_dir)
    rng = rng or np.random.default_rng(0)
    benign = np.load(synth / 'benign_samples.npy')
    attack_files = sorted(synth.glob('*_stealth_*.npy'))
    attack_chunks = []
    for fp in attack_files:
        arr = np.load(fp)
        n = min(len(arr), max_per_stealth)
        idx = rng.choice(len(arr), size=n, replace=False)
        attack_chunks.append(arr[idx])
    attacks = np.concatenate(attack_chunks)
    # Use ~80% of benign for the probe eval (leave 20% untouched for train)
    n_b = min(len(benign), len(attacks) * 2)
    b_idx = rng.choice(len(benign), size=n_b, replace=False)
    X = np.concatenate([benign[b_idx], attacks])
    y = np.array([0] * n_b + [1] * len(attacks))
    return X.astype(np.float32), y


def fine_tune_condition_c(detector, synthetic_dir, epochs, batch_size, lr, seed):
    """Condition C: NLL+SupCon with Gaussian-noise negatives (mirrors run_ablation.py)."""
    synth = Path(synthetic_dir)
    benign = np.load(synth / 'benign_samples.npy')
    rng_ft = np.random.default_rng(0)
    attacks = benign + rng_ft.normal(0, 0.3, benign.shape)  # Gaussian-noise negatives
    all_X = np.concatenate([benign, attacks]).astype(np.float32)
    all_y = np.array([0] * len(benign) + [1] * len(attacks))
    rng_split = np.random.default_rng(1)
    perm = rng_split.permutation(len(all_X))
    n_val = max(4, int(0.15 * len(all_X)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_data, train_labels = all_X[train_idx], all_y[train_idx]
    val_data, val_labels = all_X[val_idx], all_y[val_idx]
    ckpt_dir = tempfile.mkdtemp(prefix=f'moirai_probe_seed{seed}_')
    detector.fine_tune_supervised(
        train_data=train_data, train_labels=train_labels,
        val_data=val_data, val_labels=val_labels,
        n_epochs=epochs, batch_size=batch_size, learning_rate=lr,
        contrastive_weight=0.5, early_stopping_criterion='nll',
        freeze_encoder='none', checkpoint_dir=ckpt_dir,
    )


def run_one_seed(seed, args):
    logger.info(f"=== seed {seed} ===")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    X_eval, y_eval = load_probe_eval_set(args.synthetic_dir, max_per_stealth=args.max_per_stealth, rng=rng)
    logger.info(f"Probe eval set: {len(X_eval)} samples ({(y_eval == 0).sum()} benign, {(y_eval == 1).sum()} attack)")

    # 1) Pre-trained encoder reps
    logger.info("Loading pre-trained Moirai...")
    det_pre = MoiraiAnomalyDetector(
        model_size=args.model_size, context_length=96, prediction_length=32,
        confidence_level=0.95, device=args.device,
    )
    det_pre.initialize()
    reps_pre = extract_encoder_reps(det_pre, X_eval, batch_size=args.batch_size)
    logger.info(f"pre-FT reps: {reps_pre.shape}")
    auc_pre = probe_auc(reps_pre, y_eval, np.random.default_rng(seed + 10000))

    # 2) Fine-tune, then extract post-FT reps
    logger.info("Fine-tuning (condition C)...")
    det_ft = MoiraiAnomalyDetector(
        model_size=args.model_size, context_length=96, prediction_length=32,
        confidence_level=0.95, device=args.device,
    )
    det_ft.initialize()
    fine_tune_condition_c(det_ft, args.synthetic_dir, args.epochs, args.batch_size, args.lr, seed)
    reps_post = extract_encoder_reps(det_ft, X_eval, batch_size=args.batch_size)
    logger.info(f"post-FT reps: {reps_post.shape}")
    auc_post = probe_auc(reps_post, y_eval, np.random.default_rng(seed + 10000))

    cka = linear_cka(reps_pre, reps_post)
    return {
        'seed': seed,
        'n_eval_benign': int((y_eval == 0).sum()),
        'n_eval_attack': int((y_eval == 1).sum()),
        'd_model': int(reps_pre.shape[1]),
        'probe_auc_pre': auc_pre,
        'probe_auc_post': auc_post,
        'probe_auc_delta': auc_post - auc_pre,
        'cka': cka,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-dir', default='data/synthetic_diffts_500')
    parser.add_argument('--model-size', default='small')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seeds', default='42,123,456')
    parser.add_argument('--max-per-stealth', type=int, default=50)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--results-dir', default='results/v9_iot_probe')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for s in seeds:
        t0 = time.time()
        res = run_one_seed(s, args)
        res['elapsed_s'] = time.time() - t0
        all_results.append(res)
        # Save per-seed JSON eagerly
        with open(out_dir / f'seed_{s}.json', 'w') as f:
            json.dump(res, f, indent=2)
        logger.success(f"seed {s}: AUC {res['probe_auc_pre']:.3f} → {res['probe_auc_post']:.3f} (Δ {res['probe_auc_delta']:+.3f}), CKA={res['cka']:.3f}")

    # Aggregate
    summary = {
        'seeds': seeds,
        'per_seed': all_results,
        'probe_auc_pre_mean': float(np.mean([r['probe_auc_pre'] for r in all_results])),
        'probe_auc_pre_std': float(np.std([r['probe_auc_pre'] for r in all_results])),
        'probe_auc_post_mean': float(np.mean([r['probe_auc_post'] for r in all_results])),
        'probe_auc_post_std': float(np.std([r['probe_auc_post'] for r in all_results])),
        'probe_auc_delta_mean': float(np.mean([r['probe_auc_delta'] for r in all_results])),
        'probe_auc_delta_std': float(np.std([r['probe_auc_delta'] for r in all_results])),
        'cka_mean': float(np.mean([r['cka'] for r in all_results])),
        'cka_std': float(np.std([r['cka'] for r in all_results])),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("\n=== SUMMARY ===")
    logger.info(f"probe AUC pre:   {summary['probe_auc_pre_mean']:.3f} ± {summary['probe_auc_pre_std']:.3f}")
    logger.info(f"probe AUC post:  {summary['probe_auc_post_mean']:.3f} ± {summary['probe_auc_post_std']:.3f}")
    logger.info(f"Δ probe AUC:     {summary['probe_auc_delta_mean']:+.3f} ± {summary['probe_auc_delta_std']:.3f}")
    logger.info(f"CKA (pre, post): {summary['cka_mean']:.3f} ± {summary['cka_std']:.3f}")


if __name__ == '__main__':
    main()
