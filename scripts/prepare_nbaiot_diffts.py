#!/usr/bin/env python3
"""
Prepare Diffusion-TS negatives for N-BaIoT evaluation.

1. Extract N-BaIoT benign windows to .npy
2. Train a Diffusion-TS model on the benign data
3. Generate benign base samples from the trained model
4. Apply attack perturbations + post-hoc guidance to create negatives
5. Save to data/synthetic_nbaiot_diffts/

Usage:
    python scripts/prepare_nbaiot_diffts.py
    python scripts/prepare_nbaiot_diffts.py --epochs 200 --skip-training  # use existing checkpoint
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.nbaiot_loader import load_nbaiot
from src.models import IoTDiffusionGenerator


ATTACK_TYPES = ["slow_exfiltration", "lotl_mimicry", "protocol_anomaly", "beacon"]
STEALTH_LEVELS = [0.85, 0.90, 0.95]


def extract_benign(data_dir, device, seed, max_samples=200000):
    """Extract and save N-BaIoT benign training windows."""
    X_train, X_val, _, _, _, _ = load_nbaiot(
        data_dir=data_dir,
        device=device,
        max_samples_per_class=max_samples,
        seq_length=128,
        seed=seed,
    )
    logger.info(f"N-BaIoT benign: {X_train.shape} train, {X_val.shape} val")
    return X_train


def train_diffts(benign_path, output_ckpt, epochs, batch_size, seed):
    """Train Diffusion-TS on N-BaIoT benign data."""
    from src.models.diffusion_ts_adapter import setup_diffusion_ts_path
    setup_diffusion_ts_path()
    from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(benign_path)
    logger.info(f"Training DiffTS on {data.shape} benign windows ({device})")

    seq_length, feature_dim = data.shape[1], data.shape[2]
    tensor_data = torch.tensor(data, dtype=torch.float32)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True, drop_last=True)

    model = Diffusion_TS(
        seq_length=seq_length, feature_size=feature_dim,
        n_layer_enc=3, n_layer_dec=6, d_model=256,
        timesteps=1000, sampling_timesteps=50,
        loss_type="l1", beta_schedule="cosine",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    Path(output_ckpt).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "loss": best_loss,
                "config": {"seq_length": seq_length, "feature_dim": feature_dim,
                           "n_layer_enc": 3, "n_layer_dec": 6, "d_model": 256, "timesteps": 1000},
            }, output_ckpt)

    logger.success(f"DiffTS training done. Best loss: {best_loss:.4f}  Saved to {output_ckpt}")


def generate_negatives(checkpoint, output_dir, n_samples, seed):
    """Generate DiffTS negatives with attack perturbations + post-hoc guidance."""
    gen = IoTDiffusionGenerator(seq_length=128, feature_dim=12)
    gen.initialize(checkpoint_path=checkpoint)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {n_samples} benign base samples")
    all_benign = []
    remaining = n_samples
    while remaining > 0:
        bs = min(50, remaining)
        batch = gen.generate(n_samples=bs)
        all_benign.append(batch)
        remaining -= bs

    benign_samples = np.concatenate(all_benign, axis=0)
    np.save(output / "benign_samples.npy", benign_samples)
    logger.info(f"Saved benign: {benign_samples.shape}")

    for attack_type in ATTACK_TYPES:
        for stealth in STEALTH_LEVELS:
            attacks = []
            for i in range(n_samples):
                ref = benign_samples[i]
                attack = gen._inject_attack_pattern(ref.copy(), attack_type)

                target_mean = ref.mean()
                target_std = ref.std() * (1 + (1 - stealth) * 0.5)
                current_mean = attack.mean()
                current_std = attack.std()
                if current_std > 1e-8:
                    attack = (attack - current_mean) / current_std * target_std + target_mean
                attacks.append(attack)

            attacks_array = np.array(attacks)
            fname = f"{attack_type}_stealth_{int(stealth * 100)}.npy"
            np.save(output / fname, attacks_array)
            logger.info(f"  {attack_type} stealth={stealth:.0%}")

    logger.success(f"Done. {len(list(output.glob('*.npy')))} files in {output}")


def main():
    parser = argparse.ArgumentParser(description="Prepare DiffTS negatives for N-BaIoT evaluation")
    parser.add_argument("--data-dir", default="data/nbaiot/")
    parser.add_argument("--device", default="danmini_doorbell")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="models/diffusion_ts_nbaiot.pt")
    parser.add_argument("--output-dir", default="data/synthetic_nbaiot_diffts")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, use existing checkpoint")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    benign_path = ROOT / "data" / "nbaiot_benign_windows.npy"

    if not args.skip_training:
        logger.info("Step 1: Extract N-BaIoT benign windows")
        X_train = extract_benign(args.data_dir, args.device, args.seed)
        np.save(benign_path, X_train)
        logger.info(f"Saved benign windows to {benign_path}")

        logger.info("Step 2: Train Diffusion-TS on N-BaIoT benign data")
        train_diffts(str(benign_path), args.checkpoint, args.epochs, args.batch_size, args.seed)
    else:
        if not Path(args.checkpoint).exists():
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        X_train = np.load(benign_path) if benign_path.exists() else extract_benign(args.data_dir, args.device, args.seed)

    logger.info("Step 3: Generate DiffTS negatives")
    generate_negatives(args.checkpoint, args.output_dir, n_samples=len(X_train), seed=args.seed)


if __name__ == "__main__":
    main()
