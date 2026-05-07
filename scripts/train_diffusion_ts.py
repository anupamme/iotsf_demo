"""Train Diffusion-TS on CICIoT2023 benign traffic for hard-negative generation."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.diffusion_ts_adapter import setup_diffusion_ts_path

setup_diffusion_ts_path()
from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data = np.load(args.data_path)
    logger.info(f"Loaded benign data: {data.shape}")

    if args.max_samples and args.max_samples < len(data):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(data), args.max_samples, replace=False)
        data = data[idx]
        logger.info(f"Subsampled to {len(data)} samples")

    seq_length, feature_dim = data.shape[1], data.shape[2]

    tensor_data = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = Diffusion_TS(
        seq_length=seq_length,
        feature_size=feature_dim,
        n_layer_enc=3,
        n_layer_dec=6,
        d_model=256,
        timesteps=1000,
        sampling_timesteps=50,
        loss_type="l1",
        beta_schedule="cosine",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Diffusion-TS model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

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

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
                "config": {
                    "seq_length": seq_length,
                    "feature_dim": feature_dim,
                    "n_layer_enc": 3,
                    "n_layer_dec": 6,
                    "d_model": 256,
                    "timesteps": 1000,
                },
            }, output_path)

    logger.success(f"Training complete. Best loss: {best_loss:.4f}  Saved to {output_path}")

    model.eval()
    with torch.no_grad():
        sample = model.fast_sample((2, seq_length, feature_dim))
        logger.info(f"Verification sample shape: {sample.shape}, mean={sample.mean():.3f}, std={sample.std():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion-TS on benign IoT traffic")
    parser.add_argument("--data-path", default="data/synthetic/benign_samples.npy")
    parser.add_argument("--output", default="models/diffusion_ts.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)
