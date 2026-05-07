"""Generate synthetic attacks using trained Diffusion-TS model (batch mode)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import IoTDiffusionGenerator


ATTACK_TYPES = ["slow_exfiltration", "lotl_mimicry", "protocol_anomaly", "beacon"]
STEALTH_LEVELS = [0.85, 0.90, 0.95]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="data/synthetic_diffts")
    parser.add_argument("--checkpoint", type=str, default="models/diffusion_ts.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--no-guidance", action="store_true",
                        help="Skip post-hoc statistical guidance (rescaling to target stats)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    gen = IoTDiffusionGenerator(seq_length=128, feature_dim=12)
    gen.initialize(checkpoint_path=args.checkpoint)

    logger.info(f"Generating {args.n_samples} benign base samples in batches of {args.batch_size}")
    all_benign = []
    remaining = args.n_samples
    while remaining > 0:
        bs = min(args.batch_size, remaining)
        batch = gen.generate(n_samples=bs)
        all_benign.append(batch)
        remaining -= bs
        logger.info(f"  generated {args.n_samples - remaining}/{args.n_samples}")

    benign_samples = np.concatenate(all_benign, axis=0)
    np.save(output / "benign_samples.npy", benign_samples)
    logger.info(f"Saved benign: shape={benign_samples.shape}, mean={benign_samples.mean():.4f}, std={benign_samples.std():.4f}")

    for attack_type in ATTACK_TYPES:
        for stealth in STEALTH_LEVELS:
            attacks = []
            metadata_list = []
            for i in range(args.n_samples):
                ref = benign_samples[i]
                attack = gen._inject_attack_pattern(ref.copy(), attack_type)

                if not args.no_guidance:
                    s = stealth
                    target_mean = ref.mean()
                    target_std = ref.std() * (1 + (1 - s) * 0.5)
                    current_mean = attack.mean()
                    current_std = attack.std()
                    if current_std > 1e-8:
                        attack = (attack - current_mean) / current_std * target_std + target_mean

                attacks.append(attack)
                metadata_list.append({
                    'mean_diff': abs(attack.mean() - ref.mean()),
                    'std_diff': abs(attack.std() - ref.std()),
                })

            attacks_array = np.array(attacks)
            fname = f"{attack_type}_stealth_{int(stealth*100)}.npy"
            np.save(output / fname, attacks_array)

            avg_mean_diff = np.mean([m['mean_diff'] for m in metadata_list])
            avg_std_diff = np.mean([m['std_diff'] for m in metadata_list])
            logger.info(f"  {attack_type} stealth={stealth:.2f}: mean_diff={avg_mean_diff:.4f}, std_diff={avg_std_diff:.4f}")

    logger.success(f"Done. {len(list(output.glob('*.npy')))} files in {output}")


if __name__ == "__main__":
    main()
