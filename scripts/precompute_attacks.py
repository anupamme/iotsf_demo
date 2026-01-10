"""Generate synthetic attacks offline."""

import argparse
import sys
import numpy as np
from pathlib import Path
from loguru import logger

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import IoTDiffusionGenerator
from src.utils.config import Config


def generate_attacks(
    n_samples: int = 10,
    output_dir: str = "data/synthetic",
    seed: int = 42
):
    """
    Pre-generate synthetic attacks for demo.

    Args:
        n_samples: Number of attack samples per type
        output_dir: Output directory for generated attacks
        seed: Random seed for reproducibility
    """
    logger.info("Initializing Diffusion-TS generator...")

    # Load config
    config = Config()

    # Initialize generator
    generator = IoTDiffusionGenerator(
        seq_length=config.get("models.diffusion_ts.seq_length", 128),
        feature_dim=config.get("models.diffusion_ts.feature_dim", 12)
    )
    generator.initialize()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate benign baseline for reference
    logger.info(f"Generating {n_samples} benign samples...")
    benign_samples = generator.generate(n_samples=n_samples, seed=seed)
    benign_file = output_path / "benign_samples.npy"
    np.save(benign_file, benign_samples)
    logger.info(f"✅ Saved benign samples to {benign_file}")

    # Generate each attack type
    attack_types = ["slow_exfiltration", "lotl_mimicry", "protocol_anomaly", "beacon"]
    stealth_levels = [0.85, 0.90, 0.95]

    for attack_type in attack_types:
        logger.info(f"\nGenerating {attack_type} attacks...")

        for stealth_level in stealth_levels:
            attacks = []
            metadata_list = []

            for i in range(n_samples):
                # Use different benign sample as reference
                benign_ref = benign_samples[i % len(benign_samples)]

                # Generate hard-negative attack
                attack, metadata = generator.generate_hard_negative(
                    benign_sample=benign_ref,
                    attack_pattern=attack_type,
                    stealth_level=stealth_level
                )

                attacks.append(attack)
                metadata_list.append(metadata)

            # Save attacks
            attacks_array = np.array(attacks)
            filename = f"{attack_type}_stealth_{int(stealth_level*100)}.npy"
            attack_file = output_path / filename
            np.save(attack_file, attacks_array)

            # Log statistics
            avg_mean_diff = np.mean([m['mean_diff'] for m in metadata_list])
            avg_std_diff = np.mean([m['std_diff'] for m in metadata_list])
            logger.info(f"  Stealth {stealth_level:.2f}: {n_samples} samples")
            logger.info(f"    Avg mean diff: {avg_mean_diff:.4f}")
            logger.info(f"    Avg std diff:  {avg_std_diff:.4f}")
            logger.info(f"    Saved to: {attack_file}")

    logger.info("\n" + "=" * 50)
    logger.info("✅ Attack generation complete!")
    logger.info(f"Total files: {len(list(output_path.glob('*.npy')))}")
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info("=" * 50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Pre-generate synthetic attacks for demo"
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of samples per attack type (default: 10)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/synthetic',
        help='Output directory (default: data/synthetic)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Synthetic Attack Generation")
    logger.info("=" * 50)
    logger.info(f"Samples per type: {args.n_samples}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 50 + "\n")

    try:
        generate_attacks(
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            seed=args.seed
        )
    except Exception as e:
        logger.error(f"Error generating attacks: {e}")
        raise


if __name__ == "__main__":
    main()
