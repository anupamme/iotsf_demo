"""Download and setup model weights."""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def setup_diffusion_ts():
    """Check for Diffusion-TS model checkpoint."""
    checkpoint_path = Path("models/diffusion_ts.pt")

    if checkpoint_path.exists():
        logger.info(f"✅ Diffusion-TS checkpoint found: {checkpoint_path}")
        file_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"   Checkpoint size: {file_size:.2f} MB")
    else:
        logger.warning("⚠️  No Diffusion-TS checkpoint found")
        logger.info("   Using mock mode (realistic statistical generation)")
        logger.info("   To install real Diffusion-TS model:")
        logger.info("   ./scripts/install_diffusion_ts.sh")

    # Test if Diffusion-TS package is installed
    try:
        import diffusion_ts
        logger.info("✅ Diffusion-TS package is installed")
    except ImportError:
        logger.info("ℹ️  Diffusion-TS package not installed (mock mode will be used)")
        logger.info("   Install with: bash scripts/install_diffusion_ts.sh")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup model weights")
    parser.add_argument(
        '--model',
        choices=['diffusion-ts', 'all'],
        default='all',
        help='Which model to setup'
    )
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Model Setup")
    logger.info("=" * 50)

    if args.model in ['diffusion-ts', 'all']:
        logger.info("\nChecking Diffusion-TS...")
        setup_diffusion_ts()

    logger.info("\n" + "=" * 50)
    logger.info("Setup complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
