#!/usr/bin/env python3
"""
Fine-tune Moirai model on IoT traffic data.

This script fine-tunes the Salesforce Moirai foundation model on a mixture of:
- 70% benign IoT traffic (real CICIoT2023 data)
- 20% hard-negative attacks (synthetic stealthy attacks at 85-95% similarity)
- 10% standard attacks (real CICIoT2023 attacks)

The fine-tuning uses self-supervised forecasting loss to adapt the model to
IoT-specific patterns, improving detection of stealthy anomalies.

Usage:
    # Default training (10 epochs, batch size 32, lr 1e-4)
    python scripts/train_moirai.py

    # Custom hyperparameters
    python scripts/train_moirai.py --epochs 20 --batch-size 64 --lr 5e-5

    # Use GPU if available
    python scripts/train_moirai.py --use-gpu

    # Custom data composition
    python scripts/train_moirai.py --benign-ratio 0.6 --hard-neg-ratio 0.3 --attack-ratio 0.1

    # Specify output directory
    python scripts/train_moirai.py --checkpoint models/my_finetuned_moirai
"""

import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from loguru import logger

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import MoiraiAnomalyDetector
from src.data.loader import CICIoT2023Loader
from src.utils.config import Config


def plot_training_curves(history: dict, output_path: Path):
    """
    Plot and save training curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        output_path: Path to save the plot
    """
    if not history['train_loss']:
        logger.warning("No training history to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Moirai Fine-Tuning: Training Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add min val loss marker
    min_val_idx = history['val_loss'].index(min(history['val_loss']))
    min_val_loss = history['val_loss'][min_val_idx]
    ax.axvline(x=min_val_idx + 1, color='green', linestyle='--', alpha=0.5, label=f'Best (Epoch {min_val_idx + 1})')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Training curves saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Moirai on IoT network traffic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default training
  python scripts/train_moirai.py

  # Custom hyperparameters
  python scripts/train_moirai.py --epochs 20 --batch-size 64 --lr 5e-5

  # Use GPU
  python scripts/train_moirai.py --use-gpu

  # Custom checkpoint directory
  python scripts/train_moirai.py --checkpoint models/my_moirai
        """
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate for AdamW optimizer (default: 1e-4)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Early stopping patience in epochs (default: 3)'
    )

    # Data parameters
    parser.add_argument(
        '--benign-ratio',
        type=float,
        default=0.7,
        help='Proportion of benign samples (default: 0.7)'
    )
    parser.add_argument(
        '--hard-neg-ratio',
        type=float,
        default=0.2,
        help='Proportion of hard-negative attacks (default: 0.2)'
    )
    parser.add_argument(
        '--attack-ratio',
        type=float,
        default=0.1,
        help='Proportion of standard attacks (default: 0.1)'
    )
    parser.add_argument(
        '--total-samples',
        type=int,
        default=1000,
        help='Total number of training samples (default: 1000)'
    )

    # Model parameters
    parser.add_argument(
        '--model-size',
        type=str,
        default='small',
        choices=['small', 'base', 'large'],
        help='Moirai model size (default: small)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/moirai_finetuned',
        help='Directory to save checkpoints (default: models/moirai_finetuned)'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU if available (default: CPU)'
    )

    # Output
    parser.add_argument(
        '--plot',
        type=str,
        default='training_curves.png',
        help='Output path for training curves plot (default: training_curves.png)'
    )

    args = parser.parse_args()

    # Validate data ratios
    total_ratio = args.benign_ratio + args.hard_neg_ratio + args.attack_ratio
    if abs(total_ratio - 1.0) > 0.01:
        logger.error(
            f"Data ratios must sum to 1.0, got {total_ratio:.2f} "
            f"(benign={args.benign_ratio}, hard-neg={args.hard_neg_ratio}, attack={args.attack_ratio})"
        )
        sys.exit(1)

    # Load configuration
    config_path = ROOT_DIR / "config" / "config.yaml"
    config = Config(config_path if config_path.exists() else None)

    # Print configuration
    logger.info("=" * 70)
    logger.info("MOIRAI FINE-TUNING SCRIPT")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Model size: {args.model_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Early stopping patience: {args.patience}")
    logger.info(f"  Device: {'GPU (auto)' if args.use_gpu else 'CPU'}")
    logger.info(f"")
    logger.info(f"Data composition:")
    logger.info(f"  Total samples: {args.total_samples}")
    logger.info(f"  Benign: {args.benign_ratio*100:.0f}% ({int(args.total_samples * args.benign_ratio)} samples)")
    logger.info(f"  Hard-negatives: {args.hard_neg_ratio*100:.0f}% ({int(args.total_samples * args.hard_neg_ratio)} samples)")
    logger.info(f"  Standard attacks: {args.attack_ratio*100:.0f}% ({int(args.total_samples * args.attack_ratio)} samples)")
    logger.info(f"")
    logger.info(f"Output:")
    logger.info(f"  Checkpoint dir: {args.checkpoint}")
    logger.info(f"  Training plot: {args.plot}")
    logger.info("=" * 70)
    logger.info("")

    # 1. Initialize Moirai model
    logger.info("Step 1: Initializing Moirai model...")
    moirai = MoiraiAnomalyDetector(
        model_size=args.model_size,
        context_length=config.get("models.moirai.context_length", 96),
        prediction_length=config.get("models.moirai.prediction_length", 32),
        device='auto' if args.use_gpu else 'cpu'
    )

    try:
        moirai.initialize()
        logger.success(f"✓ Moirai {args.model_size} initialized on {moirai.device}")
    except Exception as e:
        logger.error(f"Failed to initialize Moirai: {e}")
        logger.error("Make sure uni2ts is installed: pip install uni2ts")
        sys.exit(1)

    if moirai._mock_mode:
        logger.error("Cannot fine-tune in mock mode. Install uni2ts and try again.")
        sys.exit(1)

    # 2. Load training data
    logger.info("Step 2: Loading training data...")
    data_dir = config.get("data.raw_dir", ROOT_DIR / "data" / "raw")
    loader = CICIoT2023Loader(str(data_dir))

    try:
        train_data, val_data = loader.load_finetune_data(
            benign_ratio=args.benign_ratio,
            hard_negative_ratio=args.hard_neg_ratio,
            standard_attack_ratio=args.attack_ratio,
            total_samples=args.total_samples
        )
        logger.success(f"✓ Data loaded: train={train_data.shape}, val={val_data.shape}")
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        logger.error("Make sure data files exist:")
        logger.error("  - Benign: data/raw/benign*.csv")
        logger.error("  - Attacks: data/raw/*attack*.csv")
        logger.error("  - Hard-negatives: data/synthetic/*stealth*.npy")
        logger.error("")
        logger.error("Run: python scripts/precompute_attacks.py")
        sys.exit(1)

    # 3. Fine-tune model
    logger.info("Step 3: Starting fine-tuning...")
    try:
        history = moirai.fine_tune(
            train_data=train_data,
            val_data=val_data,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            checkpoint_dir=args.checkpoint,
            early_stopping_patience=args.patience
        )
        logger.success("✓ Fine-tuning complete!")
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Plot training curves
    logger.info("Step 4: Plotting training curves...")
    if history['train_loss']:
        plot_output = Path(args.plot)
        plot_training_curves(history, plot_output)
        logger.success(f"✓ Training curves saved to {plot_output}")
    else:
        logger.warning("No training history to plot")

    # 5. Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINE-TUNING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results:")
    if history['train_loss']:
        logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        logger.info(f"  Best val loss: {min(history['val_loss']):.4f} (epoch {history['val_loss'].index(min(history['val_loss'])) + 1})")
        logger.info(f"  Total epochs: {len(history['train_loss'])}")
    logger.info(f"")
    logger.info(f"Outputs:")
    logger.info(f"  Best checkpoint: {args.checkpoint}/best_moirai.pt")
    logger.info(f"  Training curves: {args.plot}")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Update config.yaml:")
    logger.info(f"     models.moirai.checkpoint: '{args.checkpoint}/best_moirai.pt'")
    logger.info("")
    logger.info("  2. Regenerate demo data with fine-tuned model:")
    logger.info("     python scripts/precompute_demo_data.py --seed 42")
    logger.info("")
    logger.info("  3. Run Streamlit demo:")
    logger.info("     streamlit run app/main.py")
    logger.info("")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
