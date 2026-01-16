"""
Pre-compute all demo data including detection results.

Generates:
- 3 benign samples from CICIoT2023
- 3 synthetic attacks (slow_exfiltration, lotl_mimicry, beacon)
- Baseline IDS predictions for all 6 samples
- Moirai detection predictions for all 6 samples
- All saved to data/synthetic/demo_data.npz
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.loader import CICIoT2023Loader
from src.data.preprocessor import TrafficPreprocessor, create_sequences
from src.models import IoTDiffusionGenerator, MoiraiAnomalyDetector, ThresholdIDS
from src.utils.config import Config


def load_and_prepare_benign(
    loader: CICIoT2023Loader,
    n_samples: int,
    seq_length: int,
    feature_dim: int,
    seed: int = 42
) -> np.ndarray:
    """
    Load real benign samples from CICIoT2023 and convert to sequences.

    Args:
        loader: Data loader instance
        n_samples: Number of benign samples to load
        seq_length: Sequence length (e.g., 128)
        feature_dim: Number of features (e.g., 12)
        seed: Random seed

    Returns:
        Benign samples array of shape (n_samples, seq_length, feature_dim)
    """
    logger.info(f"Loading {n_samples} benign samples from CICIoT2023...")

    try:
        # Load benign traffic
        benign_df = loader.load_benign_samples(n_samples=n_samples * seq_length)

        if len(benign_df) < n_samples * seq_length:
            logger.warning(
                f"Only {len(benign_df)} records available, "
                f"need {n_samples * seq_length}"
            )
            raise ValueError("Not enough benign samples in dataset")

        # Convert to numpy
        benign_data = benign_df.values

        # Normalize
        preprocessor = TrafficPreprocessor(scaler_type='standard')
        normalized_data = preprocessor.fit_transform(benign_data)

        # Create sequences
        sequences = create_sequences(
            normalized_data,
            seq_length=seq_length,
            stride=seq_length  # Non-overlapping
        )

        # Select n_samples
        np.random.seed(seed)
        if len(sequences) > n_samples:
            selected_indices = np.random.choice(len(sequences), n_samples, replace=False)
            sequences = sequences[selected_indices]

        logger.success(f"Loaded {len(sequences)} benign sequences")
        return sequences[:n_samples]

    except Exception as e:
        logger.warning(f"Failed to load real data: {e}")
        logger.info("Falling back to synthetic benign samples...")

        # Fallback: use synthetic benign samples
        generator = IoTDiffusionGenerator(seq_length=seq_length, feature_dim=feature_dim)
        generator.initialize()
        benign_samples = generator.generate(n_samples=n_samples, seed=seed)

        logger.info(f"Generated {n_samples} synthetic benign samples")
        return benign_samples


def generate_attack_samples(
    generator: IoTDiffusionGenerator,
    benign_references: np.ndarray,
    attack_types: list,
    stealth_level: float = 0.95
) -> tuple:
    """
    Generate synthetic attack samples.

    Args:
        generator: Diffusion-TS generator
        benign_references: Benign samples to use as references
        attack_types: List of attack types to generate
        stealth_level: Stealth level for hard-negative generation

    Returns:
        Tuple of (attacks_array, attack_labels)
    """
    logger.info(f"Generating {len(attack_types)} attack samples...")

    attacks = []
    labels = []

    for i, attack_type in enumerate(attack_types):
        # Use corresponding benign sample as reference
        benign_ref = benign_references[i % len(benign_references)]

        logger.info(f"  Generating {attack_type} attack...")

        # Generate hard-negative attack
        attack, metadata = generator.generate_hard_negative(
            benign_sample=benign_ref,
            attack_pattern=attack_type,
            stealth_level=stealth_level
        )

        attacks.append(attack)
        labels.append(attack_type)

        logger.debug(
            f"    Mean diff: {metadata['mean_diff']:.4f}, "
            f"Std diff: {metadata['std_diff']:.4f}"
        )

    attacks_array = np.array(attacks)
    logger.success(f"Generated {len(attacks)} attack samples")

    return attacks_array, labels


def run_baseline_detection(
    baseline: ThresholdIDS,
    benign_samples: np.ndarray,
    attack_samples: np.ndarray
) -> dict:
    """
    Run baseline IDS on all samples.

    Args:
        baseline: Baseline IDS instance
        benign_samples: Benign traffic samples
        attack_samples: Attack traffic samples

    Returns:
        Dictionary with predictions and scores
    """
    logger.info("Running baseline IDS detection...")

    # Fit on benign samples
    baseline.fit(benign_samples)

    # Combine all samples
    all_samples = np.concatenate([benign_samples, attack_samples], axis=0)

    # Predict using sklearn-style interface
    predictions = baseline.predict(all_samples)
    scores = baseline.predict_proba(all_samples)

    results = {
        'predictions': predictions,
        'scores': scores
    }

    logger.success(
        f"Baseline IDS: Detected {results['predictions'].sum()}/{len(all_samples)} as attacks"
    )

    return results


def run_moirai_detection(
    moirai: MoiraiAnomalyDetector,
    benign_samples: np.ndarray,
    attack_samples: np.ndarray
) -> dict:
    """
    Run Moirai detector on all samples.

    Args:
        moirai: Moirai detector instance
        benign_samples: Benign traffic samples
        attack_samples: Attack traffic samples

    Returns:
        Dictionary with predictions, scores, and forecasts
    """
    logger.info("Running Moirai detection...")

    # Combine all samples
    all_samples = np.concatenate([benign_samples, attack_samples], axis=0)

    # Detect (MoiraiAnomalyDetector works on 2D: seq_length x features)
    predictions = []
    scores = []
    all_forecasts = []

    for i, sample in enumerate(all_samples):
        logger.debug(f"Processing sample {i+1}/{len(all_samples)}")
        result = moirai.detect_anomalies(sample, threshold=0.95, return_feature_contributions=False)

        # Aggregate anomaly info
        predictions.append(1 if result.n_anomalies > 0 else 0)
        scores.append(result.anomaly_scores.mean())  # Average anomaly score
        all_forecasts.append(result.predictions)  # Shape: (seq_length, n_features)

    # Convert forecasts to consistent shape (n_samples, pred_length, n_features)
    # Use first prediction_length steps
    pred_length = min(28, all_forecasts[0].shape[0])
    forecasts_array = np.array([f[:pred_length] for f in all_forecasts])

    results = {
        'predictions': np.array(predictions),
        'scores': np.array(scores),
        'forecasts': forecasts_array
    }

    logger.success(
        f"Moirai: Detected {results['predictions'].sum()}/{len(all_samples)} as attacks"
    )

    return results


def save_demo_data(
    output_path: Path,
    benign_samples: np.ndarray,
    attack_samples: np.ndarray,
    attack_labels: list,
    baseline_results: dict,
    moirai_results: dict,
    metadata: dict
):
    """
    Save all demo data to NPZ file.

    Args:
        output_path: Output file path
        benign_samples: Benign samples array
        attack_samples: Attack samples array
        attack_labels: List of attack type labels
        baseline_results: Baseline IDS results
        moirai_results: Moirai results
        metadata: Metadata dictionary
    """
    logger.info(f"Saving demo data to {output_path}...")

    # Ground truth labels: benign=0, attack=1
    n_benign = len(benign_samples)
    n_attacks = len(attack_samples)
    labels = np.array([0] * n_benign + [1] * n_attacks)

    # Save to NPZ
    np.savez(
        output_path,
        # Samples
        benign_samples=benign_samples,
        attack_samples=attack_samples,
        attack_labels=attack_labels,
        labels=labels,
        # Baseline IDS results
        baseline_predictions=baseline_results['predictions'],
        baseline_scores=baseline_results['scores'],
        # Moirai results
        moirai_predictions=moirai_results['predictions'],
        moirai_scores=moirai_results['scores'],
        moirai_forecasts=moirai_results['forecasts'],
        # Metadata
        metadata=metadata
    )

    file_size = output_path.stat().st_size / 1024  # KB
    logger.success(f"✅ Demo data saved ({file_size:.1f} KB)")


def generate_demo_data(
    output_dir: str = "data/synthetic",
    n_benign: int = 3,
    n_attacks: int = 3,
    attack_types: list = None,
    seed: int = 42
):
    """
    Main function to generate all demo data.

    Args:
        output_dir: Output directory
        n_benign: Number of benign samples
        n_attacks: Number of attack samples
        attack_types: List of attack types (default: ['slow_exfiltration', 'lotl_mimicry', 'beacon'])
        seed: Random seed
    """
    if attack_types is None:
        attack_types = ['slow_exfiltration', 'lotl_mimicry', 'beacon']

    if len(attack_types) != n_attacks:
        raise ValueError(f"Number of attack_types ({len(attack_types)}) must match n_attacks ({n_attacks})")

    # Load config
    config = Config()
    seq_length = config.get("models.diffusion_ts.seq_length", 128)
    feature_dim = config.get("models.diffusion_ts.feature_dim", 12)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "demo_data.npz"

    # 1. Load benign samples
    loader = CICIoT2023Loader(config.get("data.raw_dir", "data/raw"))
    benign_samples = load_and_prepare_benign(
        loader, n_benign, seq_length, feature_dim, seed
    )

    # 2. Generate attack samples
    generator = IoTDiffusionGenerator(seq_length=seq_length, feature_dim=feature_dim)
    generator.initialize()
    attack_samples, attack_labels = generate_attack_samples(
        generator, benign_samples, attack_types, stealth_level=0.95
    )

    # 3. Run baseline IDS
    baseline = ThresholdIDS()
    baseline_results = run_baseline_detection(baseline, benign_samples, attack_samples)

    # 4. Run Moirai detection
    moirai = MoiraiAnomalyDetector(
        model_size=config.get("models.moirai.model_size", "small"),
        context_length=config.get("models.moirai.context_length", 512),
        prediction_length=config.get("models.moirai.prediction_length", 64),
        confidence_level=0.95,
        device='auto'
    )
    moirai.initialize()
    moirai_results = run_moirai_detection(moirai, benign_samples, attack_samples)

    # 5. Create metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_benign': n_benign,
        'n_attacks': n_attacks,
        'attack_types': attack_labels,
        'seq_length': seq_length,
        'feature_dim': feature_dim,
        'seed': seed,
        'baseline_params': {'type': 'ThresholdIDS'},
        'moirai_params': {
            'model_size': moirai.model_size,
            'context_length': moirai.context_length,
            'prediction_length': moirai.prediction_length,
            'device': str(moirai.device)
        }
    }

    # 6. Save everything
    save_demo_data(
        output_file,
        benign_samples,
        attack_samples,
        attack_labels,
        baseline_results,
        moirai_results,
        metadata
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DEMO DATA GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output file: {output_file.absolute()}")
    logger.info(f"Benign samples: {n_benign}")
    logger.info(f"Attack samples: {n_attacks} ({', '.join(attack_labels)})")
    logger.info(f"Baseline detections: {baseline_results['predictions'].sum()}/6")
    logger.info(f"Moirai detections: {moirai_results['predictions'].sum()}/6")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-compute demo data with detection results"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/synthetic',
        help='Output directory (default: data/synthetic)'
    )
    parser.add_argument(
        '--n-benign',
        type=int,
        default=3,
        help='Number of benign samples (default: 3)'
    )
    parser.add_argument(
        '--n-attacks',
        type=int,
        default=3,
        help='Number of attack samples (default: 3)'
    )
    parser.add_argument(
        '--attack-types',
        nargs='+',
        default=['slow_exfiltration', 'lotl_mimicry', 'beacon'],
        help='Attack types to generate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRE-COMPUTE DEMO DATA PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Benign samples: {args.n_benign}")
    logger.info(f"Attack samples: {args.n_attacks}")
    logger.info(f"Attack types: {args.attack_types}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 60 + "\n")

    try:
        generate_demo_data(
            output_dir=args.output_dir,
            n_benign=args.n_benign,
            n_attacks=args.n_attacks,
            attack_types=args.attack_types,
            seed=args.seed
        )
    except Exception as e:
        logger.error(f"Error generating demo data: {e}")
        raise


if __name__ == "__main__":
    main()
