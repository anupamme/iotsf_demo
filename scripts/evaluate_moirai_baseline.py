#!/usr/bin/env python3
"""
Evaluate Moirai base model (no fine-tuning) on CICIoT2023 and synthetic attacks.

This script evaluates the base Moirai model (directly from HuggingFace, without
fine-tuning) on both real CICIoT2023 data and synthetic hard-negative attacks.
This establishes a baseline to compare against fine-tuned models.

Usage:
    # Basic evaluation
    python scripts/evaluate_moirai_baseline.py

    # With plots
    python scripts/evaluate_moirai_baseline.py --plot

    # Custom thresholds
    python scripts/evaluate_moirai_baseline.py --threshold 0.9 --detection-threshold 0.2

    # Different model size
    python scripts/evaluate_moirai_baseline.py --model-size base
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import MoiraiAnomalyDetector
from src.data.loader import CICIoT2023Loader
from src.data.preprocessor import TrafficPreprocessor, create_sequences
from src.evaluation.metrics import IDSMetrics


def load_ciciot_sequences(data_dir: str, max_samples: int = 5000):
    """
    Load CICIoT2023 data as normalized sequences.

    Args:
        data_dir: Path to directory containing CSV files
        max_samples: Maximum samples to load per class

    Returns:
        Tuple of (benign_sequences, attack_sequences, preprocessor)
    """
    loader = CICIoT2023Loader(data_dir)
    preprocessor = TrafficPreprocessor(scaler_type='standard')

    # Load benign samples
    logger.info(f"Loading benign samples from {data_dir}...")
    benign_df = loader.load_benign_samples(n_samples=max_samples)
    benign_norm = preprocessor.fit_transform(benign_df.values)
    benign_seqs = create_sequences(benign_norm, seq_length=128, stride=128)
    logger.info(f"  Loaded {len(benign_seqs)} benign sequences")

    # Load attack samples (use same scaler fitted on benign)
    logger.info(f"Loading attack samples from {data_dir}...")
    attacks_df = loader.load_any_attack_samples(n_samples=max_samples)
    attacks_norm = preprocessor.transform(attacks_df.values)
    attack_seqs = create_sequences(attacks_norm, seq_length=128, stride=128)
    logger.info(f"  Loaded {len(attack_seqs)} attack sequences")

    return benign_seqs, attack_seqs, preprocessor


def load_synthetic_attacks(synthetic_dir: str):
    """
    Load all synthetic hard-negative files.

    Args:
        synthetic_dir: Path to directory containing .npy files

    Returns:
        Dictionary mapping filename key to numpy array
    """
    results = {}
    synthetic_path = Path(synthetic_dir)

    if not synthetic_path.exists():
        logger.warning(f"Synthetic directory not found: {synthetic_path}")
        return results

    # Load attack files by stealth level
    attack_types = ['slow_exfiltration', 'lotl_mimicry', 'beacon', 'protocol_anomaly']

    for stealth in [85, 90, 95]:
        for attack_type in attack_types:
            filepath = synthetic_path / f'{attack_type}_stealth_{stealth}.npy'
            if filepath.exists():
                data = np.load(filepath)
                results[f'{attack_type}_{stealth}'] = data
                logger.debug(f"  Loaded {filepath.name}: shape {data.shape}")

    # Also load benign samples if available
    benign_path = synthetic_path / 'benign_samples.npy'
    if benign_path.exists():
        data = np.load(benign_path)
        results['benign'] = data
        logger.debug(f"  Loaded benign_samples.npy: shape {data.shape}")

    return results


def evaluate_samples(
    detector,
    samples: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.95,
    detection_rate_threshold: float = 0.3
):
    """
    Run detection and compute metrics.

    Args:
        detector: MoiraiAnomalyDetector instance
        samples: Array of shape (n_samples, seq_length, n_features)
        labels: True labels (0=benign, 1=attack)
        threshold: Anomaly score threshold for Moirai detection
        detection_rate_threshold: Anomaly rate threshold for sample-level detection

    Returns:
        Dictionary with predictions, scores, and metrics
    """
    predictions = []
    anomaly_rates = []
    mean_scores = []

    for i, sample in enumerate(samples):
        try:
            result = detector.detect_anomalies(sample, threshold=threshold)

            # Sample flagged as attack if anomaly_rate > detection_rate_threshold
            pred = 1 if result.anomaly_rate > detection_rate_threshold else 0
            predictions.append(pred)
            anomaly_rates.append(result.anomaly_rate)
            mean_scores.append(result.anomaly_scores.mean())

            if (i + 1) % 20 == 0:
                logger.debug(f"  Processed {i + 1}/{len(samples)} samples")

        except Exception as e:
            logger.warning(f"  Sample {i} failed: {e}")
            # Default to benign prediction on error
            predictions.append(0)
            anomaly_rates.append(0.0)
            mean_scores.append(0.0)

    predictions = np.array(predictions)
    mean_scores = np.array(mean_scores)

    # Compute metrics
    metrics = IDSMetrics.compute_all_metrics(
        y_true=labels,
        y_pred=predictions,
        y_scores=mean_scores
    )

    return {
        'predictions': predictions,
        'anomaly_rates': np.array(anomaly_rates),
        'mean_scores': mean_scores,
        'metrics': metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Moirai base model (no fine-tuning) on IoT traffic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate_moirai_baseline.py

  # With plots
  python scripts/evaluate_moirai_baseline.py --plot

  # Custom thresholds
  python scripts/evaluate_moirai_baseline.py --threshold 0.9 --detection-threshold 0.2
        """
    )

    # Model parameters
    parser.add_argument(
        '--model-size',
        default='small',
        choices=['small', 'base', 'large'],
        help='Moirai model size (default: small)'
    )

    # Detection parameters
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.95,
        help='Anomaly score threshold for Moirai detection (default: 0.95)'
    )
    parser.add_argument(
        '--detection-threshold',
        type=float,
        default=0.3,
        help='Anomaly rate threshold for sample-level detection (default: 0.3)'
    )

    # Data paths
    parser.add_argument(
        '--data-dir',
        default='data/raw/sample',
        help='Directory containing CICIoT2023 CSV files (default: data/raw/sample)'
    )
    parser.add_argument(
        '--synthetic-dir',
        default='data/synthetic',
        help='Directory containing synthetic .npy files (default: data/synthetic)'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory to save results (default: results)'
    )

    # Options
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=5000,
        help='Maximum samples to load per class from CICIoT2023 (default: 5000)'
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Print configuration
    logger.info("=" * 70)
    logger.info("MOIRAI BASE MODEL EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Model size: {args.model_size}")
    logger.info(f"  Anomaly score threshold: {args.threshold}")
    logger.info(f"  Detection rate threshold: {args.detection_threshold}")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Synthetic directory: {args.synthetic_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info("=" * 70)

    # Initialize Moirai BASE model (no checkpoint = no fine-tuning)
    logger.info("Step 1: Initializing Moirai base model (no fine-tuning)...")
    detector = MoiraiAnomalyDetector(
        model_size=args.model_size,
        context_length=96,
        prediction_length=32,
        confidence_level=0.95
    )
    detector.initialize()  # No checkpoint = base model from HuggingFace
    logger.success(f"  Moirai {args.model_size} initialized on {detector.device}")

    results = {}
    benign_seqs = None  # Initialize to handle case where CICIoT2023 loading fails

    # === 1. Evaluate on CICIoT2023 ===
    logger.info("")
    logger.info("Step 2: Evaluating on CICIoT2023 dataset...")
    try:
        benign_seqs, attack_seqs, _ = load_ciciot_sequences(
            args.data_dir,
            max_samples=args.max_samples
        )

        ciciot_samples = np.concatenate([benign_seqs, attack_seqs])
        ciciot_labels = np.array([0] * len(benign_seqs) + [1] * len(attack_seqs))

        logger.info(f"  Evaluating {len(ciciot_samples)} samples ({len(benign_seqs)} benign, {len(attack_seqs)} attack)...")
        results['ciciot2023'] = evaluate_samples(
            detector,
            ciciot_samples,
            ciciot_labels,
            threshold=args.threshold,
            detection_rate_threshold=args.detection_threshold
        )
        logger.success(f"  CICIoT2023 evaluation complete")

    except Exception as e:
        logger.error(f"Failed to evaluate CICIoT2023: {e}")
        logger.error("Make sure data files exist in the data directory")

    # === 2. Evaluate on Synthetic Hard-Negatives ===
    logger.info("")
    logger.info("Step 3: Evaluating on synthetic hard-negatives...")
    synthetic_data = load_synthetic_attacks(args.synthetic_dir)

    if not synthetic_data:
        logger.warning("No synthetic data found, skipping...")
    else:
        # Get benign samples for comparison - require valid benign data
        benign_synthetic = synthetic_data.get('benign', None)
        if benign_synthetic is None:
            # Try to use CICIoT2023 benign samples if available
            if 'ciciot2023' in results and benign_seqs is not None and len(benign_seqs) > 0:
                logger.warning("No synthetic benign samples, using first 10 CICIoT2023 benign")
                benign_synthetic = benign_seqs[:10]
            else:
                # Cannot proceed without valid benign samples - metrics would be meaningless
                logger.error("No valid benign samples available for synthetic evaluation")
                logger.error("Skipping synthetic hard-negative evaluation (FPR would be meaningless)")
                synthetic_data = {}  # Clear to skip the evaluation loop

        # Evaluate each stealth level separately
        attack_types = ['slow_exfiltration', 'lotl_mimicry', 'beacon', 'protocol_anomaly']

        for stealth in [85, 90, 95]:
            stealth_attacks = []
            for attack_type in attack_types:
                key = f'{attack_type}_{stealth}'
                if key in synthetic_data:
                    stealth_attacks.append(synthetic_data[key])

            if stealth_attacks:
                all_attacks = np.concatenate(stealth_attacks)
                samples = np.concatenate([benign_synthetic, all_attacks])
                labels = np.array([0] * len(benign_synthetic) + [1] * len(all_attacks))

                logger.info(f"  Evaluating Stealth {stealth}%: {len(benign_synthetic)} benign, {len(all_attacks)} attacks...")
                results[f'synthetic_stealth_{stealth}'] = evaluate_samples(
                    detector,
                    samples,
                    labels,
                    threshold=args.threshold,
                    detection_rate_threshold=args.detection_threshold
                )
                logger.success(f"  Stealth {stealth}% evaluation complete")

    # === 3. Print Results ===
    print("\n" + "=" * 70)
    print("MOIRAI BASE MODEL EVALUATION RESULTS")
    print("=" * 70)

    for name, result in results.items():
        m = result['metrics']
        print(f"\nDataset: {name}")
        print(f"  Detection Rate (Recall): {m.get('recall', 0) * 100:.1f}%")
        print(f"  False Positive Rate:     {m.get('false_positive_rate', 0) * 100:.1f}%")
        print(f"  Precision:               {m.get('precision', 0) * 100:.1f}%")
        print(f"  F1 Score:                {m.get('f1', 0):.3f}")
        roc_auc = m.get('roc_auc')
        if roc_auc is not None:
            print(f"  ROC-AUC:                 {roc_auc:.3f}")
        print(f"  Mean Anomaly Score:      {result['mean_scores'].mean():.3f}")

    print("\n" + "=" * 70)

    # === 4. Save Results ===
    output_file = output_path / 'moirai_baseline_evaluation.json'

    # Convert numpy arrays and handle non-serializable types for JSON
    json_results = {}
    for name, result in results.items():
        metrics_copy = result['metrics'].copy()
        # Convert confusion_matrix numpy array to list
        if 'confusion_matrix' in metrics_copy:
            metrics_copy['confusion_matrix'] = metrics_copy['confusion_matrix'].tolist()

        json_results[name] = {
            'metrics': metrics_copy,
            'mean_anomaly_score': float(result['mean_scores'].mean()),
            'std_anomaly_score': float(result['mean_scores'].std()),
            'n_samples': len(result['predictions']),
            'n_detected': int(result['predictions'].sum())
        }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    logger.success(f"Results saved to {output_file}")

    # === 5. Generate Plots (optional) ===
    if args.plot and results:
        try:
            import matplotlib.pyplot as plt

            # Detection rate by dataset
            datasets = list(results.keys())
            detection_rates = [results[d]['metrics'].get('recall', 0) * 100 for d in datasets]
            fpr_rates = [results[d]['metrics'].get('false_positive_rate', 0) * 100 for d in datasets]

            # Plot 1: Detection Rate Bar Chart
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Detection rates
            colors = ['#2ecc71' if 'ciciot' in d else
                     '#e74c3c' if '85' in d else
                     '#f39c12' if '90' in d else
                     '#9b59b6' for d in datasets]

            bars = axes[0].bar(datasets, detection_rates, color=colors)
            axes[0].set_ylabel('Detection Rate (%)', fontsize=12)
            axes[0].set_title('Moirai Base Model - Detection Rate by Dataset', fontsize=14, fontweight='bold')
            axes[0].set_ylim(0, 100)

            for bar, rate in zip(bars, detection_rates):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f'{rate:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )

            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

            # False positive rates
            bars = axes[1].bar(datasets, fpr_rates, color=colors)
            axes[1].set_ylabel('False Positive Rate (%)', fontsize=12)
            axes[1].set_title('Moirai Base Model - False Positive Rate by Dataset', fontsize=14, fontweight='bold')
            axes[1].set_ylim(0, max(fpr_rates) * 1.2 + 5 if fpr_rates else 20)

            for bar, rate in zip(bars, fpr_rates):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{rate:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )

            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

            plt.tight_layout()

            plot_file = output_path / 'moirai_baseline_detection_rates.png'
            plt.savefig(plot_file, dpi=150)
            logger.success(f"Plot saved to {plot_file}")

        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")

    # === 6. Summary ===
    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_file}")
    if args.plot:
        logger.info(f"Plots saved to: {output_path / 'moirai_baseline_detection_rates.png'}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
