"""
Quick test script for Moirai anomaly detector.

Usage:
    python scripts/test_moirai.py --model-size small
    python scripts/test_moirai.py --sample benign
    python scripts/test_moirai.py --sample slow_exfiltration_stealth_95
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import MoiraiAnomalyDetector
from src.utils.config import Config
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Test Moirai anomaly detector")
    parser.add_argument(
        '--model-size',
        type=str,
        default='small',
        choices=['small', 'base', 'large'],
        help='Moirai model size'
    )
    parser.add_argument(
        '--sample',
        type=str,
        default='benign',
        help='Sample to test (e.g., benign, slow_exfiltration_stealth_95)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.95,
        help='Detection threshold (0-1)'
    )
    parser.add_argument(
        '--seq-idx',
        type=int,
        default=0,
        help='Sequence index to analyze from batch'
    )

    args = parser.parse_args()

    # Load config
    config = Config()

    # Load sample first to determine if we need to adjust context/prediction lengths
    synthetic_dir = Path(config.get('data.synthetic_dir', 'data/synthetic'))
    sample_path = synthetic_dir / f"{args.sample}.npy" if args.sample != 'benign' else synthetic_dir / "benign_samples.npy"

    if not sample_path.exists():
        logger.error(f"Sample not found: {sample_path}")
        logger.info("Available samples:")
        for f in synthetic_dir.glob("*.npy"):
            logger.info(f"  - {f.stem}")
        sys.exit(1)

    logger.info(f"Loading sample: {sample_path}")
    sample_data = np.load(sample_path)
    logger.info(f"Sample shape: {sample_data.shape}")

    # Adjust context/prediction lengths based on sample size
    seq_length = sample_data.shape[1]
    context_length = min(config.get('models.moirai.context_length', 512), seq_length // 2)
    prediction_length = min(config.get('models.moirai.prediction_length', 64), seq_length // 4)

    logger.info(f"Adjusted parameters: context_length={context_length}, prediction_length={prediction_length}")

    # Initialize detector
    logger.info(f"Initializing Moirai {args.model_size} detector...")
    detector = MoiraiAnomalyDetector(
        model_size=args.model_size,
        context_length=context_length,
        prediction_length=prediction_length,
        confidence_level=config.get('models.moirai.confidence_level', 0.95)
    )
    detector.initialize()

    logger.info(f"Detector initialized (mock_mode={detector._mock_mode})")

    # Select sequence
    if args.seq_idx >= len(sample_data):
        logger.error(f"Sequence index {args.seq_idx} out of range (max: {len(sample_data)-1})")
        sys.exit(1)

    traffic = sample_data[args.seq_idx]
    logger.info(f"Selected sequence {args.seq_idx}: shape={traffic.shape}")

    # Run detection
    logger.info("Running anomaly detection...")
    result = detector.detect_anomalies(
        traffic,
        threshold=args.threshold,
        return_feature_contributions=True
    )

    # Print results
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)

    summary = result.summary()
    print(f"\nSample: {args.sample} (sequence {args.seq_idx})")
    print(f"Model: Moirai {args.model_size} (mock_mode={detector._mock_mode})")
    print(f"\nSequence Info:")
    print(f"  Length: {summary['seq_length']}")
    print(f"  Features: {summary['n_features']}")

    print(f"\nDetection Results:")
    print(f"  Threshold: {summary['threshold']:.3f}")
    print(f"  Anomalies Detected: {summary['n_anomalies']}")
    print(f"  Anomaly Rate: {summary['anomaly_rate']:.2%}")
    print(f"  Mean Anomaly Score: {summary['mean_anomaly_score']:.4f}")
    print(f"  Max Anomaly Score: {summary['max_anomaly_score']:.4f}")
    print(f"  Min Anomaly Score: {summary['min_anomaly_score']:.4f}")

    print(f"\nPerformance:")
    print(f"  Inference Time: {summary['metadata'].get('inference_time', 0):.2f}s")

    if result.n_anomalies > 0:
        anomalous_timesteps = result.get_anomalous_timesteps()
        print(f"\nAnomalous Timesteps:")
        print(f"  {anomalous_timesteps[:20].tolist()}" + ("..." if len(anomalous_timesteps) > 20 else ""))

        # Show top contributing features for first anomaly
        first_anomaly = anomalous_timesteps[0]
        top_features = result.get_top_anomalous_features(first_anomaly, top_k=3)

        print(f"\nTop 3 Contributing Features at Timestep {first_anomaly}:")
        feature_names = [
            'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot',
            'fwd_data_pkts_tot', 'bwd_data_pkts_tot',
            'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec',
            'fwd_byts_b_avg', 'bwd_byts_b_avg',
            'fwd_iat_mean', 'bwd_iat_mean'
        ]
        for i, feat_idx in enumerate(top_features):
            contrib = result.feature_contributions[first_anomaly, feat_idx]
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"F{feat_idx}"
            print(f"  {i+1}. {feat_name}: {contrib:.3f}")

    print("\n" + "="*60)

    logger.success("Detection complete!")


if __name__ == "__main__":
    main()
