"""
Verify integrity of pre-computed demo data.

Checks:
- File exists and is readable
- All arrays have correct shapes
- No NaN/Inf values
- Detection results are in valid ranges
- Attack samples differ from benign samples
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from scipy import stats
from loguru import logger

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


class DemoDataVerifier:
    """Verifies integrity of demo_data.npz file."""

    def __init__(self, data_path: str):
        """
        Initialize verifier.

        Args:
            data_path: Path to demo_data.npz file
        """
        self.data_path = Path(data_path)
        self.data = None
        self.errors = []
        self.warnings = []
        self.passed_checks = 0
        self.total_checks = 0

    def load_data(self) -> bool:
        """Load demo data file."""
        logger.info(f"Loading data from {self.data_path}...")

        if not self.data_path.exists():
            logger.error(f"❌ File not found: {self.data_path}")
            return False

        try:
            self.data = np.load(self.data_path, allow_pickle=True)
            logger.success(f"✅ File loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load file: {e}")
            return False

    def check_required_keys(self) -> bool:
        """Check that all required keys are present."""
        self.total_checks += 1
        logger.info("Checking required keys...")

        required_keys = [
            'benign_samples',
            'attack_samples',
            'attack_labels',
            'labels',
            'baseline_predictions',
            'baseline_scores',
            'moirai_predictions',
            'moirai_scores',
            'moirai_forecasts',
            'metadata'
        ]

        missing_keys = [key for key in required_keys if key not in self.data]

        if missing_keys:
            logger.error(f"❌ Missing keys: {missing_keys}")
            self.errors.append(f"Missing keys: {missing_keys}")
            return False

        logger.success(f"✅ All {len(required_keys)} required keys present")
        self.passed_checks += 1
        return True

    def check_array_shapes(self) -> bool:
        """Check that arrays have expected shapes."""
        self.total_checks += 1
        logger.info("Checking array shapes...")

        try:
            benign = self.data['benign_samples']
            attacks = self.data['attack_samples']
            labels = self.data['labels']
            baseline_pred = self.data['baseline_predictions']
            baseline_scores = self.data['baseline_scores']
            moirai_pred = self.data['moirai_predictions']
            moirai_scores = self.data['moirai_scores']
            moirai_forecasts = self.data['moirai_forecasts']

            # Expected shapes
            n_benign = benign.shape[0]
            n_attacks = attacks.shape[0]
            n_total = n_benign + n_attacks
            seq_length = benign.shape[1]
            n_features = benign.shape[2]

            # Check samples
            assert benign.ndim == 3, f"benign_samples should be 3D, got {benign.ndim}D"
            assert attacks.shape == (n_attacks, seq_length, n_features), \
                f"attack_samples shape mismatch: {attacks.shape}"

            # Check labels
            assert labels.shape == (n_total,), f"labels shape mismatch: {labels.shape}"

            # Check predictions and scores
            assert baseline_pred.shape == (n_total,), \
                f"baseline_predictions shape mismatch: {baseline_pred.shape}"
            assert baseline_scores.shape == (n_total,), \
                f"baseline_scores shape mismatch: {baseline_scores.shape}"
            assert moirai_pred.shape == (n_total,), \
                f"moirai_predictions shape mismatch: {moirai_pred.shape}"
            assert moirai_scores.shape == (n_total,), \
                f"moirai_scores shape mismatch: {moirai_scores.shape}"

            # Check forecasts
            assert moirai_forecasts.ndim == 3, \
                f"moirai_forecasts should be 3D, got {moirai_forecasts.ndim}D"
            assert moirai_forecasts.shape[0] == n_total, \
                f"moirai_forecasts should have {n_total} samples, got {moirai_forecasts.shape[0]}"
            assert moirai_forecasts.shape[2] == n_features, \
                f"moirai_forecasts should have {n_features} features, got {moirai_forecasts.shape[2]}"

            logger.success(
                f"✅ All shapes correct: "
                f"{n_benign} benign + {n_attacks} attacks, "
                f"seq_len={seq_length}, features={n_features}"
            )
            self.passed_checks += 1
            return True

        except AssertionError as e:
            logger.error(f"❌ Shape check failed: {e}")
            self.errors.append(str(e))
            return False

    def check_no_invalid_values(self) -> bool:
        """Check for NaN and Inf values."""
        self.total_checks += 1
        logger.info("Checking for NaN/Inf values...")

        arrays_to_check = [
            ('benign_samples', self.data['benign_samples']),
            ('attack_samples', self.data['attack_samples']),
            ('baseline_scores', self.data['baseline_scores']),
            ('moirai_scores', self.data['moirai_scores']),
            ('moirai_forecasts', self.data['moirai_forecasts'])
        ]

        has_invalid = False
        for name, arr in arrays_to_check:
            if np.any(np.isnan(arr)):
                logger.error(f"❌ {name} contains NaN values")
                self.errors.append(f"{name} has NaN values")
                has_invalid = True
            if np.any(np.isinf(arr)):
                logger.error(f"❌ {name} contains Inf values")
                self.errors.append(f"{name} has Inf values")
                has_invalid = True

        if not has_invalid:
            logger.success("✅ No NaN or Inf values found")
            self.passed_checks += 1
            return True

        return False

    def check_score_ranges(self) -> bool:
        """Check that scores are in valid range [0, 1]."""
        self.total_checks += 1
        logger.info("Checking score ranges...")

        baseline_scores = self.data['baseline_scores']
        moirai_scores = self.data['moirai_scores']

        issues = []
        if np.any(baseline_scores < 0) or np.any(baseline_scores > 1):
            issues.append("baseline_scores outside [0, 1]")
            logger.error(
                f"❌ Baseline scores range: [{baseline_scores.min():.3f}, "
                f"{baseline_scores.max():.3f}]"
            )

        if np.any(moirai_scores < 0) or np.any(moirai_scores > 1):
            issues.append("moirai_scores outside [0, 1]")
            logger.error(
                f"❌ Moirai scores range: [{moirai_scores.min():.3f}, "
                f"{moirai_scores.max():.3f}]"
            )

        if issues:
            self.errors.extend(issues)
            return False

        logger.success(
            f"✅ Scores in valid range: "
            f"baseline=[{baseline_scores.min():.3f}, {baseline_scores.max():.3f}], "
            f"moirai=[{moirai_scores.min():.3f}, {moirai_scores.max():.3f}]"
        )
        self.passed_checks += 1
        return True

    def check_binary_predictions(self) -> bool:
        """Check that predictions are binary (0 or 1)."""
        self.total_checks += 1
        logger.info("Checking binary predictions...")

        baseline_pred = self.data['baseline_predictions']
        moirai_pred = self.data['moirai_predictions']

        issues = []
        if not np.all(np.isin(baseline_pred, [0, 1])):
            issues.append("baseline_predictions not binary")
            logger.error(f"❌ Baseline predictions contain non-binary values")

        if not np.all(np.isin(moirai_pred, [0, 1])):
            issues.append("moirai_predictions not binary")
            logger.error(f"❌ Moirai predictions contain non-binary values")

        if issues:
            self.errors.extend(issues)
            return False

        logger.success("✅ All predictions are binary (0 or 1)")
        self.passed_checks += 1
        return True

    def check_statistical_difference(self) -> bool:
        """Check that attacks are statistically different from benign."""
        self.total_checks += 1
        logger.info("Checking statistical differences...")

        benign = self.data['benign_samples']
        attacks = self.data['attack_samples']

        # Flatten samples for comparison
        benign_flat = benign.reshape(benign.shape[0], -1)
        attacks_flat = attacks.reshape(attacks.shape[0], -1)

        # Compute mean over features
        benign_means = benign_flat.mean(axis=1)
        attack_means = attacks_flat.mean(axis=1)

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(benign_means, attack_means)

        if ks_pval > 0.05:
            logger.warning(
                f"⚠️  Attacks may not be statistically different from benign "
                f"(KS p-value={ks_pval:.3f})"
            )
            self.warnings.append(
                f"Attacks not significantly different (p={ks_pval:.3f})"
            )
        else:
            logger.success(
                f"✅ Attacks are statistically different from benign "
                f"(KS stat={ks_stat:.3f}, p={ks_pval:.4f})"
            )

        self.passed_checks += 1
        return True

    def check_detection_quality(self) -> bool:
        """Check that at least some attacks are detected."""
        self.total_checks += 1
        logger.info("Checking detection quality...")

        labels = self.data['labels']
        baseline_pred = self.data['baseline_predictions']
        moirai_pred = self.data['moirai_predictions']

        # Compute accuracy
        n_attacks = labels.sum()
        baseline_detected = baseline_pred[labels == 1].sum()
        moirai_detected = moirai_pred[labels == 1].sum()

        logger.info(f"Ground truth: {n_attacks} attacks out of {len(labels)} samples")
        logger.info(f"Baseline detected: {baseline_detected}/{n_attacks} attacks")
        logger.info(f"Moirai detected: {moirai_detected}/{n_attacks} attacks")

        if baseline_detected == 0:
            logger.warning("⚠️  Baseline IDS detected no attacks")
            self.warnings.append("Baseline detected 0 attacks")

        if moirai_detected == 0:
            logger.warning("⚠️  Moirai detected no attacks")
            self.warnings.append("Moirai detected 0 attacks")

        if baseline_detected > 0 or moirai_detected > 0:
            logger.success(f"✅ At least one method detected attacks")
            self.passed_checks += 1
            return True

        logger.error("❌ No attacks detected by any method")
        self.errors.append("No attacks detected")
        return False

    def check_file_size(self) -> bool:
        """Check that file size is reasonable."""
        self.total_checks += 1
        logger.info("Checking file size...")

        file_size_kb = self.data_path.stat().st_size / 1024
        min_size_kb = 50
        max_size_kb = 5000

        if file_size_kb < min_size_kb:
            logger.warning(f"⚠️  File size very small: {file_size_kb:.1f} KB")
            self.warnings.append(f"Small file size: {file_size_kb:.1f} KB")
        elif file_size_kb > max_size_kb:
            logger.warning(f"⚠️  File size very large: {file_size_kb:.1f} KB")
            self.warnings.append(f"Large file size: {file_size_kb:.1f} KB")
        else:
            logger.success(f"✅ File size reasonable: {file_size_kb:.1f} KB")

        self.passed_checks += 1
        return True

    def print_summary(self):
        """Print verification summary."""
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 60)

        # Metadata
        if 'metadata' in self.data:
            metadata = self.data['metadata'].item()
            logger.info(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
            logger.info(f"Samples: {metadata.get('n_benign', 'N/A')} benign + "
                       f"{metadata.get('n_attacks', 'N/A')} attacks")
            logger.info(f"Attack types: {', '.join(metadata.get('attack_types', []))}")

        logger.info(f"\nChecks passed: {self.passed_checks}/{self.total_checks}")

        if self.warnings:
            logger.info(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  ⚠️  {warning}")

        if self.errors:
            logger.info(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  ❌ {error}")

        logger.info("=" * 60)

        if self.errors:
            logger.error("❌ VERIFICATION FAILED")
            return False
        elif self.warnings:
            logger.warning("⚠️  VERIFICATION PASSED WITH WARNINGS")
            return True
        else:
            logger.success("✅ VERIFICATION PASSED")
            return True

    def verify(self) -> bool:
        """Run all verification checks."""
        if not self.load_data():
            return False

        # Run all checks
        self.check_required_keys()
        self.check_array_shapes()
        self.check_no_invalid_values()
        self.check_score_ranges()
        self.check_binary_predictions()
        self.check_statistical_difference()
        self.check_detection_quality()
        self.check_file_size()

        # Print summary
        return self.print_summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify integrity of pre-computed demo data"
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default='data/synthetic/demo_data.npz',
        help='Path to demo_data.npz file'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DEMO DATA VERIFICATION")
    logger.info("=" * 60)
    logger.info(f"Data file: {args.data_file}\n")

    verifier = DemoDataVerifier(args.data_file)
    success = verifier.verify()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
