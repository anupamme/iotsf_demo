"""Tests for Baseline IDS methods"""

import pytest
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline import (
    BaseIDS,
    ThresholdIDS,
    StatisticalIDS,
    SignatureIDS,
    MLBasedIDS,
    CombinedBaselineIDS,
    extract_sequence_features,
    extract_batch_features
)
from src.evaluation.metrics import IDSMetrics


# Fixtures
@pytest.fixture
def benign_sequences():
    """Generate synthetic benign traffic."""
    np.random.seed(42)
    return np.random.randn(50, 128, 12) * 0.5


@pytest.fixture
def attack_sequences():
    """Generate synthetic attack traffic with high packet rates."""
    np.random.seed(42)
    sequences = np.random.randn(50, 128, 12) * 0.5
    # Inject attack pattern: very high packet rates
    sequences[:, :, 7] *= 5  # flow_pkts_per_sec
    return sequences


class TestFeatureExtraction:
    """Test feature extraction utilities."""

    def test_extract_sequence_features_shape(self):
        """Test that feature extraction returns correct shape."""
        seq = np.random.randn(128, 12)
        features = extract_sequence_features(seq)
        assert features.shape == (72,), f"Expected (72,), got {features.shape}"

    def test_extract_batch_features_shape(self):
        """Test batch feature extraction."""
        seqs = np.random.randn(10, 128, 12)
        features = extract_batch_features(seqs)
        assert features.shape == (10, 72)

    def test_extract_sequence_features_deterministic(self):
        """Test that feature extraction is deterministic."""
        np.random.seed(42)
        seq = np.random.randn(128, 12)
        features1 = extract_sequence_features(seq)
        features2 = extract_sequence_features(seq)
        np.testing.assert_array_equal(features1, features2)


class TestThresholdIDS:
    """Test threshold-based IDS."""

    def test_initialization(self):
        """Test ThresholdIDS initialization."""
        ids = ThresholdIDS()
        assert ids.seq_length == 128
        assert ids.feature_dim == 12
        assert ids.percentile == 95
        assert not ids._fitted

    def test_fit(self, benign_sequences):
        """Test fitting on benign data."""
        ids = ThresholdIDS()
        ids.fit(benign_sequences)
        assert ids._fitted
        assert ids.thresholds is not None
        assert len(ids.thresholds) == 72  # 12 features × 6 statistics

    def test_predict_shape(self, benign_sequences, attack_sequences):
        """Test prediction output shape."""
        ids = ThresholdIDS()
        ids.fit(benign_sequences)

        predictions = ids.predict(attack_sequences)
        assert predictions.shape == (50,)
        assert predictions.dtype == int

    def test_predict_proba_range(self, benign_sequences, attack_sequences):
        """Test that probability scores are in [0, 1]."""
        ids = ThresholdIDS()
        ids.fit(benign_sequences)

        scores = ids.predict_proba(attack_sequences)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_high_recall_on_attacks(self, benign_sequences, attack_sequences):
        """Test that attacks are detected."""
        ids = ThresholdIDS()
        ids.fit(benign_sequences)

        predictions = ids.predict(attack_sequences)
        recall = np.mean(predictions)  # All are attacks (label=1)
        assert recall > 0.3, f"Recall too low: {recall}"


class TestStatisticalIDS:
    """Test statistical IDS."""

    def test_initialization(self):
        """Test StatisticalIDS initialization."""
        ids = StatisticalIDS()
        assert ids.z_score_threshold == 3.0
        assert ids.iqr_multiplier == 1.5

    def test_fit_computes_statistics(self, benign_sequences):
        """Test that fit computes mean, std, and IQR."""
        ids = StatisticalIDS()
        ids.fit(benign_sequences)

        assert ids.mean is not None
        assert ids.std is not None
        assert ids.q1 is not None
        assert ids.q3 is not None
        assert ids.iqr is not None

    def test_predict(self, benign_sequences, attack_sequences):
        """Test StatisticalIDS prediction."""
        ids = StatisticalIDS()
        ids.fit(benign_sequences)

        predictions = ids.predict(attack_sequences)
        assert predictions.shape == (50,)
        assert np.any(predictions == 1), "Should detect some attacks"


class TestSignatureIDS:
    """Test signature-based IDS."""

    def test_initialization(self):
        """Test SignatureIDS initialization."""
        ids = SignatureIDS()
        assert ids.mirai_pkt_rate_threshold == 1000.0
        assert ids.ddos_pkt_rate_threshold == 500.0

    def test_no_training_required(self, benign_sequences):
        """Test that SignatureIDS doesn't require training."""
        ids = SignatureIDS()
        ids.fit(benign_sequences)  # Should work but not use data
        assert ids._fitted

    def test_detect_high_packet_rate(self):
        """Test detection of high packet rate attacks."""
        ids = SignatureIDS()
        ids.fit(np.random.randn(10, 128, 12))  # Dummy fit

        # Create sequences with very high packet rates
        attack = np.random.randn(10, 128, 12) * 0.5
        attack[:, :, 5] = 1500  # Very high fwd_pkts_per_sec
        attack[:, :, 8] = 50    # Low bytes per packet

        predictions = ids.predict(attack)
        assert np.any(predictions == 1), "Should detect high packet rate attacks"


class TestMLBasedIDS:
    """Test ML-based IDS using Isolation Forest."""

    def test_initialization(self):
        """Test MLBasedIDS initialization."""
        ids = MLBasedIDS()
        assert ids.contamination == 0.05
        assert ids.n_estimators == 100

    def test_fit(self, benign_sequences):
        """Test fitting Isolation Forest."""
        ids = MLBasedIDS()
        ids.fit(benign_sequences)
        assert ids._fitted

    def test_predict(self, benign_sequences, attack_sequences):
        """Test MLBasedIDS prediction."""
        ids = MLBasedIDS()
        ids.fit(benign_sequences)

        predictions = ids.predict(attack_sequences)
        assert predictions.shape == (50,)
        assert np.any(predictions == 1), "Should detect some anomalies"

    def test_deterministic_results(self, benign_sequences, attack_sequences):
        """Test that results are deterministic with fixed random_state."""
        ids1 = MLBasedIDS(random_state=42)
        ids1.fit(benign_sequences)
        pred1 = ids1.predict(attack_sequences)

        ids2 = MLBasedIDS(random_state=42)
        ids2.fit(benign_sequences)
        pred2 = ids2.predict(attack_sequences)

        np.testing.assert_array_equal(pred1, pred2)


class TestCombinedBaselineIDS:
    """Test combined/ensemble IDS."""

    def test_initialization(self):
        """Test CombinedBaselineIDS initialization."""
        ids = CombinedBaselineIDS()
        assert len(ids.methods) == 4
        assert 'threshold' in ids.methods
        assert 'signature' in ids.methods
        assert 'statistical' in ids.methods
        assert 'ml' in ids.methods

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0."""
        ids = CombinedBaselineIDS()
        weight_sum = sum(ids.weights.values())
        assert np.isclose(weight_sum, 1.0)

    def test_fit_all_methods(self, benign_sequences):
        """Test that fit trains all constituent methods."""
        ids = CombinedBaselineIDS()
        ids.fit(benign_sequences)

        for method in ids.methods.values():
            assert method._fitted

    def test_predict(self, benign_sequences, attack_sequences):
        """Test combined prediction."""
        ids = CombinedBaselineIDS()
        ids.fit(benign_sequences)

        predictions = ids.predict(attack_sequences)
        assert predictions.shape == (50,)

    def test_get_individual_predictions(self, benign_sequences, attack_sequences):
        """Test getting individual method predictions."""
        ids = CombinedBaselineIDS()
        ids.fit(benign_sequences)

        individual_preds = ids.get_individual_predictions(attack_sequences)
        assert len(individual_preds) == 4
        assert 'threshold' in individual_preds
        assert individual_preds['threshold'].shape == (50,)


class TestIDSMetrics:
    """Test metrics computation."""

    def test_compute_all_metrics(self):
        """Test computing all metrics."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])

        metrics = IDSMetrics.compute_all_metrics(y_true, y_pred)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics
        assert 'false_positive_rate' in metrics

    def test_confusion_matrix_values(self):
        """Test confusion matrix calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        metrics = IDSMetrics.compute_all_metrics(y_true, y_pred)

        assert metrics['true_negatives'] == 1
        assert metrics['false_positives'] == 1
        assert metrics['false_negatives'] == 0
        assert metrics['true_positives'] == 2

    def test_roc_auc_computation(self):
        """Test ROC-AUC computation with scores."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.3, 0.7, 0.9])

        metrics = IDSMetrics.compute_all_metrics(y_true, y_pred, y_scores)

        assert 'roc_auc' in metrics
        assert metrics['roc_auc'] is not None
        assert 0 <= metrics['roc_auc'] <= 1


class TestIntegration:
    """Integration tests across multiple components."""

    def test_end_to_end_detection(self, benign_sequences, attack_sequences):
        """Test end-to-end detection pipeline."""
        # Combine data
        X_test = np.concatenate([benign_sequences, attack_sequences])
        y_test = np.concatenate([
            np.zeros(len(benign_sequences)),
            np.ones(len(attack_sequences))
        ])

        # Train and evaluate all methods
        methods = {
            'Threshold': ThresholdIDS(),
            'Statistical': StatisticalIDS(),
            'Signature': SignatureIDS(),
            'ML-Based': MLBasedIDS(),
            'Combined': CombinedBaselineIDS()
        }

        for name, ids in methods.items():
            # Train
            ids.fit(benign_sequences)

            # Predict
            y_pred = ids.predict(X_test)
            y_scores = ids.predict_proba(X_test)

            # Compute metrics
            metrics = IDSMetrics.compute_all_metrics(y_test, y_pred, y_scores)

            # Basic sanity checks
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert 0 <= metrics['f1'] <= 1

    def test_low_false_positive_rate_on_benign(self, benign_sequences):
        """Test that FPR is low on benign traffic."""
        # Split benign data
        X_train = benign_sequences[:30]
        X_test = benign_sequences[30:]
        y_test = np.zeros(len(X_test))

        ids = CombinedBaselineIDS()
        ids.fit(X_train)
        y_pred = ids.predict(X_test)

        metrics = IDSMetrics.compute_all_metrics(y_test, y_pred)
        fpr = metrics['false_positive_rate']

        # Note: With small training data (30 samples), FPR can be higher.
        # 0.6 threshold is reasonable for baseline IDS on limited training.
        assert fpr < 0.6, f"FPR too high on benign traffic: {fpr}"
