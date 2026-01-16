"""Tests for Moirai foundation model integration.

NOTE: These tests are skipped because Moirai is not yet implemented.
They define the expected API and behavior for future implementation.
"""

import pytest
import numpy as np

pytestmark = pytest.mark.skip(reason="Moirai not yet implemented")


class TestMoraiInitialization:
    """Test Moirai model initialization."""

    def test_init_with_small_model(self):
        """Should initialize Moirai-small model."""
        from src.models.moirai import MoraiDetector
        detector = MoraiDetector(model_size='small')
        assert detector.model_size == 'small'

    def test_init_with_base_model(self):
        """Should initialize Moirai-base model."""
        from src.models.moirai import MoraiDetector
        detector = MoraiDetector(model_size='base')
        assert detector.model_size == 'base'

    def test_init_from_config(self):
        """Should initialize from config.yaml settings."""
        from src.models.moirai import MoraiDetector
        from src.utils.config import Config

        config = Config()
        detector = MoraiDetector(
            model_size=config.get('models.moirai.model_size'),
            context_length=config.get('models.moirai.context_length')
        )
        assert detector.model_size is not None


class TestMoraiDetection:
    """Test anomaly detection with Moirai."""

    def test_detect_benign_traffic(self):
        """Should classify benign traffic with low anomaly score."""
        from src.models.moirai import MoraiDetector

        detector = MoraiDetector(model_size='small')
        benign_sample = np.random.randn(128, 12)

        score = detector.detect(benign_sample)
        assert 0.0 <= score < 0.3, "Benign traffic should have low anomaly score"

    def test_detect_obvious_attack(self):
        """Should detect clear attacks with high anomaly score."""
        from src.models.moirai import MoraiDetector

        detector = MoraiDetector(model_size='small')
        attack_sample = np.random.randn(128, 12) * 3.0  # Clear anomaly

        score = detector.detect(attack_sample)
        assert 0.7 < score <= 1.0, "Clear attacks should have high anomaly score"

    def test_detect_hard_negative(self):
        """Should detect subtle attacks (key feature for demo)."""
        from src.models.moirai import MoraiDetector

        detector = MoraiDetector(model_size='small')
        # Hard-negative: subtle attack that looks like benign
        hard_negative = np.random.randn(128, 12)
        hard_negative[64:80, 8] += 0.05  # Subtle exfiltration pattern

        score = detector.detect(hard_negative)
        assert 0.5 <= score <= 0.7, "Hard-negatives should have medium anomaly score"

    def test_batch_detection(self):
        """Should process multiple sequences efficiently."""
        from src.models.moirai import MoraiDetector

        detector = MoraiDetector(model_size='small')
        batch = np.random.randn(10, 128, 12)

        scores = detector.detect_batch(batch)
        assert len(scores) == 10
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestMoraiForecasting:
    """Test Moirai's time series forecasting capability."""

    def test_forecast_next_n_steps(self):
        """Should forecast future time steps given context."""
        from src.models.moirai import MoraiDetector

        detector = MoraiDetector(model_size='small')
        context = np.random.randn(128, 12)

        forecast = detector.forecast(context, n_steps=16)
        assert forecast.shape == (16, 12)

    def test_forecast_with_confidence_intervals(self):
        """Should provide uncertainty estimates."""
        from src.models.moirai import MoraiDetector

        detector = MoraiDetector(model_size='small')
        context = np.random.randn(128, 12)

        forecast, lower, upper = detector.forecast_with_uncertainty(context, n_steps=16)
        assert forecast.shape == (16, 12)
        assert lower.shape == (16, 12)
        assert upper.shape == (16, 12)
        assert np.all(lower <= forecast)
        assert np.all(forecast <= upper)


class TestMoraiConfiguration:
    """Test configuration options."""

    def test_context_length_setting(self):
        """Should respect context_length from config."""
        from src.models.moirai import MoraiDetector

        detector = MoraiDetector(model_size='small', context_length=512)
        assert detector.context_length == 512

    def test_prediction_length_setting(self):
        """Should respect prediction_length from config."""
        from src.models.moirai import MoraiDetector

        detector = MoraiDetector(model_size='small', prediction_length=64)
        assert detector.prediction_length == 64

    def test_device_placement(self):
        """Should use configured device (GPU/CPU)."""
        from src.models.moirai import MoraiDetector

        detector = MoraiDetector(model_size='small', device='cpu')
        assert detector.device == 'cpu'
