"""Tests for MQTT protocol validator."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.constraints.protocols.mqtt import MQTTValidator
from src.models.constraints.types import (
    FEATURE_IDX_FWD_BYTS_B_AVG,
    FEATURE_IDX_BWD_BYTS_B_AVG,
    FEATURE_IDX_FWD_IAT_MEAN,
    FEATURE_IDX_FLOW_DURATION
)


class TestMQTTValidatorInitialization:
    """Test MQTT validator initialization."""

    def test_validator_creation(self):
        """Test creating an MQTT validator."""
        validator = MQTTValidator()
        assert validator.get_protocol_name() == "mqtt"

    def test_validator_has_constraints(self):
        """Test that validator initializes with constraints."""
        validator = MQTTValidator()
        assert len(validator.get_hard_constraints()) > 0
        assert len(validator.get_soft_constraints()) > 0

    def test_validator_with_custom_config(self):
        """Test validator with custom configuration."""
        config = {
            'mqtt': {
                'packet_size_range': [5, 2048],
                'keep_alive_range': [30, 600]
            }
        }
        validator = MQTTValidator(config=config)
        hard_constraints = validator.get_hard_constraints()
        assert len(hard_constraints) > 0


class TestMQTTPacketSizeConstraints:
    """Test MQTT packet size validation."""

    def test_valid_packet_sizes(self):
        """Test validation with typical MQTT packet sizes."""
        validator = MQTTValidator()

        sample = np.zeros((128, 12))
        # Typical MQTT: Small control messages and medium data
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = np.random.lognormal(np.log(128), 0.5, 128)
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = np.random.lognormal(np.log(128), 0.5, 128)
        sample[:, FEATURE_IDX_FLOW_DURATION] = 300.0  # 5 minutes

        report = validator.validate(sample, strictness='moderate')

        # Should not have critical violations
        critical = [v for v in report.violations if v.severity == 'critical']
        assert len(critical) == 0

    def test_packet_size_too_small(self):
        """Test validation with packets below minimum."""
        validator = MQTTValidator()

        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 1.0  # Below minimum
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 1.0
        sample[:, FEATURE_IDX_FLOW_DURATION] = 60.0

        report = validator.validate(sample, strictness='moderate')

        violations = [v for v in report.violations if 'packet_size' in v.constraint_name]
        assert len(violations) > 0

    def test_packet_size_too_large(self):
        """Test validation with excessively large packets."""
        validator = MQTTValidator()

        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 2000.0  # Very large
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 2000.0
        sample[:, FEATURE_IDX_FLOW_DURATION] = 60.0

        report = validator.validate(sample, strictness='moderate')

        violations = [v for v in report.violations if 'packet_size' in v.constraint_name]
        assert len(violations) > 0


class TestMQTTConnectionDuration:
    """Test MQTT connection duration validation."""

    def test_valid_persistent_connection(self):
        """Test validation with typical persistent connection."""
        validator = MQTTValidator()

        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FLOW_DURATION] = 300.0  # 5 minutes
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 128.0
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 128.0

        report = validator.validate(sample, strictness='moderate')

        duration_errors = [v for v in report.violations
                          if 'connection_duration' in v.constraint_name and v.severity == 'error']
        assert len(duration_errors) == 0

    def test_very_short_connection(self):
        """Test validation with suspiciously short connection."""
        validator = MQTTValidator()

        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FLOW_DURATION] = 2.0  # 2 seconds (very short)
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 128.0
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 128.0

        report = validator.validate(sample, strictness='moderate')

        # Should have warning about short duration
        duration_violations = [v for v in report.violations
                              if 'connection_duration' in v.constraint_name]
        assert len(duration_violations) > 0


class TestMQTTValidationReports:
    """Test MQTT validation reporting."""

    def test_validation_report_structure(self):
        """Test that reports have correct structure."""
        validator = MQTTValidator()

        sample = np.random.lognormal(np.log(128), 0.5, (128, 12))
        report = validator.validate(sample)

        assert report.protocol == "mqtt"
        assert hasattr(report, 'violations')
        assert hasattr(report, 'statistics')

    def test_suggestion_for_packet_size_violation(self):
        """Test that violations include helpful suggestions."""
        validator = MQTTValidator()

        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 3000.0  # Too large

        report = validator.validate(sample)

        violations = [v for v in report.violations if 'packet_size' in v.constraint_name]
        if violations:
            assert len(violations[0].suggestion) > 0
            assert 'bytes' in violations[0].suggestion.lower()


class TestMQTTGuidanceHints:
    """Test MQTT guidance hints."""

    def test_guidance_hints_structure(self):
        """Test that guidance hints have expected structure."""
        validator = MQTTValidator()

        hints = validator.get_guidance_hints()

        assert 'protocol' in hints
        assert hints['protocol'] == 'mqtt'
        assert 'port' in hints
        assert hints['port'] == 1883
        assert 'connection_model' in hints
        assert hints['connection_model'] == 'persistent'
        assert 'pattern' in hints
        assert hints['pattern'] == 'pub-sub'

    def test_guidance_hints_with_custom_stats(self):
        """Test guidance hints with custom statistics."""
        validator = MQTTValidator()

        custom = {'custom_field': 42}
        hints = validator.get_guidance_hints(target_statistics=custom)

        assert 'custom_field' in hints
        assert hints['custom_field'] == 42
        assert 'protocol' in hints


class TestMQTTAdvancedValidation:
    """Test advanced MQTT-specific validations."""

    def test_qos_consistency_check(self):
        """Test QoS consistency validation."""
        validator = MQTTValidator()

        # Create sample with reasonable QoS pattern
        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 100.0
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 50.0  # Some ACKs

        result = validator.validate_qos_consistency(sample)
        assert result == True

    def test_message_type_diversity_check(self):
        """Test message type diversity validation."""
        validator = MQTTValidator()

        # Create sample with varied packet sizes
        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = np.concatenate([
            np.ones(64) * 20,   # PINGREQ/DISCONNECT (small)
            np.ones(64) * 200   # PUBLISH (larger)
        ])

        result = validator.validate_message_type_diversity(sample)
        assert result == True
