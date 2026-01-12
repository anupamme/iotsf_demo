"""Tests for Modbus TCP protocol validator."""

import pytest
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.constraints.protocols.modbus import ModbusValidator
from src.models.constraints.types import (
    FEATURE_IDX_FWD_BYTS_B_AVG,
    FEATURE_IDX_BWD_BYTS_B_AVG,
    FEATURE_IDX_FWD_IAT_MEAN,
    FEATURE_IDX_BWD_IAT_MEAN
)


class TestModbusValidatorInitialization:
    """Test Modbus validator initialization."""

    def test_validator_creation(self):
        """Test creating a Modbus validator."""
        validator = ModbusValidator()
        assert validator.get_protocol_name() == "modbus"

    def test_validator_has_constraints(self):
        """Test that validator initializes with constraints."""
        validator = ModbusValidator()
        hard_constraints = validator.get_hard_constraints()
        soft_constraints = validator.get_soft_constraints()

        assert len(hard_constraints) > 0, "Should have hard constraints"
        assert len(soft_constraints) > 0, "Should have soft constraints"

    def test_validator_with_custom_config(self):
        """Test validator with custom configuration."""
        config = {
            'modbus': {
                'packet_size_range': [10, 250],
                'timing_min_ms': 5,
                'timing_max_ms': 200
            }
        }
        validator = ModbusValidator(config=config)
        hard_constraints = validator.get_hard_constraints()

        # Check that custom config is applied
        packet_size_constraint = next(
            (c for c in hard_constraints if 'packet_size' in c.name), None
        )
        assert packet_size_constraint is not None
        assert packet_size_constraint.parameters['range'] == (10, 250)


class TestModbusPacketSizeConstraints:
    """Test Modbus packet size validation."""

    def test_valid_packet_sizes(self):
        """Test validation with valid Modbus packet sizes."""
        validator = ModbusValidator()

        # Create sample with typical Modbus packet sizes (64 bytes average)
        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = np.random.normal(64, 20, 128)
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = np.random.normal(64, 20, 128)

        report = validator.validate(sample, strictness='moderate')

        # Should not have critical packet size violations
        packet_violations = [v for v in report.violations
                            if 'packet_size' in v.constraint_name and v.severity == 'critical']
        assert len(packet_violations) == 0

    def test_packet_size_too_small(self):
        """Test validation with packets below minimum size."""
        validator = ModbusValidator()

        # Create sample with packets below minimum (7 bytes MBAP header minimum)
        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 5.0  # Below minimum
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 5.0

        report = validator.validate(sample, strictness='moderate')

        # Should have packet size violations
        fwd_violations = [v for v in report.violations
                         if 'fwd_packet_size' in v.constraint_name]
        assert len(fwd_violations) > 0

    def test_packet_size_too_large(self):
        """Test validation with packets exceeding maximum size."""
        validator = ModbusValidator()

        # Create sample with packets above maximum (260 bytes)
        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 300.0  # Above maximum
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 300.0

        report = validator.validate(sample, strictness='moderate')

        # Should have packet size violations
        violations = [v for v in report.violations
                     if 'packet_size' in v.constraint_name]
        assert len(violations) > 0

    def test_packet_size_at_boundaries(self):
        """Test validation at exact boundary values."""
        validator = ModbusValidator()

        # Test minimum boundary (8 bytes = 7 byte header + 1 byte PDU)
        sample_min = np.zeros((128, 12))
        sample_min[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 8.0
        sample_min[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 8.0

        report_min = validator.validate(sample_min, strictness='moderate')
        # Should pass (at minimum)
        critical_violations = [v for v in report_min.violations if v.severity == 'critical']
        assert len(critical_violations) == 0

        # Test maximum boundary (260 bytes)
        sample_max = np.zeros((128, 12))
        sample_max[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 260.0
        sample_max[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 260.0

        report_max = validator.validate(sample_max, strictness='moderate')
        # Should pass (at maximum)
        critical_violations = [v for v in report_max.violations if v.severity == 'critical']
        assert len(critical_violations) == 0


class TestModbusTimingConstraints:
    """Test Modbus timing validation."""

    def test_valid_timing(self):
        """Test validation with valid Modbus timing."""
        validator = ModbusValidator()

        sample = np.zeros((128, 12))
        # Typical response time: 10-50ms (0.01-0.05 seconds)
        sample[:, FEATURE_IDX_FWD_IAT_MEAN] = np.random.uniform(0.01, 0.05, 128)
        sample[:, FEATURE_IDX_BWD_IAT_MEAN] = np.random.uniform(0.01, 0.05, 128)
        # Valid packet sizes
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 64.0
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 64.0

        report = validator.validate(sample, strictness='moderate')

        # Timing violations should be warnings at most, not errors
        timing_errors = [v for v in report.violations
                        if 'timing' in v.constraint_name and v.severity == 'error']
        assert len(timing_errors) == 0

    def test_timing_too_fast(self):
        """Test validation with unrealistically fast timing."""
        validator = ModbusValidator()

        sample = np.zeros((128, 12))
        # Unrealistically fast: 0.1ms (0.0001 seconds)
        sample[:, FEATURE_IDX_FWD_IAT_MEAN] = 0.0001
        sample[:, FEATURE_IDX_BWD_IAT_MEAN] = 0.0001
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 64.0
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 64.0

        report = validator.validate(sample, strictness='moderate')

        # Should have timing warnings (fast timing is suspicious but not critical)
        timing_violations = [v for v in report.violations if 'timing' in v.constraint_name]
        assert len(timing_violations) > 0

    def test_timing_too_slow(self):
        """Test validation with slow timing (near timeout)."""
        validator = ModbusValidator()

        sample = np.zeros((128, 12))
        # Very slow: 2 seconds (beyond typical timeout)
        sample[:, FEATURE_IDX_FWD_IAT_MEAN] = 2.0
        sample[:, FEATURE_IDX_BWD_IAT_MEAN] = 2.0
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 64.0
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 64.0

        report = validator.validate(sample, strictness='moderate')

        # Should have timing warnings
        timing_violations = [v for v in report.violations if 'timing' in v.constraint_name]
        assert len(timing_violations) > 0


class TestModbusSoftConstraints:
    """Test Modbus statistical (soft) constraints."""

    def test_packet_size_distribution(self):
        """Test packet size distribution constraint."""
        validator = ModbusValidator()

        # Create sample matching expected distribution (Normal(64, 32))
        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = np.random.normal(64, 32, 128)
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = np.random.normal(64, 32, 128)
        # Add valid timing
        sample[:, FEATURE_IDX_FWD_IAT_MEAN] = 0.05
        sample[:, FEATURE_IDX_BWD_IAT_MEAN] = 0.05

        report = validator.validate(sample, strictness='moderate')

        # Should have minimal soft constraint violations
        soft_violations = [v for v in report.violations
                          if v.constraint_name == 'modbus_packet_size_distribution']
        # Distribution should be reasonably close
        assert len(soft_violations) <= 1

    def test_packet_size_distribution_mismatch(self):
        """Test packet size distribution that doesn't match expected."""
        validator = ModbusValidator()

        # Create sample with very different distribution
        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = np.random.normal(150, 10, 128)  # Wrong mean
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = np.random.normal(150, 10, 128)
        sample[:, FEATURE_IDX_FWD_IAT_MEAN] = 0.05
        sample[:, FEATURE_IDX_BWD_IAT_MEAN] = 0.05

        report = validator.validate(sample, strictness='strict')

        # Should have soft constraint violations in strict mode
        soft_violations = [v for v in report.violations
                          if 'distribution' in v.constraint_name]
        assert len(soft_violations) > 0


class TestModbusValidationReports:
    """Test Modbus validation reporting."""

    def test_validation_report_structure(self):
        """Test that validation reports have correct structure."""
        validator = ModbusValidator()

        sample = np.random.randn(128, 12) * 10 + 64
        report = validator.validate(sample)

        assert report.protocol == "modbus"
        assert hasattr(report, 'violations')
        assert hasattr(report, 'statistics')
        assert isinstance(report.violations, list)

    def test_suggestion_for_packet_size_violation(self):
        """Test that violations include helpful suggestions."""
        validator = ModbusValidator()

        # Create sample with packet size violation
        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 300.0  # Too large

        report = validator.validate(sample)

        # Find packet size violation
        packet_violations = [v for v in report.violations
                            if 'packet_size' in v.constraint_name]

        if packet_violations:
            violation = packet_violations[0]
            assert len(violation.suggestion) > 0
            assert 'bytes' in violation.suggestion.lower()

    def test_strictness_levels(self):
        """Test validation at different strictness levels."""
        validator = ModbusValidator()

        # Create sample with minor violations
        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 75.0  # Slightly off from typical
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 75.0
        sample[:, FEATURE_IDX_FWD_IAT_MEAN] = 0.05
        sample[:, FEATURE_IDX_BWD_IAT_MEAN] = 0.05

        report_strict = validator.validate(sample, strictness='strict')
        report_moderate = validator.validate(sample, strictness='moderate')
        report_permissive = validator.validate(sample, strictness='permissive')

        # Strict should be most restrictive
        assert not report_strict.is_valid('strict') or report_strict.is_valid('moderate')


class TestModbusGuidanceHints:
    """Test Modbus guidance hints for generation."""

    def test_guidance_hints_structure(self):
        """Test that guidance hints have expected structure."""
        validator = ModbusValidator()

        hints = validator.get_guidance_hints()

        assert 'target_mean' in hints
        assert 'target_std' in hints
        assert 'protocol' in hints
        assert hints['protocol'] == 'modbus'
        assert 'port' in hints
        assert hints['port'] == 502

    def test_guidance_hints_with_custom_stats(self):
        """Test guidance hints with custom target statistics."""
        validator = ModbusValidator()

        custom_stats = {'custom_param': 123}
        hints = validator.get_guidance_hints(target_statistics=custom_stats)

        # Should include custom stats
        assert 'custom_param' in hints
        assert hints['custom_param'] == 123

        # Should still have default hints
        assert 'protocol' in hints
