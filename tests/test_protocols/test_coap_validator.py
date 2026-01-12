"""Tests for CoAP protocol validator."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.constraints.protocols.coap import CoAPValidator
from src.models.constraints.types import (
    FEATURE_IDX_FWD_BYTS_B_AVG,
    FEATURE_IDX_BWD_BYTS_B_AVG,
    FEATURE_IDX_FWD_IAT_MEAN
)


class TestCoAPValidatorInitialization:
    """Test CoAP validator initialization."""

    def test_validator_creation(self):
        """Test creating a CoAP validator."""
        validator = CoAPValidator()
        assert validator.get_protocol_name() == "coap"

    def test_validator_has_constraints(self):
        """Test that validator initializes with constraints."""
        validator = CoAPValidator()
        assert len(validator.get_hard_constraints()) > 0
        assert len(validator.get_soft_constraints()) > 0


class TestCoAPPacketSizeConstraints:
    """Test CoAP packet size validation."""

    def test_valid_packet_sizes(self):
        """Test validation with typical CoAP packet sizes."""
        validator = CoAPValidator()

        sample = np.zeros((128, 12))
        # Typical CoAP: 4-byte header + payload
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = np.random.normal(100, 50, 128)
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = np.random.normal(150, 80, 128)
        sample[:, FEATURE_IDX_FWD_IAT_MEAN] = 0.1

        report = validator.validate(sample, strictness='moderate')

        critical = [v for v in report.violations if v.severity == 'critical']
        assert len(critical) == 0

    def test_packet_size_too_small(self):
        """Test validation with packets below minimum."""
        validator = CoAPValidator()

        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 2.0  # Below 4-byte header minimum
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 2.0

        report = validator.validate(sample, strictness='moderate')

        violations = [v for v in report.violations if 'packet_size' in v.constraint_name]
        assert len(violations) > 0

    def test_packet_size_too_large(self):
        """Test validation with packets exceeding UDP limit."""
        validator = CoAPValidator()

        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 1500.0  # Exceeds typical limit
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 1500.0

        report = validator.validate(sample, strictness='moderate')

        violations = [v for v in report.violations if 'packet_size' in v.constraint_name]
        assert len(violations) > 0

    def test_packet_size_at_boundaries(self):
        """Test validation at boundary values."""
        validator = CoAPValidator()

        # Test minimum (4 bytes)
        sample_min = np.zeros((128, 12))
        sample_min[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 4.0
        sample_min[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 4.0

        report_min = validator.validate(sample_min, strictness='moderate')
        critical = [v for v in report_min.violations if v.severity == 'critical']
        assert len(critical) == 0


class TestCoAPTimingConstraints:
    """Test CoAP timing validation."""

    def test_valid_timing(self):
        """Test validation with valid CoAP timing."""
        validator = CoAPValidator()

        sample = np.zeros((128, 12))
        # Typical request-response timing
        sample[:, FEATURE_IDX_FWD_IAT_MEAN] = np.random.uniform(0.1, 5.0, 128)
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 100.0
        sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = 150.0

        report = validator.validate(sample, strictness='moderate')

        timing_errors = [v for v in report.violations
                        if 'timing' in v.constraint_name and v.severity == 'error']
        assert len(timing_errors) == 0


class TestCoAPValidationReports:
    """Test CoAP validation reporting."""

    def test_validation_report_structure(self):
        """Test that reports have correct structure."""
        validator = CoAPValidator()

        sample = np.random.normal(128, 50, (128, 12))
        report = validator.validate(sample)

        assert report.protocol == "coap"
        assert hasattr(report, 'violations')
        assert hasattr(report, 'statistics')

    def test_suggestion_for_packet_size_violation(self):
        """Test that violations include helpful suggestions."""
        validator = CoAPValidator()

        sample = np.zeros((128, 12))
        sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = 2000.0  # Too large

        report = validator.validate(sample)

        violations = [v for v in report.violations if 'packet_size' in v.constraint_name]
        if violations:
            assert len(violations[0].suggestion) > 0
            assert 'bytes' in violations[0].suggestion.lower()


class TestCoAPGuidanceHints:
    """Test CoAP guidance hints."""

    def test_guidance_hints_structure(self):
        """Test that guidance hints have expected structure."""
        validator = CoAPValidator()

        hints = validator.get_guidance_hints()

        assert 'protocol' in hints
        assert hints['protocol'] == 'coap'
        assert 'port' in hints
        assert hints['port'] == 5683
        assert 'transport' in hints
        assert hints['transport'] == 'udp'
        assert 'pattern' in hints
        assert hints['pattern'] == 'request-response'

    def test_guidance_hints_with_custom_stats(self):
        """Test guidance hints with custom statistics."""
        validator = CoAPValidator()

        custom = {'custom_field': 99}
        hints = validator.get_guidance_hints(target_statistics=custom)

        assert 'custom_field' in hints
        assert hints['custom_field'] == 99
        assert 'protocol' in hints
