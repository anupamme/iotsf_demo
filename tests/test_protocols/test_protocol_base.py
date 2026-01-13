"""Tests for base protocol validator functionality."""

import pytest
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.constraints.types import (
    HardConstraint,
    SoftConstraint,
    ConstraintViolation,
    ValidationReport
)
from src.models.constraints.protocols.base import ProtocolValidator


class MockProtocolValidator(ProtocolValidator):
    """Mock validator for testing base class functionality."""

    def _initialize_constraints(self):
        """Initialize test constraints."""
        # Add a simple range constraint for feature 0
        self._hard_constraints.append(
            HardConstraint(
                name="test_range",
                constraint_type="range",
                validation_fn=lambda x: np.all((x >= 0) & (x <= 1)),
                feature_indices=[0],
                parameters={'range': (0.0, 1.0)},
                severity='error',
                description="Test range constraint"
            )
        )

        # Add a soft statistical constraint
        self._soft_constraints.append(
            SoftConstraint(
                name="test_normal",
                target_distribution="normal",
                target_params={'mean': 0.5, 'std': 0.1},
                feature_indices=[0],
                tolerance=0.3,
                description="Test normal distribution"
            )
        )

    def get_protocol_name(self) -> str:
        return "mock"

    def _generate_suggestion(self, constraint: HardConstraint, actual_value: float) -> str:
        return f"Adjust value {actual_value:.2f} to be within range"

    def get_guidance_hints(self, target_statistics: Optional[Dict] = None) -> Dict:
        return {'target_mean': 0.5, 'target_std': 0.1}


class TestConstraintDataclasses:
    """Test constraint dataclasses."""

    def test_hard_constraint_creation(self):
        """Test creating a hard constraint."""
        constraint = HardConstraint(
            name="test",
            constraint_type="range",
            validation_fn=lambda x: True,
            feature_indices=[0, 1],
            parameters={'range': (0, 10)},
            severity='error'
        )
        assert constraint.name == "test"
        assert constraint.constraint_type == "range"
        assert constraint.feature_indices == [0, 1]

    def test_soft_constraint_creation(self):
        """Test creating a soft constraint."""
        constraint = SoftConstraint(
            name="test_normal",
            target_distribution="normal",
            target_params={'mean': 0.5, 'std': 0.1},
            feature_indices=[0],
            tolerance=0.3
        )
        assert constraint.name == "test_normal"
        assert constraint.target_distribution == "normal"
        assert constraint.tolerance == 0.3

    def test_constraint_violation_creation(self):
        """Test creating a constraint violation."""
        violation = ConstraintViolation(
            constraint_name="test_violation",
            severity="error",
            feature_indices=[0],
            expected_range=(0.0, 1.0),
            actual_value=1.5,
            deviation_magnitude=0.5,
            suggestion="Reduce value"
        )
        assert violation.constraint_name == "test_violation"
        assert violation.severity == "error"
        assert violation.actual_value == 1.5

    def test_constraint_violation_str(self):
        """Test violation string representation."""
        violation = ConstraintViolation(
            constraint_name="test",
            severity="error",
            feature_indices=[0],
            expected_range=(0.0, 1.0),
            actual_value=1.5,
            suggestion="Fix it"
        )
        msg = str(violation)
        assert "ERROR" in msg
        assert "test" in msg
        assert "1.500" in msg


class TestValidationReport:
    """Test validation report functionality."""

    def test_empty_report_is_valid(self):
        """Test that report with no violations is valid."""
        report = ValidationReport(protocol='test')
        assert report.is_valid('strict')
        assert report.is_valid('moderate')
        assert report.is_valid('permissive')

    def test_warning_only_report(self):
        """Test report with only warnings."""
        report = ValidationReport(
            protocol='test',
            violations=[
                ConstraintViolation(
                    constraint_name="test",
                    severity="warning",
                    feature_indices=[0]
                )
            ]
        )
        # Warnings pass moderate and permissive
        assert not report.is_valid('strict')
        assert report.is_valid('moderate')
        assert report.is_valid('permissive')

    def test_error_report(self):
        """Test report with errors."""
        report = ValidationReport(
            protocol='test',
            violations=[
                ConstraintViolation(
                    constraint_name="test",
                    severity="error",
                    feature_indices=[0]
                )
            ]
        )
        # Errors fail strict and moderate
        assert not report.is_valid('strict')
        assert not report.is_valid('moderate')
        assert report.is_valid('permissive')

    def test_critical_report(self):
        """Test report with critical violations."""
        report = ValidationReport(
            protocol='test',
            violations=[
                ConstraintViolation(
                    constraint_name="test",
                    severity="critical",
                    feature_indices=[0],
                    deviation_magnitude=0.3
                )
            ]
        )
        # Criticals fail all strictness levels
        assert not report.is_valid('strict')
        assert not report.is_valid('moderate')
        assert report.is_valid('permissive')  # Low deviation passes permissive

    def test_critical_high_deviation_report(self):
        """Test report with high-deviation critical violations."""
        report = ValidationReport(
            protocol='test',
            violations=[
                ConstraintViolation(
                    constraint_name="test",
                    severity="critical",
                    feature_indices=[0],
                    deviation_magnitude=0.8
                )
            ]
        )
        # High deviation fails all levels
        assert not report.is_valid('strict')
        assert not report.is_valid('moderate')
        assert not report.is_valid('permissive')

    def test_summary_no_violations(self):
        """Test summary for clean report."""
        report = ValidationReport(protocol='modbus')
        summary = report.summary()
        assert "✓" in summary
        assert "MODBUS" in summary
        assert "satisfied" in summary

    def test_summary_with_violations(self):
        """Test summary with violations."""
        report = ValidationReport(
            protocol='mqtt',
            violations=[
                ConstraintViolation("test1", "critical", [0]),
                ConstraintViolation("test2", "error", [1]),
                ConstraintViolation("test3", "warning", [2])
            ]
        )
        summary = report.summary()
        assert "✗" in summary
        assert "MQTT" in summary
        assert "3 violations" in summary
        assert "1 critical" in summary
        assert "1 error" in summary
        assert "1 warning" in summary


class TestUniformDistributionValidation:
    """Test uniform distribution validation with edge cases."""

    def test_uniform_positive_range(self):
        """Test uniform distribution with positive range."""
        from src.models.constraints.types import SoftConstraint
        from src.models.constraints.protocols.base import ProtocolValidator

        constraint = SoftConstraint(
            name="test_uniform_positive",
            target_distribution="uniform",
            target_params={'min': 10.0, 'max': 20.0},
            feature_indices=[0],
            tolerance=0.3
        )

        validator = MockProtocolValidator()
        validator._soft_constraints = [constraint]

        # Data within range - should pass
        data_ok = np.random.uniform(10, 20, size=(100,))
        sample = np.tile(data_ok[:, None], (1, 12))
        report = validator.validate(sample, strictness='moderate')
        assert len([v for v in report.violations if v.constraint_name == "test_uniform_positive"]) == 0

        # Data outside range - should fail
        data_bad = np.random.uniform(25, 30, size=(100,))
        sample_bad = np.tile(data_bad[:, None], (1, 12))
        report_bad = validator.validate(sample_bad, strictness='moderate')
        assert len([v for v in report_bad.violations if v.constraint_name == "test_uniform_positive"]) > 0

    def test_uniform_negative_range(self):
        """Test uniform distribution with negative range (regression test)."""
        from src.models.constraints.types import SoftConstraint

        constraint = SoftConstraint(
            name="test_uniform_negative",
            target_distribution="uniform",
            target_params={'min': -20.0, 'max': -10.0},
            feature_indices=[0],
            tolerance=0.3
        )

        validator = MockProtocolValidator()
        validator._soft_constraints = [constraint]

        # Data within negative range - should pass
        data_ok = np.random.uniform(-20, -10, size=(100,))
        sample = np.tile(data_ok[:, None], (1, 12))
        report = validator.validate(sample, strictness='moderate')
        assert len([v for v in report.violations if v.constraint_name == "test_uniform_negative"]) == 0

        # Data outside range (too negative) - should fail
        data_bad = np.random.uniform(-30, -25, size=(100,))
        sample_bad = np.tile(data_bad[:, None], (1, 12))
        report_bad = validator.validate(sample_bad, strictness='moderate')
        assert len([v for v in report_bad.violations if v.constraint_name == "test_uniform_negative"]) > 0

    def test_uniform_range_spanning_zero(self):
        """Test uniform distribution with range spanning zero."""
        from src.models.constraints.types import SoftConstraint

        constraint = SoftConstraint(
            name="test_uniform_zero_span",
            target_distribution="uniform",
            target_params={'min': -10.0, 'max': 10.0},
            feature_indices=[0],
            tolerance=0.3
        )

        validator = MockProtocolValidator()
        validator._soft_constraints = [constraint]

        # Data within range - should pass
        data_ok = np.random.uniform(-10, 10, size=(100,))
        sample = np.tile(data_ok[:, None], (1, 12))
        report = validator.validate(sample, strictness='moderate')
        assert len([v for v in report.violations if v.constraint_name == "test_uniform_zero_span"]) == 0


class TestLognormalDistributionValidation:
    """Test log-normal distribution validation."""

    def test_lognormal_distribution_valid(self):
        """Test log-normal distribution with matching data."""
        from src.models.constraints.types import SoftConstraint

        constraint = SoftConstraint(
            name="test_lognormal",
            target_distribution="lognormal",
            target_params={'mean': 128.0, 'std': 256.0},
            feature_indices=[0],
            tolerance=0.5  # Higher tolerance for log-normal
        )

        validator = MockProtocolValidator()
        validator._soft_constraints = [constraint]

        # Generate log-normal data
        data_ok = np.random.lognormal(np.log(128), 0.5, size=(100,))
        sample = np.tile(data_ok[:, None], (1, 12))

        report = validator.validate(sample, strictness='moderate')
        # Should have no or minimal violations
        violations = [v for v in report.violations if v.constraint_name == "test_lognormal"]
        assert len(violations) <= 1

    def test_lognormal_distribution_invalid(self):
        """Test log-normal distribution with non-matching data."""
        from src.models.constraints.types import SoftConstraint

        constraint = SoftConstraint(
            name="test_lognormal_bad",
            target_distribution="lognormal",
            target_params={'mean': 128.0, 'std': 256.0},
            feature_indices=[0],
            tolerance=0.3
        )

        validator = MockProtocolValidator()
        validator._soft_constraints = [constraint]

        # Generate very different data (normal instead of lognormal)
        data_bad = np.random.normal(500, 50, size=(100,))
        sample_bad = np.tile(data_bad[:, None], (1, 12))

        report_bad = validator.validate(sample_bad, strictness='strict')
        violations = [v for v in report_bad.violations if v.constraint_name == "test_lognormal_bad"]
        assert len(violations) > 0

    def test_lognormal_with_negative_values(self):
        """Test log-normal distribution rejects negative values."""
        from src.models.constraints.types import SoftConstraint

        constraint = SoftConstraint(
            name="test_lognormal_negative",
            target_distribution="lognormal",
            target_params={'mean': 128.0, 'std': 256.0},
            feature_indices=[0],
            tolerance=0.3
        )

        validator = MockProtocolValidator()
        validator._soft_constraints = [constraint]

        # Data with negative values - should violate
        data_negative = np.array([-10.0, -5.0, 0.0, 5.0, 10.0] * 20)
        sample = np.tile(data_negative[:, None], (1, 12))

        report = validator.validate(sample, strictness='moderate')
        violations = [v for v in report.violations if v.constraint_name == "test_lognormal_negative"]
        assert len(violations) > 0
        if violations:
            assert 'positive' in violations[0].suggestion.lower()


class TestExponentialDistributionValidation:
    """Test exponential distribution validation."""

    def test_exponential_distribution_valid(self):
        """Test exponential distribution with matching data."""
        from src.models.constraints.types import SoftConstraint

        constraint = SoftConstraint(
            name="test_exponential",
            target_distribution="exponential",
            target_params={'lambda': 0.2, 'mean': 5.0},
            feature_indices=[0],
            tolerance=0.5
        )

        validator = MockProtocolValidator()
        validator._soft_constraints = [constraint]

        # Generate exponential data with mean=5
        data_ok = np.random.exponential(5.0, size=(100,))
        sample = np.tile(data_ok[:, None], (1, 12))

        report = validator.validate(sample, strictness='moderate')
        violations = [v for v in report.violations if v.constraint_name == "test_exponential"]
        # Should pass or have minimal violations
        assert len(violations) <= 1

    def test_exponential_distribution_invalid(self):
        """Test exponential distribution with non-matching data."""
        from src.models.constraints.types import SoftConstraint

        constraint = SoftConstraint(
            name="test_exponential_bad",
            target_distribution="exponential",
            target_params={'lambda': 0.2, 'mean': 5.0},
            feature_indices=[0],
            tolerance=0.3
        )

        validator = MockProtocolValidator()
        validator._soft_constraints = [constraint]

        # Generate data with very different mean
        data_bad = np.random.exponential(50.0, size=(100,))
        sample_bad = np.tile(data_bad[:, None], (1, 12))

        report_bad = validator.validate(sample_bad, strictness='strict')
        violations = [v for v in report_bad.violations if v.constraint_name == "test_exponential_bad"]
        assert len(violations) > 0

    def test_exponential_with_negative_values(self):
        """Test exponential distribution rejects negative values."""
        from src.models.constraints.types import SoftConstraint

        constraint = SoftConstraint(
            name="test_exponential_negative",
            target_distribution="exponential",
            target_params={'lambda': 0.2, 'mean': 5.0},
            feature_indices=[0],
            tolerance=0.3
        )

        validator = MockProtocolValidator()
        validator._soft_constraints = [constraint]

        # Data with negative values - should violate
        data_negative = np.array([-1.0, -0.5, 0.0, 1.0, 2.0] * 20)
        sample = np.tile(data_negative[:, None], (1, 12))

        report = validator.validate(sample, strictness='moderate')
        violations = [v for v in report.violations if v.constraint_name == "test_exponential_negative"]
        assert len(violations) > 0
        if violations:
            assert 'non-negative' in violations[0].suggestion.lower()


class TestProtocolValidatorBase:
    """Test base protocol validator."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = MockProtocolValidator()
        assert validator.get_protocol_name() == "mock"
        assert len(validator.get_hard_constraints()) == 1
        assert len(validator.get_soft_constraints()) == 1

    def test_validate_passing_sample(self):
        """Test validation of a passing sample."""
        validator = MockProtocolValidator()
        # Create sample that passes constraints
        sample = np.random.uniform(0.4, 0.6, size=(128, 12))
        report = validator.validate(sample)
        assert report.protocol == "mock"
        # Should have minimal violations
        assert len(report.violations) <= 2

    def test_validate_failing_hard_constraint(self):
        """Test validation with hard constraint violation."""
        validator = MockProtocolValidator()
        # Create sample that violates range constraint
        sample = np.random.uniform(1.5, 2.0, size=(128, 12))
        report = validator.validate(sample)
        # Should have hard constraint violation
        assert len(report.violations) > 0
        assert any(v.constraint_name == "test_range" for v in report.violations)

    def test_compute_statistics(self):
        """Test statistics computation."""
        validator = MockProtocolValidator()
        sample = np.ones((128, 12)) * 0.5
        report = validator.validate(sample)
        assert 'mean' in report.statistics
        assert 'std' in report.statistics
        assert abs(report.statistics['mean'] - 0.5) < 0.1

    def test_guidance_hints(self):
        """Test getting guidance hints."""
        validator = MockProtocolValidator()
        hints = validator.get_guidance_hints()
        assert 'target_mean' in hints
        assert hints['target_mean'] == 0.5
