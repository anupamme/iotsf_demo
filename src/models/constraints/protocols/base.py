"""
Protocol Validator Abstract Base Class

Defines the interface that all protocol-specific validators must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
from loguru import logger

from ..types import (
    HardConstraint,
    SoftConstraint,
    ConstraintViolation,
    ValidationReport
)


class ProtocolValidator(ABC):
    """
    Abstract base class for protocol-specific validators.

    Each protocol (Modbus, MQTT, CoAP) implements this interface
    to provide protocol-specific constraint validation.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the validator with protocol-specific configuration.

        Args:
            config: Protocol-specific configuration dict (from config.yaml)
        """
        self.config = config or {}
        self._hard_constraints: List[HardConstraint] = []
        self._soft_constraints: List[SoftConstraint] = []
        self._initialize_constraints()

    @abstractmethod
    def _initialize_constraints(self):
        """
        Initialize protocol-specific constraints.

        Subclasses must implement this to define their hard and soft constraints.
        """
        pass

    @abstractmethod
    def get_protocol_name(self) -> str:
        """Return the protocol name (e.g., 'modbus', 'mqtt', 'coap')."""
        pass

    def validate(
        self,
        sample: np.ndarray,
        strictness: str = 'moderate'
    ) -> ValidationReport:
        """
        Validate a generated sample against all constraints.

        Args:
            sample: Generated traffic sample of shape (seq_length, feature_dim)
            strictness: Validation strictness level

        Returns:
            ValidationReport with all violations and statistics
        """
        violations: List[ConstraintViolation] = []

        # Validate hard constraints
        for constraint in self._hard_constraints:
            violation = self._check_hard_constraint(sample, constraint)
            if violation:
                violations.append(violation)

        # Validate soft constraints
        for constraint in self._soft_constraints:
            violation = self._check_soft_constraint(sample, constraint, strictness)
            if violation:
                violations.append(violation)

        # Compute statistics
        statistics = self._compute_statistics(sample)

        report = ValidationReport(
            protocol=self.get_protocol_name(),
            violations=violations,
            statistics=statistics
        )

        return report

    def _check_hard_constraint(
        self,
        sample: np.ndarray,
        constraint: HardConstraint
    ) -> Optional[ConstraintViolation]:
        """
        Check a single hard constraint.

        Args:
            sample: Generated traffic sample
            constraint: Hard constraint to check

        Returns:
            ConstraintViolation if constraint is violated, None otherwise
        """
        try:
            # Extract relevant features
            if constraint.feature_indices:
                relevant_data = sample[:, constraint.feature_indices]
            else:
                relevant_data = sample

            # Run validation function
            is_valid = constraint.validation_fn(relevant_data)

            if not is_valid:
                # Compute actual value and deviation
                actual_value = self._compute_representative_value(
                    relevant_data,
                    constraint.constraint_type
                )

                expected_range = constraint.parameters.get('range')
                deviation = self._compute_deviation(
                    actual_value,
                    expected_range,
                    constraint.constraint_type
                )

                return ConstraintViolation(
                    constraint_name=constraint.name,
                    severity=constraint.severity,
                    feature_indices=constraint.feature_indices,
                    expected_range=expected_range,
                    actual_value=actual_value,
                    deviation_magnitude=deviation,
                    suggestion=self._generate_suggestion(constraint, actual_value)
                )

        except Exception as e:
            logger.warning(f"Error checking hard constraint {constraint.name}: {e}")
            return ConstraintViolation(
                constraint_name=constraint.name,
                severity='warning',
                feature_indices=constraint.feature_indices,
                suggestion=f"Validation error: {str(e)}"
            )

        return None

    def _check_soft_constraint(
        self,
        sample: np.ndarray,
        constraint: SoftConstraint,
        strictness: str
    ) -> Optional[ConstraintViolation]:
        """
        Check a single soft (statistical) constraint.

        Args:
            sample: Generated traffic sample
            constraint: Soft constraint to check
            strictness: Validation strictness level

        Returns:
            ConstraintViolation if constraint is violated beyond tolerance
        """
        try:
            # Extract relevant features
            if constraint.feature_indices:
                relevant_data = sample[:, constraint.feature_indices].flatten()
            else:
                relevant_data = sample.flatten()

            # Check statistical properties based on distribution type
            violation = self._check_distribution(
                relevant_data,
                constraint,
                strictness
            )

            return violation

        except Exception as e:
            logger.warning(f"Error checking soft constraint {constraint.name}: {e}")
            return None

    def _check_distribution(
        self,
        data: np.ndarray,
        constraint: SoftConstraint,
        strictness: str
    ) -> Optional[ConstraintViolation]:
        """Check if data matches expected distribution within tolerance."""
        dist_type = constraint.target_distribution
        params = constraint.target_params

        # Adjust tolerance based on strictness
        tolerance_multiplier = {
            'strict': 0.1,
            'moderate': 0.3,
            'permissive': 0.5
        }.get(strictness, 0.3)
        tolerance = constraint.tolerance * tolerance_multiplier

        if dist_type == 'normal':
            target_mean = params.get('mean', 0.0)
            target_std = params.get('std', 1.0)

            actual_mean = float(np.mean(data))
            actual_std = float(np.std(data))

            mean_diff = abs(actual_mean - target_mean) / (abs(target_mean) + 1e-8)
            std_diff = abs(actual_std - target_std) / (abs(target_std) + 1e-8)

            if mean_diff > tolerance or std_diff > tolerance:
                deviation = max(mean_diff, std_diff)
                return ConstraintViolation(
                    constraint_name=constraint.name,
                    severity='warning',
                    feature_indices=constraint.feature_indices,
                    expected_range=(target_mean - tolerance * target_mean,
                                    target_mean + tolerance * target_mean),
                    actual_value=actual_mean,
                    deviation_magnitude=deviation,
                    suggestion=f"Adjust generation to target mean={target_mean:.2f}, std={target_std:.2f}"
                )

        elif dist_type == 'uniform':
            min_val = params.get('min', 0.0)
            max_val = params.get('max', 1.0)

            actual_min = float(np.min(data))
            actual_max = float(np.max(data))

            # Use absolute tolerance based on range (robust for negative/zero values)
            range_size = max_val - min_val
            abs_tolerance = range_size * tolerance

            # Expand bounds using absolute tolerance
            lower_bound = min_val - abs_tolerance
            upper_bound = max_val + abs_tolerance

            # Check if data falls outside expected range
            if actual_min < lower_bound or actual_max > upper_bound:
                # Compute deviation as fraction of range outside bounds
                lower_deviation = max(0, (lower_bound - actual_min) / (range_size + 1e-8))
                upper_deviation = max(0, (actual_max - upper_bound) / (range_size + 1e-8))
                deviation = max(lower_deviation, upper_deviation)

                return ConstraintViolation(
                    constraint_name=constraint.name,
                    severity='warning',
                    feature_indices=constraint.feature_indices,
                    expected_range=(min_val, max_val),
                    actual_value=(actual_min + actual_max) / 2,
                    deviation_magnitude=deviation,
                    suggestion=f"Adjust generation to uniform range [{min_val}, {max_val}]"
                )

        # Add more distribution types as needed

        return None

    def _compute_representative_value(
        self,
        data: np.ndarray,
        constraint_type: str
    ) -> float:
        """Compute a representative value from data based on constraint type."""
        if constraint_type == 'range':
            return float(np.mean(data))
        elif constraint_type == 'timing':
            return float(np.median(data))
        else:
            return float(np.mean(data))

    def _compute_deviation(
        self,
        actual_value: float,
        expected_range: Optional[tuple],
        constraint_type: str
    ) -> float:
        """
        Compute deviation magnitude (0-1 scale).

        0 means within range, 1 means far outside range.
        """
        if expected_range is None:
            return 0.0

        min_val, max_val = expected_range
        range_size = max_val - min_val

        if range_size == 0:
            return 1.0 if actual_value != min_val else 0.0

        if actual_value < min_val:
            deviation = (min_val - actual_value) / range_size
        elif actual_value > max_val:
            deviation = (actual_value - max_val) / range_size
        else:
            return 0.0  # Within range

        return min(deviation, 1.0)  # Cap at 1.0

    @abstractmethod
    def _generate_suggestion(
        self,
        constraint: HardConstraint,
        actual_value: float
    ) -> str:
        """
        Generate human-readable suggestion for fixing violation.

        Subclasses should provide protocol-specific suggestions.
        """
        pass

    def _compute_statistics(self, sample: np.ndarray) -> Dict[str, float]:
        """Compute general statistics about the sample."""
        return {
            'mean': float(np.mean(sample)),
            'std': float(np.std(sample)),
            'min': float(np.min(sample)),
            'max': float(np.max(sample)),
            'shape': sample.shape
        }

    @abstractmethod
    def get_guidance_hints(self, target_statistics: Optional[Dict] = None) -> Dict:
        """
        Get protocol-specific guidance hints for the diffusion generation process.

        Args:
            target_statistics: Optional target statistics to guide generation

        Returns:
            Dict with guidance parameters for constraint-aware generation
        """
        pass

    def get_hard_constraints(self) -> List[HardConstraint]:
        """Return list of hard constraints."""
        return self._hard_constraints

    def get_soft_constraints(self) -> List[SoftConstraint]:
        """Return list of soft constraints."""
        return self._soft_constraints
