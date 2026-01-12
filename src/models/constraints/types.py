"""
Constraint System Type Definitions

This module defines the core data structures for the constraint system.
These types are used throughout the constraint validation and enforcement pipeline.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any, Literal
import numpy as np


@dataclass
class HardConstraint:
    """
    Represents a hard constraint that must be satisfied.

    Hard constraints are strict protocol requirements that generated traffic
    must satisfy to be considered protocol-valid.
    """
    name: str
    constraint_type: Literal['range', 'enum', 'structure', 'timing']
    validation_fn: Callable[[np.ndarray], bool]
    feature_indices: List[int]
    parameters: Dict[str, Any]
    severity: Literal['critical', 'error', 'warning'] = 'error'
    description: str = ''


@dataclass
class SoftConstraint:
    """
    Represents a soft (statistical) constraint for realistic traffic generation.

    Soft constraints ensure generated traffic matches statistical properties
    of benign traffic, enabling stealthy hard-negative attacks.
    """
    name: str
    target_distribution: Literal['normal', 'uniform', 'exponential', 'lognormal', 'categorical', 'custom']
    target_params: Dict[str, float]
    feature_indices: List[int]
    tolerance: float = 0.3  # 30% tolerance by default
    weight: float = 1.0  # Importance for multi-objective optimization
    description: str = ''


@dataclass
class ConstraintViolation:
    """
    Represents a single constraint violation with details for debugging.
    """
    constraint_name: str
    severity: Literal['critical', 'error', 'warning']
    feature_indices: List[int]
    expected_range: Optional[Tuple[float, float]] = None
    actual_value: Optional[float] = None
    deviation_magnitude: float = 0.0  # How far outside valid range (0-1 scale)
    suggestion: str = ''  # Human-readable fix suggestion
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable violation description."""
        msg = f"[{self.severity.upper()}] {self.constraint_name}"
        if self.actual_value is not None and self.expected_range is not None:
            msg += f": value {self.actual_value:.3f} outside range {self.expected_range}"
        if self.suggestion:
            msg += f"\n  → {self.suggestion}"
        return msg


@dataclass
class ValidationReport:
    """
    Comprehensive validation report for a generated sample.

    Contains all violations, statistics, and a validity assessment
    based on the configured strictness level.
    """
    sample_id: Optional[int] = None
    protocol: str = 'unknown'
    violations: List[ConstraintViolation] = field(default_factory=list)
    statistics: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[str] = None

    def is_valid(self, strictness: Literal['strict', 'moderate', 'permissive'] = 'moderate') -> bool:
        """
        Check if sample is valid under given strictness level.

        Args:
            strictness: Validation strictness
                - strict: No violations allowed at all
                - moderate: Only warnings allowed (default)
                - permissive: Only criticals rejected

        Returns:
            True if sample passes validation under the given strictness
        """
        if strictness == 'strict':
            # Reject if any violation at all
            return len(self.violations) == 0
        elif strictness == 'moderate':
            # Reject if error or critical
            return not any(v.severity in ['error', 'critical'] for v in self.violations)
        else:  # permissive
            # Only reject severe criticals (deviation > 0.5)
            return not any(
                v.severity == 'critical' and v.deviation_magnitude > 0.5
                for v in self.violations
            )

    def summary(self) -> str:
        """Generate a human-readable summary of the validation report."""
        if not self.violations:
            return f"✓ {self.protocol.upper()} - All constraints satisfied"

        criticals = sum(1 for v in self.violations if v.severity == 'critical')
        errors = sum(1 for v in self.violations if v.severity == 'error')
        warnings = sum(1 for v in self.violations if v.severity == 'warning')

        summary = f"✗ {self.protocol.upper()} - {len(self.violations)} violations: "
        parts = []
        if criticals:
            parts.append(f"{criticals} critical")
        if errors:
            parts.append(f"{errors} error")
        if warnings:
            parts.append(f"{warnings} warning")
        summary += ", ".join(parts)

        return summary

    def detailed_report(self) -> str:
        """Generate detailed violation report."""
        if not self.violations:
            return self.summary()

        lines = [self.summary(), ""]
        for i, violation in enumerate(self.violations, 1):
            lines.append(f"{i}. {violation}")

        return "\n".join(lines)


# Protocol-specific constants
# These map CICIoT2023 feature indices to protocol-relevant features

# Feature indices (0-based) for typical IoT network flow features
FEATURE_IDX_FLOW_DURATION = 0
FEATURE_IDX_FWD_PKTS_TOT = 1
FEATURE_IDX_BWD_PKTS_TOT = 2
FEATURE_IDX_FWD_DATA_PKTS_TOT = 3
FEATURE_IDX_BWD_DATA_PKTS_TOT = 4
FEATURE_IDX_FWD_PKTS_PER_SEC = 5
FEATURE_IDX_BWD_PKTS_PER_SEC = 6
FEATURE_IDX_FLOW_PKTS_PER_SEC = 7
FEATURE_IDX_FWD_BYTS_B_AVG = 8
FEATURE_IDX_BWD_BYTS_B_AVG = 9
FEATURE_IDX_FWD_IAT_MEAN = 10
FEATURE_IDX_BWD_IAT_MEAN = 11

# Protocol-specific feature mappings
MODBUS_FEATURE_MAP = {
    'packet_size': [FEATURE_IDX_FWD_DATA_PKTS_TOT, FEATURE_IDX_BWD_DATA_PKTS_TOT],
    'timing': [FEATURE_IDX_FWD_IAT_MEAN, FEATURE_IDX_BWD_IAT_MEAN],
    'rate': [FEATURE_IDX_FWD_PKTS_PER_SEC, FEATURE_IDX_BWD_PKTS_PER_SEC],
    'bytes': [FEATURE_IDX_FWD_BYTS_B_AVG, FEATURE_IDX_BWD_BYTS_B_AVG],
}

MQTT_FEATURE_MAP = {
    'packet_size': [FEATURE_IDX_FWD_DATA_PKTS_TOT, FEATURE_IDX_BWD_DATA_PKTS_TOT],
    'timing': [FEATURE_IDX_FWD_IAT_MEAN, FEATURE_IDX_BWD_IAT_MEAN],
    'rate': [FEATURE_IDX_FLOW_PKTS_PER_SEC, FEATURE_IDX_FWD_PKTS_PER_SEC, FEATURE_IDX_BWD_PKTS_PER_SEC],
    'bytes': [FEATURE_IDX_FWD_BYTS_B_AVG, FEATURE_IDX_BWD_BYTS_B_AVG],
    'duration': [FEATURE_IDX_FLOW_DURATION],
}

COAP_FEATURE_MAP = {
    'packet_size': [FEATURE_IDX_FWD_DATA_PKTS_TOT, FEATURE_IDX_BWD_DATA_PKTS_TOT],
    'timing': [FEATURE_IDX_FWD_IAT_MEAN, FEATURE_IDX_BWD_IAT_MEAN],
    'rate': [FEATURE_IDX_FLOW_PKTS_PER_SEC, FEATURE_IDX_FWD_PKTS_PER_SEC, FEATURE_IDX_BWD_PKTS_PER_SEC],
    'bytes': [FEATURE_IDX_FWD_BYTS_B_AVG, FEATURE_IDX_BWD_BYTS_B_AVG],
    'duration': [FEATURE_IDX_FLOW_DURATION],
}
