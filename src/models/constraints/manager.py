"""
IoT Constraint Manager

Orchestrates constraint validation across multiple IoT protocols.
Loads configurations, manages protocol validators, and aggregates reports.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Union
import numpy as np
import yaml
from loguru import logger

from .types import ValidationReport, ConstraintViolation
from .protocols import ProtocolValidator, ModbusValidator, MQTTValidator, CoAPValidator


class IoTConstraintManager:
    """
    Central manager for IoT protocol constraint validation.

    Responsibilities:
    - Load and parse constraint configurations
    - Instantiate protocol-specific validators
    - Orchestrate validation across multiple protocols
    - Aggregate and report violations
    - Support configurable strictness levels
    """

    # Protocol validator registry
    VALIDATOR_REGISTRY = {
        'modbus': ModbusValidator,
        'mqtt': MQTTValidator,
        'coap': CoAPValidator,
    }

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict] = None,
        protocol: Optional[str] = None,
        strictness: Optional[Literal['strict', 'moderate', 'permissive']] = None
    ):
        """
        Initialize the constraint manager.

        Args:
            config_path: Path to config.yaml file
            config_dict: Configuration dictionary (alternative to config_path)
            protocol: Override default protocol ('modbus', 'mqtt', 'coap')
            strictness: Override default strictness level
        """
        self.config = self._load_config(config_path, config_dict)
        self.constraints_config = self.config.get('constraints', {})

        # Check if constraints are enabled
        if not self.constraints_config.get('enabled', True):
            logger.warning("Constraint system is disabled in configuration")

        # Set protocol and strictness (with override support)
        self.default_protocol = protocol or self.constraints_config.get('default_protocol', 'modbus')
        self.default_strictness = strictness or self.constraints_config.get('strictness', 'moderate')

        # Initialize protocol validators
        self.validators: Dict[str, ProtocolValidator] = {}
        self._initialize_validators()

        logger.info(
            f"IoTConstraintManager initialized: "
            f"protocol={self.default_protocol}, strictness={self.default_strictness}, "
            f"validators={list(self.validators.keys())}"
        )

    def _load_config(
        self,
        config_path: Optional[Union[str, Path]],
        config_dict: Optional[Dict]
    ) -> Dict:
        """Load configuration from file or dict."""
        if config_dict is not None:
            return config_dict

        if config_path is None:
            # Default to project config.yaml
            config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'config.yaml'

        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using default configuration")
            return {'constraints': {'enabled': True}}

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _initialize_validators(self):
        """Initialize protocol validators based on configuration."""
        protocols_config = self.constraints_config.get('protocols', {})

        for protocol_name, protocol_config in protocols_config.items():
            # Check if protocol is enabled
            if not protocol_config.get('enabled', True):
                logger.debug(f"Protocol {protocol_name} is disabled, skipping validator")
                continue

            # Get validator class from registry
            validator_class = self.VALIDATOR_REGISTRY.get(protocol_name)
            if validator_class is None:
                logger.warning(f"Unknown protocol: {protocol_name}, skipping")
                continue

            # Instantiate validator with protocol-specific config
            try:
                validator = validator_class(config={protocol_name: protocol_config})
                self.validators[protocol_name] = validator
                logger.debug(f"Initialized {protocol_name} validator")
            except Exception as e:
                logger.error(f"Failed to initialize {protocol_name} validator: {e}")

    def validate(
        self,
        sample: np.ndarray,
        protocol: Optional[str] = None,
        strictness: Optional[Literal['strict', 'moderate', 'permissive']] = None
    ) -> ValidationReport:
        """
        Validate a sample against protocol constraints.

        Args:
            sample: Generated traffic sample of shape (seq_length, feature_dim)
            protocol: Protocol to validate against (default: manager's default_protocol)
            strictness: Strictness level (default: manager's default_strictness)

        Returns:
            ValidationReport with violations and statistics
        """
        protocol = protocol or self.default_protocol
        strictness = strictness or self.default_strictness

        # Check if constraints are enabled
        if not self.constraints_config.get('enabled', True):
            logger.debug("Constraints disabled, returning empty report")
            return ValidationReport(protocol=protocol, violations=[])

        # Get validator for protocol
        validator = self.validators.get(protocol)
        if validator is None:
            logger.warning(f"No validator found for protocol: {protocol}")
            return ValidationReport(
                protocol=protocol,
                violations=[
                    ConstraintViolation(
                        constraint_name='validator_missing',
                        severity='warning',
                        feature_indices=[],
                        suggestion=f"No validator configured for protocol '{protocol}'"
                    )
                ]
            )

        # Run validation
        try:
            report = validator.validate(sample, strictness=strictness)
            logger.debug(
                f"Validation complete: protocol={protocol}, strictness={strictness}, "
                f"violations={len(report.violations)}, valid={report.is_valid(strictness)}"
            )
            return report
        except Exception as e:
            logger.error(f"Validation error for protocol {protocol}: {e}")
            return ValidationReport(
                protocol=protocol,
                violations=[
                    ConstraintViolation(
                        constraint_name='validation_error',
                        severity='error',
                        feature_indices=[],
                        suggestion=f"Validation failed: {str(e)}"
                    )
                ]
            )

    def validate_batch(
        self,
        samples: np.ndarray,
        protocol: Optional[str] = None,
        strictness: Optional[Literal['strict', 'moderate', 'permissive']] = None
    ) -> List[ValidationReport]:
        """
        Validate a batch of samples.

        Args:
            samples: Batch of samples, shape (batch_size, seq_length, feature_dim)
            protocol: Protocol to validate against
            strictness: Strictness level

        Returns:
            List of ValidationReport objects, one per sample
        """
        reports = []
        for i, sample in enumerate(samples):
            report = self.validate(sample, protocol=protocol, strictness=strictness)
            report.sample_id = i
            reports.append(report)

        return reports

    def validate_multi_protocol(
        self,
        sample: np.ndarray,
        protocols: Optional[List[str]] = None,
        strictness: Optional[Literal['strict', 'moderate', 'permissive']] = None
    ) -> Dict[str, ValidationReport]:
        """
        Validate a sample against multiple protocols.

        Args:
            sample: Generated traffic sample
            protocols: List of protocols to validate against (default: all enabled)
            strictness: Strictness level

        Returns:
            Dict mapping protocol name to ValidationReport
        """
        if protocols is None:
            protocols = list(self.validators.keys())

        reports = {}
        for protocol in protocols:
            reports[protocol] = self.validate(sample, protocol=protocol, strictness=strictness)

        return reports

    def aggregate_violations(
        self,
        reports: List[ValidationReport]
    ) -> Dict[str, Any]:
        """
        Aggregate violations across multiple reports.

        Args:
            reports: List of validation reports

        Returns:
            Dictionary with aggregated violation statistics
        """
        total_violations = sum(len(r.violations) for r in reports)
        total_samples = len(reports)

        # Count by severity
        severity_counts = {'critical': 0, 'error': 0, 'warning': 0}
        for report in reports:
            for violation in report.violations:
                severity_counts[violation.severity] += 1

        # Count by constraint name
        constraint_counts: Dict[str, int] = {}
        for report in reports:
            for violation in report.violations:
                constraint_counts[violation.constraint_name] = \
                    constraint_counts.get(violation.constraint_name, 0) + 1

        # Calculate pass rate by strictness
        pass_rates = {}
        for strictness in ['strict', 'moderate', 'permissive']:
            passed = sum(1 for r in reports if r.is_valid(strictness))
            pass_rates[strictness] = passed / total_samples if total_samples > 0 else 0.0

        return {
            'total_samples': total_samples,
            'total_violations': total_violations,
            'violations_per_sample': total_violations / total_samples if total_samples > 0 else 0.0,
            'severity_counts': severity_counts,
            'constraint_counts': constraint_counts,
            'pass_rates': pass_rates,
            'violation_rate': 1.0 - pass_rates.get(self.default_strictness, 0.0)
        }

    def get_guidance_hints(
        self,
        protocol: Optional[str] = None,
        target_statistics: Optional[Dict] = None
    ) -> Dict:
        """
        Get protocol-specific guidance hints for generation.

        Args:
            protocol: Protocol to get hints for (default: manager's default_protocol)
            target_statistics: Optional target statistics to merge

        Returns:
            Dictionary with guidance parameters
        """
        protocol = protocol or self.default_protocol
        validator = self.validators.get(protocol)

        if validator is None:
            logger.warning(f"No validator for protocol {protocol}, returning empty hints")
            return {}

        return validator.get_guidance_hints(target_statistics)

    def is_enabled(self) -> bool:
        """Check if constraint system is enabled."""
        return self.constraints_config.get('enabled', True)

    def get_available_protocols(self) -> List[str]:
        """Get list of available protocols with validators."""
        return list(self.validators.keys())

    def get_validator(self, protocol: str) -> Optional[ProtocolValidator]:
        """Get validator for a specific protocol."""
        return self.validators.get(protocol)

    def summary(self) -> str:
        """Generate a summary of the constraint manager configuration."""
        lines = [
            "IoTConstraintManager Configuration:",
            f"  Enabled: {self.is_enabled()}",
            f"  Default Protocol: {self.default_protocol}",
            f"  Default Strictness: {self.default_strictness}",
            f"  Available Protocols: {', '.join(self.get_available_protocols())}"
        ]

        for protocol, validator in self.validators.items():
            hard_count = len(validator.get_hard_constraints())
            soft_count = len(validator.get_soft_constraints())
            lines.append(f"    {protocol}: {hard_count} hard, {soft_count} soft constraints")

        return "\n".join(lines)
