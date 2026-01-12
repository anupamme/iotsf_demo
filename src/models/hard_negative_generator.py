"""
Hard Negative Generator with Constraint Validation

Wraps IoTDiffusionGenerator and integrates IoTConstraintManager to generate
protocol-valid hard-negative attacks that satisfy both statistical mimicry
and protocol constraints.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Literal
from pathlib import Path
from loguru import logger

from .diffusion_ts import IoTDiffusionGenerator
from .constraints.manager import IoTConstraintManager
from .constraints.types import ValidationReport


class HardNegativeGenerator:
    """
    Constraint-aware hard-negative attack generator.

    This class combines IoTDiffusionGenerator with IoTConstraintManager to:
    - Generate statistically stealthy attacks that mimic benign traffic
    - Ensure generated attacks satisfy protocol-specific constraints
    - Automatically retry generation with adjusted guidance on validation failure
    - Provide detailed validation reports for quality assurance

    Architecture:
    - Composition over inheritance (wraps IoTDiffusionGenerator)
    - Maintains 100% backward compatibility with existing generator
    - Adds optional constraint validation with configurable strictness
    """

    def __init__(
        self,
        diffusion_generator: Optional[IoTDiffusionGenerator] = None,
        constraint_manager: Optional[IoTConstraintManager] = None,
        seq_length: int = 128,
        feature_dim: int = 12,
        n_diffusion_steps: int = 1000,
        device: str = 'auto',
        max_retries: int = 3,
        strictness: Literal['strict', 'moderate', 'permissive'] = 'moderate'
    ):
        """
        Initialize the hard-negative generator with constraints.

        Args:
            diffusion_generator: Existing IoTDiffusionGenerator instance (optional)
            constraint_manager: Existing IoTConstraintManager instance (optional)
            seq_length: Length of generated sequences
            feature_dim: Number of features
            n_diffusion_steps: Number of diffusion steps
            device: Device for generation ('auto', 'cuda', or 'cpu')
            max_retries: Maximum number of retries on validation failure
            strictness: Validation strictness level
        """
        # Initialize or use provided diffusion generator
        if diffusion_generator is not None:
            self.generator = diffusion_generator
        else:
            self.generator = IoTDiffusionGenerator(
                seq_length=seq_length,
                feature_dim=feature_dim,
                n_diffusion_steps=n_diffusion_steps,
                device=device
            )

        # Initialize or use provided constraint manager
        self.constraint_manager = constraint_manager

        # Configuration
        self.max_retries = max_retries
        self.default_strictness = strictness

        # Statistics
        self.generation_stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'failed_validations': 0,
            'retry_counts': []
        }

        logger.info(
            f"HardNegativeGenerator initialized: "
            f"max_retries={max_retries}, strictness={strictness}, "
            f"constraints_enabled={constraint_manager is not None}"
        )

    def initialize(self, checkpoint_path: Optional[str] = None):
        """Initialize the underlying diffusion generator."""
        self.generator.initialize(checkpoint_path)

    def generate_constrained(
        self,
        n_samples: int = 1,
        protocol: Optional[str] = None,
        attack_pattern: Optional[str] = None,
        benign_sample: Optional[np.ndarray] = None,
        stealth_level: float = 0.95,
        strictness: Optional[Literal['strict', 'moderate', 'permissive']] = None,
        target_statistics: Optional[Dict] = None,
        guidance_scale: Optional[float] = None,
        validate: bool = True
    ) -> Tuple[np.ndarray, List[ValidationReport], Dict]:
        """
        Generate protocol-constrained hard-negative attacks.

        Args:
            n_samples: Number of samples to generate
            protocol: Target protocol ('modbus', 'mqtt', 'coap')
            attack_pattern: Attack pattern ('slow_exfiltration', 'lotl_mimicry', etc.)
            benign_sample: Optional reference benign traffic to mimic
            stealth_level: Stealth level (0-1, higher = stealthier)
            strictness: Validation strictness (overrides default)
            target_statistics: Optional target statistics for guidance
            guidance_scale: Optional guidance scale (overrides stealth-based)
            validate: Whether to validate constraints (default: True)

        Returns:
            Tuple of (generated_samples, validation_reports, metadata)
        """
        strictness = strictness or self.default_strictness
        samples = []
        reports = []
        metadata = {
            'protocol': protocol,
            'attack_pattern': attack_pattern,
            'stealth_level': stealth_level,
            'strictness': strictness,
            'validation_enabled': validate and self.constraint_manager is not None
        }

        for i in range(n_samples):
            sample, report, sample_meta = self._generate_single_with_retry(
                protocol=protocol,
                attack_pattern=attack_pattern,
                benign_sample=benign_sample,
                stealth_level=stealth_level,
                strictness=strictness,
                target_statistics=target_statistics,
                guidance_scale=guidance_scale,
                validate=validate
            )
            samples.append(sample)
            reports.append(report)

            # Aggregate metadata
            if i == 0:
                metadata.update(sample_meta)

        return np.array(samples), reports, metadata

    def _generate_single_with_retry(
        self,
        protocol: Optional[str],
        attack_pattern: Optional[str],
        benign_sample: Optional[np.ndarray],
        stealth_level: float,
        strictness: str,
        target_statistics: Optional[Dict],
        guidance_scale: Optional[float],
        validate: bool
    ) -> Tuple[np.ndarray, Optional[ValidationReport], Dict]:
        """
        Generate a single sample with validation and retry logic.

        Returns:
            Tuple of (sample, validation_report, metadata)
        """
        self.generation_stats['total_attempts'] += 1

        for attempt in range(self.max_retries + 1):
            # Generate sample
            if attack_pattern and benign_sample is not None:
                # Generate hard-negative with attack pattern
                sample, gen_meta = self.generator.generate_hard_negative(
                    benign_sample=benign_sample,
                    attack_pattern=attack_pattern,
                    stealth_level=stealth_level
                )
            else:
                # Standard generation
                stats = target_statistics or {}
                scale = guidance_scale if guidance_scale is not None else (stealth_level * 5)
                sample = self.generator.generate(
                    n_samples=1,
                    target_statistics=stats if stats else None,
                    guidance_scale=scale if stats else 1.0
                )[0]
                gen_meta = {'generation_type': 'standard'}

            # Validate if enabled
            if validate and self.constraint_manager is not None and protocol is not None:
                report = self.constraint_manager.validate(
                    sample, protocol=protocol, strictness=strictness
                )

                # Check if validation passed
                if report.is_valid(strictness):
                    # Success!
                    self.generation_stats['successful_generations'] += 1
                    self.generation_stats['retry_counts'].append(attempt)

                    metadata = {
                        **gen_meta,
                        'attempts': attempt + 1,
                        'validation_passed': True,
                        'validation_strictness': strictness
                    }

                    logger.debug(
                        f"Sample generated successfully on attempt {attempt + 1}: "
                        f"{len(report.violations)} warnings"
                    )

                    return sample, report, metadata
                else:
                    # Validation failed
                    self.generation_stats['failed_validations'] += 1

                    logger.debug(
                        f"Validation failed on attempt {attempt + 1}/{self.max_retries + 1}: "
                        f"{len(report.violations)} violations"
                    )

                    # Last attempt - return anyway with warning
                    if attempt == self.max_retries:
                        logger.warning(
                            f"Max retries ({self.max_retries}) reached. "
                            f"Returning sample with {len(report.violations)} violations."
                        )

                        metadata = {
                            **gen_meta,
                            'attempts': attempt + 1,
                            'validation_passed': False,
                            'validation_strictness': strictness,
                            'max_retries_reached': True
                        }

                        return sample, report, metadata

                    # Adjust guidance for retry
                    if attempt < self.max_retries:
                        stealth_level = min(0.99, stealth_level + 0.01)  # Slightly increase stealth
                        logger.debug(f"Retrying with adjusted stealth_level={stealth_level:.2f}")
            else:
                # No validation - return immediately
                metadata = {
                    **gen_meta,
                    'attempts': 1,
                    'validation_passed': None,  # Not validated
                }
                return sample, None, metadata

        # Should not reach here, but for safety
        logger.error("Unexpected: exited retry loop without return")
        return sample, report, {'error': 'unexpected_exit'}

    def generate_batch(
        self,
        n_samples: int,
        protocol: str,
        attack_pattern: str = 'slow_exfiltration',
        benign_samples: Optional[np.ndarray] = None,
        stealth_level: float = 0.95,
        strictness: Optional[Literal['strict', 'moderate', 'permissive']] = None,
        validate: bool = True
    ) -> Tuple[np.ndarray, List[ValidationReport], Dict]:
        """
        Generate a batch of constrained hard-negative attacks.

        Args:
            n_samples: Number of samples to generate
            protocol: Target protocol
            attack_pattern: Attack pattern to inject
            benign_samples: Optional batch of benign samples to mimic
            stealth_level: Stealth level (0-1)
            strictness: Validation strictness
            validate: Whether to validate

        Returns:
            Tuple of (generated_samples, validation_reports, batch_metadata)
        """
        samples = []
        reports = []

        for i in range(n_samples):
            benign_sample = benign_samples[i] if benign_samples is not None else None

            batch_samples, batch_reports, _ = self.generate_constrained(
                n_samples=1,
                protocol=protocol,
                attack_pattern=attack_pattern,
                benign_sample=benign_sample,
                stealth_level=stealth_level,
                strictness=strictness,
                validate=validate
            )

            samples.append(batch_samples[0])
            reports.append(batch_reports[0])

        # Aggregate statistics
        if validate and self.constraint_manager is not None:
            agg_stats = self.constraint_manager.aggregate_violations(reports)
        else:
            agg_stats = {}

        metadata = {
            'batch_size': n_samples,
            'protocol': protocol,
            'attack_pattern': attack_pattern,
            'stealth_level': stealth_level,
            'validation_enabled': validate,
            'aggregated_violations': agg_stats
        }

        return np.array(samples), reports, metadata

    def get_generation_statistics(self) -> Dict:
        """
        Get statistics about generation performance.

        Returns:
            Dictionary with generation statistics
        """
        stats = dict(self.generation_stats)

        if stats['retry_counts']:
            stats['average_retries'] = np.mean(stats['retry_counts'])
            stats['max_retries_used'] = max(stats['retry_counts'])
        else:
            stats['average_retries'] = 0
            stats['max_retries_used'] = 0

        if stats['total_attempts'] > 0:
            stats['success_rate'] = stats['successful_generations'] / stats['total_attempts']
        else:
            stats['success_rate'] = 0.0

        return stats

    def reset_statistics(self):
        """Reset generation statistics."""
        self.generation_stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'failed_validations': 0,
            'retry_counts': []
        }

    # Backward compatibility methods - delegate to underlying generator

    def generate(
        self,
        n_samples: int = 1,
        target_statistics: Optional[Dict] = None,
        guidance_scale: float = 1.0,
        n_inference_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate samples (backward compatible with IoTDiffusionGenerator).

        This method delegates to the underlying generator for backward compatibility.
        Use generate_constrained() for constraint-aware generation.
        """
        return self.generator.generate(
            n_samples=n_samples,
            target_statistics=target_statistics,
            guidance_scale=guidance_scale,
            n_inference_steps=n_inference_steps
        )

    def generate_hard_negative(
        self,
        benign_sample: np.ndarray,
        attack_pattern: str = 'slow_exfiltration',
        stealth_level: float = 0.95
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate hard-negative (backward compatible with IoTDiffusionGenerator).

        This method delegates to the underlying generator for backward compatibility.
        Use generate_constrained() for constraint-aware generation with validation.
        """
        return self.generator.generate_hard_negative(
            benign_sample=benign_sample,
            attack_pattern=attack_pattern,
            stealth_level=stealth_level
        )

    def get_decomposition(self, sample: np.ndarray) -> Dict[str, np.ndarray]:
        """Get trend/seasonality decomposition (delegates to underlying generator)."""
        return self.generator.get_decomposition(sample)

    def save_checkpoint(self, path: str):
        """Save generator checkpoint (delegates to underlying generator)."""
        self.generator.save_checkpoint(path)

    def load_checkpoint(self, path: str):
        """Load generator checkpoint (delegates to underlying generator)."""
        self.generator.load_checkpoint(path)
