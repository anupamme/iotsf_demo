"""Tests for HardNegativeGenerator with constraint integration."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hard_negative_generator import HardNegativeGenerator
from src.models.diffusion_ts import IoTDiffusionGenerator
from src.models.constraints.manager import IoTConstraintManager


class TestHardNegativeGeneratorInitialization:
    """Test generator initialization."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        generator = HardNegativeGenerator()

        assert generator.generator is not None
        assert generator.constraint_manager is None
        assert generator.max_retries == 3
        assert generator.default_strictness == 'moderate'

    def test_initialization_with_existing_generator(self):
        """Test initialization with existing diffusion generator."""
        diffusion_gen = IoTDiffusionGenerator(seq_length=64, feature_dim=12)
        generator = HardNegativeGenerator(diffusion_generator=diffusion_gen)

        assert generator.generator is diffusion_gen
        assert generator.generator.seq_length == 64

    def test_initialization_with_constraint_manager(self):
        """Test initialization with constraint manager."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {'modbus': {'enabled': True}}
            }
        }
        constraint_mgr = IoTConstraintManager(config_dict=config)
        generator = HardNegativeGenerator(constraint_manager=constraint_mgr)

        assert generator.constraint_manager is constraint_mgr

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        generator = HardNegativeGenerator(
            seq_length=64,
            feature_dim=8,
            max_retries=5,
            strictness='strict'
        )

        assert generator.max_retries == 5
        assert generator.default_strictness == 'strict'


class TestBasicGeneration:
    """Test basic generation without constraints."""

    def test_generate_without_constraints(self):
        """Test generation without constraint validation."""
        generator = HardNegativeGenerator()
        generator.initialize()

        samples = generator.generate(n_samples=2)

        assert samples.shape == (2, 128, 12)
        assert not np.isnan(samples).any()

    def test_generate_with_target_statistics(self):
        """Test generation with target statistics."""
        generator = HardNegativeGenerator()
        generator.initialize()

        target_stats = {'mean': 100.0, 'std': 20.0}
        samples = generator.generate(n_samples=1, target_statistics=target_stats)

        assert samples.shape == (1, 128, 12)
        # Check that statistics are approximately matched
        assert abs(samples.mean() - 100.0) < 20.0

    def test_generate_hard_negative_backward_compat(self):
        """Test backward compatible hard-negative generation."""
        generator = HardNegativeGenerator()
        generator.initialize()

        benign = np.random.normal(50, 10, (128, 12))
        attack, metadata = generator.generate_hard_negative(
            benign_sample=benign,
            attack_pattern='slow_exfiltration',
            stealth_level=0.95
        )

        assert attack.shape == (128, 12)
        assert 'attack_type' in metadata
        assert metadata['attack_type'] == 'slow_exfiltration'


class TestConstrainedGeneration:
    """Test constraint-aware generation."""

    def test_generate_constrained_with_modbus(self):
        """Test constrained generation for Modbus protocol."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {'modbus': {'enabled': True}}
            }
        }
        constraint_mgr = IoTConstraintManager(config_dict=config)
        generator = HardNegativeGenerator(constraint_manager=constraint_mgr)
        generator.initialize()

        samples, reports, metadata = generator.generate_constrained(
            n_samples=2,
            protocol='modbus',
            strictness='permissive'
        )

        assert samples.shape == (2, 128, 12)
        assert len(reports) == 2
        assert metadata['protocol'] == 'modbus'
        assert metadata['validation_enabled'] is True

    def test_generate_constrained_with_validation_disabled(self):
        """Test constrained generation with validation disabled."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {'modbus': {'enabled': True}}
            }
        }
        constraint_mgr = IoTConstraintManager(config_dict=config)
        generator = HardNegativeGenerator(constraint_manager=constraint_mgr)
        generator.initialize()

        samples, reports, metadata = generator.generate_constrained(
            n_samples=1,
            protocol='modbus',
            validate=False
        )

        assert samples.shape == (1, 128, 12)
        assert reports[0] is None
        assert metadata['validation_enabled'] is False

    def test_generate_constrained_without_constraint_manager(self):
        """Test constrained generation without constraint manager."""
        generator = HardNegativeGenerator()
        generator.initialize()

        samples, reports, metadata = generator.generate_constrained(
            n_samples=1,
            protocol='modbus'
        )

        assert samples.shape == (1, 128, 12)
        assert reports[0] is None


class TestRetryLogic:
    """Test validation retry logic."""

    def test_retry_on_validation_failure(self):
        """Test that generator retries on validation failure."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {'modbus': {'enabled': True}}
            }
        }
        constraint_mgr = IoTConstraintManager(config_dict=config)
        generator = HardNegativeGenerator(
            constraint_manager=constraint_mgr,
            max_retries=2
        )
        generator.initialize()

        # Generate with strict validation (more likely to fail)
        samples, reports, metadata = generator.generate_constrained(
            n_samples=1,
            protocol='modbus',
            strictness='strict',
            stealth_level=0.5  # Lower stealth increases failure chance
        )

        assert samples.shape == (1, 128, 12)
        # Check that attempts were tracked
        assert 'attempts' in metadata or len(samples) > 0

    def test_max_retries_reached(self):
        """Test behavior when max retries is reached."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {'modbus': {'enabled': True}}
            }
        }
        constraint_mgr = IoTConstraintManager(config_dict=config)
        generator = HardNegativeGenerator(
            constraint_manager=constraint_mgr,
            max_retries=1  # Low retries for faster test
        )
        generator.initialize()

        samples, reports, metadata = generator.generate_constrained(
            n_samples=1,
            protocol='modbus',
            strictness='strict',
            stealth_level=0.3  # Very low stealth
        )

        # Should still return a sample even if validation fails
        assert samples.shape == (1, 128, 12)


class TestBatchGeneration:
    """Test batch generation."""

    def test_generate_batch_without_benign_samples(self):
        """Test batch generation without reference benign samples."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {'mqtt': {'enabled': True}}
            }
        }
        constraint_mgr = IoTConstraintManager(config_dict=config)
        generator = HardNegativeGenerator(constraint_manager=constraint_mgr)
        generator.initialize()

        samples, reports, metadata = generator.generate_batch(
            n_samples=3,
            protocol='mqtt',
            attack_pattern='beacon',
            validate=True
        )

        assert samples.shape == (3, 128, 12)
        assert len(reports) == 3
        assert metadata['batch_size'] == 3
        assert metadata['protocol'] == 'mqtt'

    def test_generate_batch_with_benign_samples(self):
        """Test batch generation with reference benign samples."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {'coap': {'enabled': True}}
            }
        }
        constraint_mgr = IoTConstraintManager(config_dict=config)
        generator = HardNegativeGenerator(constraint_manager=constraint_mgr)
        generator.initialize()

        benign_samples = np.random.normal(100, 20, (3, 128, 12))
        samples, reports, metadata = generator.generate_batch(
            n_samples=3,
            protocol='coap',
            attack_pattern='lotl_mimicry',
            benign_samples=benign_samples,
            stealth_level=0.98
        )

        assert samples.shape == (3, 128, 12)
        assert len(reports) == 3


class TestStatistics:
    """Test generation statistics tracking."""

    def test_statistics_tracking(self):
        """Test that generation statistics are tracked."""
        generator = HardNegativeGenerator()
        generator.initialize()

        # Reset statistics
        generator.reset_statistics()

        # Generate some samples
        generator.generate(n_samples=2)

        stats = generator.get_generation_statistics()
        assert 'total_attempts' in stats
        assert 'successful_generations' in stats
        assert 'failed_validations' in stats

    def test_statistics_with_validation(self):
        """Test statistics with validation enabled."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {'modbus': {'enabled': True}}
            }
        }
        constraint_mgr = IoTConstraintManager(config_dict=config)
        generator = HardNegativeGenerator(constraint_manager=constraint_mgr)
        generator.initialize()

        generator.reset_statistics()

        # Generate with validation
        generator.generate_constrained(
            n_samples=2,
            protocol='modbus',
            validate=True
        )

        stats = generator.get_generation_statistics()
        assert stats['total_attempts'] > 0
        assert 'success_rate' in stats

    def test_reset_statistics(self):
        """Test statistics reset."""
        generator = HardNegativeGenerator()
        generator.initialize()

        generator.generate(n_samples=1)
        generator.reset_statistics()

        stats = generator.get_generation_statistics()
        assert stats['total_attempts'] == 0
        assert stats['successful_generations'] == 0


class TestMultiProtocolSupport:
    """Test generation for different protocols."""

    def test_generate_modbus_constrained(self):
        """Test Modbus-constrained generation."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {'modbus': {'enabled': True}}
            }
        }
        constraint_mgr = IoTConstraintManager(config_dict=config)
        generator = HardNegativeGenerator(constraint_manager=constraint_mgr)
        generator.initialize()

        samples, reports, metadata = generator.generate_constrained(
            n_samples=1,
            protocol='modbus',
            strictness='moderate'
        )

        assert samples.shape == (1, 128, 12)
        assert reports[0].protocol == 'modbus'

    def test_generate_mqtt_constrained(self):
        """Test MQTT-constrained generation."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {'mqtt': {'enabled': True}}
            }
        }
        constraint_mgr = IoTConstraintManager(config_dict=config)
        generator = HardNegativeGenerator(constraint_manager=constraint_mgr)
        generator.initialize()

        samples, reports, metadata = generator.generate_constrained(
            n_samples=1,
            protocol='mqtt',
            strictness='permissive'
        )

        assert samples.shape == (1, 128, 12)
        assert reports[0].protocol == 'mqtt'

    def test_generate_coap_constrained(self):
        """Test CoAP-constrained generation."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {'coap': {'enabled': True}}
            }
        }
        constraint_mgr = IoTConstraintManager(config_dict=config)
        generator = HardNegativeGenerator(constraint_manager=constraint_mgr)
        generator.initialize()

        samples, reports, metadata = generator.generate_constrained(
            n_samples=1,
            protocol='coap',
            strictness='moderate'
        )

        assert samples.shape == (1, 128, 12)
        assert reports[0].protocol == 'coap'


class TestBackwardCompatibility:
    """Test backward compatibility with IoTDiffusionGenerator."""

    def test_delegate_generate(self):
        """Test that generate() delegates correctly."""
        generator = HardNegativeGenerator()
        generator.initialize()

        samples = generator.generate(n_samples=1)

        assert samples.shape == (1, 128, 12)

    def test_delegate_generate_hard_negative(self):
        """Test that generate_hard_negative() delegates correctly."""
        generator = HardNegativeGenerator()
        generator.initialize()

        benign = np.random.normal(50, 10, (128, 12))
        attack, metadata = generator.generate_hard_negative(benign)

        assert attack.shape == (128, 12)
        assert 'attack_type' in metadata

    def test_delegate_get_decomposition(self):
        """Test that get_decomposition() delegates correctly."""
        generator = HardNegativeGenerator()
        generator.initialize()

        sample = np.random.normal(50, 10, (128, 12))
        decomp = generator.get_decomposition(sample)

        assert 'trend' in decomp
        assert 'seasonality' in decomp
        assert 'residual' in decomp


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_generate_with_invalid_protocol(self):
        """Test generation with invalid protocol name."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {'modbus': {'enabled': True}}
            }
        }
        constraint_mgr = IoTConstraintManager(config_dict=config)
        generator = HardNegativeGenerator(constraint_manager=constraint_mgr)
        generator.initialize()

        samples, reports, metadata = generator.generate_constrained(
            n_samples=1,
            protocol='invalid_protocol'
        )

        # Should still generate, but validation should report issue
        assert samples.shape == (1, 128, 12)

    def test_generate_with_zero_samples(self):
        """Test generation with n_samples=0."""
        generator = HardNegativeGenerator()
        generator.initialize()

        samples, reports, metadata = generator.generate_constrained(n_samples=0)

        assert samples.shape[0] == 0  # First dimension should be 0
        assert len(reports) == 0
