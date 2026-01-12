"""Tests for IoTConstraintManager."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.constraints.manager import IoTConstraintManager
from src.models.constraints.types import ValidationReport


class TestManagerInitialization:
    """Test manager initialization."""

    def test_manager_creation_with_config_file(self):
        """Test creating manager with config file."""
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        manager = IoTConstraintManager(config_path=config_path)

        assert manager is not None
        assert manager.is_enabled()
        assert manager.default_protocol in ['modbus', 'mqtt', 'coap']
        assert manager.default_strictness in ['strict', 'moderate', 'permissive']

    def test_manager_creation_with_config_dict(self):
        """Test creating manager with config dictionary."""
        config = {
            'constraints': {
                'enabled': True,
                'default_protocol': 'mqtt',
                'strictness': 'strict',
                'protocols': {
                    'mqtt': {
                        'enabled': True,
                        'packet_size_range': [2, 1024]
                    }
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)

        assert manager.default_protocol == 'mqtt'
        assert manager.default_strictness == 'strict'
        assert 'mqtt' in manager.get_available_protocols()

    def test_manager_with_protocol_override(self):
        """Test manager with protocol override."""
        config = {
            'constraints': {
                'enabled': True,
                'default_protocol': 'modbus',
                'protocols': {
                    'modbus': {'enabled': True},
                    'mqtt': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config, protocol='mqtt')

        assert manager.default_protocol == 'mqtt'

    def test_manager_with_strictness_override(self):
        """Test manager with strictness override."""
        config = {
            'constraints': {
                'enabled': True,
                'strictness': 'moderate',
                'protocols': {
                    'modbus': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config, strictness='strict')

        assert manager.default_strictness == 'strict'

    def test_manager_loads_validators(self):
        """Test that manager loads protocol validators."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'modbus': {'enabled': True},
                    'mqtt': {'enabled': True},
                    'coap': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)

        assert len(manager.validators) == 3
        assert 'modbus' in manager.validators
        assert 'mqtt' in manager.validators
        assert 'coap' in manager.validators

    def test_manager_disabled_protocol(self):
        """Test that disabled protocols are not loaded."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'modbus': {'enabled': True},
                    'mqtt': {'enabled': False}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)

        assert 'modbus' in manager.validators
        assert 'mqtt' not in manager.validators


class TestValidation:
    """Test validation functionality."""

    def test_validate_single_sample(self):
        """Test validating a single sample."""
        config = {
            'constraints': {
                'enabled': True,
                'default_protocol': 'modbus',
                'protocols': {
                    'modbus': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)

        # Create a valid sample
        sample = np.random.normal(50, 10, (128, 12))
        report = manager.validate(sample, protocol='modbus')

        assert isinstance(report, ValidationReport)
        assert report.protocol == 'modbus'
        assert hasattr(report, 'violations')
        assert hasattr(report, 'statistics')

    def test_validate_with_protocol_override(self):
        """Test validation with protocol override."""
        config = {
            'constraints': {
                'enabled': True,
                'default_protocol': 'modbus',
                'protocols': {
                    'modbus': {'enabled': True},
                    'mqtt': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        sample = np.random.normal(50, 10, (128, 12))

        # Validate with default protocol
        report1 = manager.validate(sample)
        assert report1.protocol == 'modbus'

        # Validate with override
        report2 = manager.validate(sample, protocol='mqtt')
        assert report2.protocol == 'mqtt'

    def test_validate_with_strictness_override(self):
        """Test validation with strictness override."""
        config = {
            'constraints': {
                'enabled': True,
                'strictness': 'permissive',
                'protocols': {
                    'modbus': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        sample = np.random.uniform(200, 300, (128, 12))  # May violate some constraints

        report = manager.validate(sample, strictness='strict')
        # Strict mode should be more restrictive than default permissive

    def test_validate_invalid_protocol(self):
        """Test validation with invalid protocol."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'modbus': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        sample = np.random.normal(50, 10, (128, 12))

        report = manager.validate(sample, protocol='unknown_protocol')

        assert report.protocol == 'unknown_protocol'
        assert len(report.violations) > 0
        assert any('validator_missing' in v.constraint_name for v in report.violations)

    def test_validate_disabled_constraints(self):
        """Test validation when constraints are disabled."""
        config = {
            'constraints': {
                'enabled': False,
                'protocols': {
                    'modbus': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        sample = np.random.normal(50, 10, (128, 12))

        report = manager.validate(sample, protocol='modbus')

        # Should return empty report when disabled
        assert len(report.violations) == 0


class TestBatchValidation:
    """Test batch validation."""

    def test_validate_batch(self):
        """Test validating a batch of samples."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'modbus': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)

        # Create batch of 5 samples
        batch = np.random.normal(50, 10, (5, 128, 12))
        reports = manager.validate_batch(batch, protocol='modbus')

        assert len(reports) == 5
        for i, report in enumerate(reports):
            assert isinstance(report, ValidationReport)
            assert report.sample_id == i
            assert report.protocol == 'modbus'


class TestMultiProtocolValidation:
    """Test multi-protocol validation."""

    def test_validate_multi_protocol(self):
        """Test validating against multiple protocols."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'modbus': {'enabled': True},
                    'mqtt': {'enabled': True},
                    'coap': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        sample = np.random.normal(100, 20, (128, 12))

        reports = manager.validate_multi_protocol(sample)

        assert len(reports) == 3
        assert 'modbus' in reports
        assert 'mqtt' in reports
        assert 'coap' in reports
        assert all(isinstance(r, ValidationReport) for r in reports.values())

    def test_validate_specific_protocols(self):
        """Test validating against specific protocols."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'modbus': {'enabled': True},
                    'mqtt': {'enabled': True},
                    'coap': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        sample = np.random.normal(100, 20, (128, 12))

        reports = manager.validate_multi_protocol(sample, protocols=['modbus', 'mqtt'])

        assert len(reports) == 2
        assert 'modbus' in reports
        assert 'mqtt' in reports
        assert 'coap' not in reports


class TestViolationAggregation:
    """Test violation aggregation."""

    def test_aggregate_violations_empty(self):
        """Test aggregating empty violation list."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'modbus': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        reports = []

        stats = manager.aggregate_violations(reports)

        assert stats['total_samples'] == 0
        assert stats['total_violations'] == 0

    def test_aggregate_violations_batch(self):
        """Test aggregating violations from batch."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'modbus': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)

        # Create batch with mix of valid and invalid samples
        batch = np.random.uniform(10, 200, (10, 128, 12))
        reports = manager.validate_batch(batch, protocol='modbus')

        stats = manager.aggregate_violations(reports)

        assert stats['total_samples'] == 10
        assert 'total_violations' in stats
        assert 'violations_per_sample' in stats
        assert 'severity_counts' in stats
        assert 'constraint_counts' in stats
        assert 'pass_rates' in stats
        assert 'strict' in stats['pass_rates']
        assert 'moderate' in stats['pass_rates']
        assert 'permissive' in stats['pass_rates']


class TestGuidanceHints:
    """Test guidance hints."""

    def test_get_guidance_hints(self):
        """Test getting guidance hints for protocol."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'modbus': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        hints = manager.get_guidance_hints(protocol='modbus')

        assert isinstance(hints, dict)
        assert 'protocol' in hints
        assert hints['protocol'] == 'modbus'

    def test_get_guidance_hints_with_target_stats(self):
        """Test guidance hints with target statistics."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'mqtt': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        target_stats = {'custom_param': 42}
        hints = manager.get_guidance_hints(protocol='mqtt', target_statistics=target_stats)

        assert 'custom_param' in hints
        assert hints['custom_param'] == 42

    def test_get_guidance_hints_default_protocol(self):
        """Test guidance hints using default protocol."""
        config = {
            'constraints': {
                'enabled': True,
                'default_protocol': 'coap',
                'protocols': {
                    'coap': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        hints = manager.get_guidance_hints()

        assert hints['protocol'] == 'coap'


class TestUtilityMethods:
    """Test utility methods."""

    def test_is_enabled(self):
        """Test checking if constraints are enabled."""
        config_enabled = {
            'constraints': {
                'enabled': True,
                'protocols': {}
            }
        }
        manager = IoTConstraintManager(config_dict=config_enabled)
        assert manager.is_enabled() is True

        config_disabled = {
            'constraints': {
                'enabled': False,
                'protocols': {}
            }
        }
        manager = IoTConstraintManager(config_dict=config_disabled)
        assert manager.is_enabled() is False

    def test_get_available_protocols(self):
        """Test getting available protocols."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'modbus': {'enabled': True},
                    'mqtt': {'enabled': False},
                    'coap': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        protocols = manager.get_available_protocols()

        assert 'modbus' in protocols
        assert 'coap' in protocols
        assert 'mqtt' not in protocols

    def test_get_validator(self):
        """Test getting specific validator."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'modbus': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        validator = manager.get_validator('modbus')

        assert validator is not None
        assert validator.get_protocol_name() == 'modbus'

    def test_get_validator_invalid(self):
        """Test getting validator for invalid protocol."""
        config = {
            'constraints': {
                'enabled': True,
                'protocols': {
                    'modbus': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        validator = manager.get_validator('invalid')

        assert validator is None

    def test_summary(self):
        """Test generating summary."""
        config = {
            'constraints': {
                'enabled': True,
                'default_protocol': 'modbus',
                'strictness': 'moderate',
                'protocols': {
                    'modbus': {'enabled': True},
                    'mqtt': {'enabled': True}
                }
            }
        }

        manager = IoTConstraintManager(config_dict=config)
        summary = manager.summary()

        assert 'IoTConstraintManager' in summary
        assert 'modbus' in summary
        assert 'mqtt' in summary
        assert 'moderate' in summary
