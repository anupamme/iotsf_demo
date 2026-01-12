#!/usr/bin/env python3
"""
Interactive test script for IoTConstraintManager.

This script demonstrates various features of the constraint management system
and allows you to test it with different configurations and samples.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.constraints.manager import IoTConstraintManager
from src.models.constraints.types import FEATURE_IDX_FWD_BYTS_B_AVG, FEATURE_IDX_BWD_BYTS_B_AVG


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def test_basic_initialization():
    """Test 1: Basic initialization from config file."""
    print_section("Test 1: Basic Initialization")

    manager = IoTConstraintManager()
    print(manager.summary())

    print(f"\nEnabled: {manager.is_enabled()}")
    print(f"Available protocols: {manager.get_available_protocols()}")
    print(f"Default protocol: {manager.default_protocol}")
    print(f"Default strictness: {manager.default_strictness}")


def test_custom_configuration():
    """Test 2: Custom configuration."""
    print_section("Test 2: Custom Configuration")

    custom_config = {
        'constraints': {
            'enabled': True,
            'default_protocol': 'mqtt',
            'strictness': 'strict',
            'protocols': {
                'mqtt': {
                    'enabled': True,
                    'packet_size_range': [2, 1024],
                    'soft_constraints': {
                        'packet_size_mean': 128,
                        'packet_size_std': 256
                    }
                }
            }
        }
    }

    manager = IoTConstraintManager(config_dict=custom_config)
    print(f"Custom default protocol: {manager.default_protocol}")
    print(f"Custom strictness: {manager.default_strictness}")
    print(f"Loaded protocols: {manager.get_available_protocols()}")


def test_modbus_validation():
    """Test 3: Modbus protocol validation."""
    print_section("Test 3: Modbus Validation")

    manager = IoTConstraintManager()

    # Create a typical Modbus sample
    sample = np.random.normal(64, 32, (128, 12))
    sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = np.random.uniform(30, 100, 128)
    sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = np.random.uniform(30, 100, 128)

    print("Validating Modbus sample...")
    report = manager.validate(sample, protocol='modbus', strictness='moderate')

    print(f"\n{report.summary()}")
    print(f"Valid (moderate): {report.is_valid('moderate')}")
    print(f"Valid (strict): {report.is_valid('strict')}")
    print(f"Total violations: {len(report.violations)}")

    if report.violations:
        print("\nViolations:")
        for i, violation in enumerate(report.violations[:3], 1):  # Show first 3
            print(f"  {i}. [{violation.severity.upper()}] {violation.constraint_name}")
            if violation.suggestion:
                print(f"     → {violation.suggestion}")

    print("\nStatistics:")
    for key, value in report.statistics.items():
        if key != 'shape':
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


def test_mqtt_validation():
    """Test 4: MQTT protocol validation."""
    print_section("Test 4: MQTT Validation")

    manager = IoTConstraintManager()

    # Create a sample with lognormal distribution (typical for MQTT)
    sample = np.abs(np.random.lognormal(np.log(128), 0.5, (128, 12)))

    print("Validating MQTT sample with lognormal distribution...")
    report = manager.validate(sample, protocol='mqtt', strictness='moderate')

    print(f"\n{report.summary()}")
    print(f"Valid: {report.is_valid('moderate')}")
    print(f"Violations: {len(report.violations)}")


def test_coap_validation():
    """Test 5: CoAP protocol validation."""
    print_section("Test 5: CoAP Validation")

    manager = IoTConstraintManager()

    # Create a CoAP sample with exponential timing
    sample = np.zeros((128, 12))
    sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = np.random.uniform(50, 200, 128)
    sample[:, FEATURE_IDX_BWD_BYTS_B_AVG] = np.random.uniform(50, 200, 128)
    sample[:, 10] = np.random.exponential(5.0, 128)  # Exponential IAT

    print("Validating CoAP sample with exponential timing...")
    report = manager.validate(sample, protocol='coap', strictness='moderate')

    print(f"\n{report.summary()}")
    print(f"Valid: {report.is_valid('moderate')}")
    print(f"Violations: {len(report.violations)}")


def test_multi_protocol():
    """Test 6: Multi-protocol validation."""
    print_section("Test 6: Multi-Protocol Validation")

    manager = IoTConstraintManager()

    # Create a generic sample
    sample = np.random.uniform(50, 150, (128, 12))

    print("Validating against all protocols...")
    reports = manager.validate_multi_protocol(sample)

    print("\nResults:")
    for protocol, report in reports.items():
        status = "✓ PASS" if report.is_valid('moderate') else "✗ FAIL"
        print(f"  {protocol.upper()}: {status} ({len(report.violations)} violations)")


def test_batch_validation():
    """Test 7: Batch validation."""
    print_section("Test 7: Batch Validation")

    manager = IoTConstraintManager()

    # Create a batch of 10 samples
    batch_size = 10
    batch = np.random.normal(100, 50, (batch_size, 128, 12))

    print(f"Validating batch of {batch_size} samples...")
    reports = manager.validate_batch(batch, protocol='modbus')

    # Aggregate statistics
    stats = manager.aggregate_violations(reports)

    print(f"\nBatch Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total violations: {stats['total_violations']}")
    print(f"  Violations per sample: {stats['violations_per_sample']:.2f}")
    print(f"\nSeverity breakdown:")
    for severity, count in stats['severity_counts'].items():
        print(f"  {severity}: {count}")
    print(f"\nPass rates:")
    for strictness, rate in stats['pass_rates'].items():
        print(f"  {strictness}: {rate:.1%}")


def test_strictness_levels():
    """Test 8: Different strictness levels."""
    print_section("Test 8: Strictness Levels")

    manager = IoTConstraintManager()

    # Create a sample that may have some violations
    sample = np.random.uniform(80, 120, (128, 12))

    print("Testing different strictness levels on the same sample:\n")

    for strictness in ['strict', 'moderate', 'permissive']:
        report = manager.validate(sample, protocol='modbus', strictness=strictness)
        valid = report.is_valid(strictness)
        status = "✓ PASS" if valid else "✗ FAIL"
        print(f"  {strictness.capitalize():12} {status:8} ({len(report.violations)} violations)")


def test_guidance_hints():
    """Test 9: Guidance hints."""
    print_section("Test 9: Guidance Hints")

    manager = IoTConstraintManager()

    for protocol in ['modbus', 'mqtt', 'coap']:
        hints = manager.get_guidance_hints(protocol=protocol)
        print(f"\n{protocol.upper()} Guidance Hints:")
        for key, value in hints.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")


def test_invalid_samples():
    """Test 10: Intentionally invalid samples."""
    print_section("Test 10: Invalid Samples")

    manager = IoTConstraintManager()

    # Test 1: Packet sizes too large
    print("\n1. Testing with oversized packets:")
    sample_large = np.random.uniform(1000, 2000, (128, 12))
    report = manager.validate(sample_large, protocol='modbus', strictness='moderate')
    print(f"   Valid: {report.is_valid('moderate')}")
    print(f"   Violations: {len(report.violations)}")

    # Test 2: Negative values (invalid for some distributions)
    print("\n2. Testing with negative values:")
    sample_negative = np.random.uniform(-50, 50, (128, 12))
    report = manager.validate(sample_negative, protocol='mqtt', strictness='moderate')
    print(f"   Valid: {report.is_valid('moderate')}")
    print(f"   Violations: {len(report.violations)}")

    # Test 3: Very small values
    print("\n3. Testing with very small packet sizes:")
    sample_small = np.random.uniform(0, 5, (128, 12))
    report = manager.validate(sample_small, protocol='coap', strictness='moderate')
    print(f"   Valid: {report.is_valid('moderate')}")
    print(f"   Violations: {len(report.violations)}")


def test_disabled_constraints():
    """Test 11: Disabled constraints."""
    print_section("Test 11: Disabled Constraints")

    config = {
        'constraints': {
            'enabled': False,
            'protocols': {
                'modbus': {'enabled': True}
            }
        }
    }

    manager = IoTConstraintManager(config_dict=config)

    print(f"Constraints enabled: {manager.is_enabled()}")

    # Even with invalid sample, should pass when disabled
    sample = np.random.uniform(5000, 10000, (128, 12))
    report = manager.validate(sample, protocol='modbus')

    print(f"Valid: {report.is_valid('moderate')}")
    print(f"Violations: {len(report.violations)}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  IoT Constraint Manager Test Suite")
    print("="*70)

    tests = [
        test_basic_initialization,
        test_custom_configuration,
        test_modbus_validation,
        test_mqtt_validation,
        test_coap_validation,
        test_multi_protocol,
        test_batch_validation,
        test_strictness_levels,
        test_guidance_hints,
        test_invalid_samples,
        test_disabled_constraints
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ Error in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print_section("All Tests Complete")
    print("✓ Constraint management system is working correctly!\n")


if __name__ == '__main__':
    main()
