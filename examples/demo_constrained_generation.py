"""
Demo: Generating Protocol-Valid IoT Attacks with Constraint Validation

This example demonstrates how to use the HardNegativeGenerator with the
constraint system enabled to generate protocol-valid, stealthy attacks.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hard_negative_generator import HardNegativeGenerator
from src.models.constraints.manager import IoTConstraintManager


def main():
    print("=" * 70)
    print("Protocol-Valid Attack Generation Demo")
    print("=" * 70)

    # Initialize constraint manager (loads from config.yaml)
    print("\n1. Initializing constraint manager...")
    constraint_manager = IoTConstraintManager()

    print(f"   ✓ Constraints enabled: {constraint_manager.is_enabled()}")
    print(f"   ✓ Protocols available: {list(constraint_manager.validators.keys())}")
    print(f"   ✓ Default strictness: {constraint_manager.default_strictness}")

    # Initialize generator with constraint validation
    print("\n2. Initializing hard-negative generator...")
    generator = HardNegativeGenerator(
        constraint_manager=constraint_manager,
        max_retries=3,
        strictness='moderate'
    )
    generator.initialize()
    print("   ✓ Generator initialized")

    # Example 1: Generate Modbus attacks
    print("\n" + "=" * 70)
    print("Example 1: Modbus TCP (Industrial SCADA Protocol)")
    print("=" * 70)

    samples, reports, metadata = generator.generate_constrained(
        n_samples=3,
        protocol='modbus',
        attack_pattern='slow_exfiltration',
        stealth_level=0.95,
        strictness='moderate'
    )

    print(f"\n✓ Generated {len(samples)} Modbus attack samples")
    print(f"  Protocol: {metadata['protocol']}")
    print(f"  Attack pattern: {metadata['attack_pattern']}")
    print(f"  Validation enabled: {metadata['validation_enabled']}")

    print("\nValidation Results:")
    for i, report in enumerate(reports):
        print(f"\n  Sample {i+1}:")
        print(f"    Valid (moderate): {report.is_valid('moderate')}")
        print(f"    Violations: {len(report.violations)}")
        if report.violations:
            for v in report.violations[:2]:  # Show first 2
                print(f"      - {v.severity.upper()}: {v.constraint_name}")

    # Example 2: Generate MQTT attacks
    print("\n" + "=" * 70)
    print("Example 2: MQTT (IoT Pub-Sub Protocol)")
    print("=" * 70)

    samples, reports, metadata = generator.generate_constrained(
        n_samples=3,
        protocol='mqtt',
        attack_pattern='beacon',
        stealth_level=0.98,
        strictness='moderate'
    )

    print(f"\n✓ Generated {len(samples)} MQTT attack samples")
    print(f"  Protocol: {metadata['protocol']}")
    print(f"  Attack pattern: {metadata['attack_pattern']}")

    print("\nValidation Results:")
    for i, report in enumerate(reports):
        print(f"  Sample {i+1}: {report.summary()}")

    # Example 3: Batch generation with validation
    print("\n" + "=" * 70)
    print("Example 3: CoAP Batch Generation with Aggregated Statistics")
    print("=" * 70)

    samples, reports, metadata = generator.generate_batch(
        n_samples=5,
        protocol='coap',
        attack_pattern='lotl_mimicry',
        stealth_level=0.96,
        strictness='moderate',
        validate=True
    )

    print(f"\n✓ Generated {metadata['batch_size']} CoAP attack samples")

    if 'aggregated_violations' in metadata:
        agg = metadata['aggregated_violations']
        print("\nAggregated Validation Statistics:")
        print(f"  Total samples: {agg['total_samples']}")
        print(f"  Pass rate (strict): {agg['pass_rates']['strict']:.1%}")
        print(f"  Pass rate (moderate): {agg['pass_rates']['moderate']:.1%}")
        print(f"  Pass rate (permissive): {agg['pass_rates']['permissive']:.1%}")
        print(f"  Total violations: {agg['total_violations']}")

    # Example 4: Multi-protocol validation
    print("\n" + "=" * 70)
    print("Example 4: Multi-Protocol Validation")
    print("=" * 70)

    # Generate a sample
    sample = generator.generate(n_samples=1)[0]

    # Validate against all protocols
    print("\nValidating same sample against all protocols:")
    multi_reports = constraint_manager.validate_multi_protocol(sample)

    for protocol, report in multi_reports.items():
        print(f"\n  {protocol.upper()}:")
        print(f"    Valid (moderate): {report.is_valid('moderate')}")
        print(f"    Violations: {len(report.violations)}")

    # Generation statistics
    print("\n" + "=" * 70)
    print("Generation Statistics")
    print("=" * 70)

    stats = generator.get_generation_statistics()
    print(f"\nTotal attempts: {stats['total_attempts']}")
    print(f"Successful: {stats['successful_generations']}")
    print(f"Failed validations: {stats['failed_validations']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Average retries: {stats['average_retries']:.2f}")

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Constraint system validates protocol-specific rules")
    print("  • Automatic retry logic ensures high success rate")
    print("  • Supports multiple IoT protocols (Modbus, MQTT, CoAP)")
    print("  • Configurable strictness levels for different use cases")
    print("  • Generated attacks are both stealthy AND protocol-valid")
    print()


if __name__ == "__main__":
    main()
