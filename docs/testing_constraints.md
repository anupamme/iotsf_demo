# Testing the Constraint Management System

This guide provides comprehensive instructions for testing the IoT constraint management system implemented in Phase 3.

## Quick Start

### 1. Run Automated Tests

The fastest way to verify everything works:

```bash
# Activate virtual environment
source .venv12/bin/activate

# Run all constraint tests
pytest tests/test_constraint_manager.py -v

# Run all constraint-related tests (including protocol validators)
pytest tests/test_protocols/ tests/test_constraint_manager.py -v

# Run with coverage report
pytest tests/test_constraint_manager.py --cov=src/models/constraints/manager --cov-report=term-missing
```

### 2. Interactive Testing Script

Run the comprehensive interactive test suite:

```bash
python scripts/test_constraint_manager.py
```

This script runs 11 different tests covering:
- Basic initialization
- Custom configurations
- Modbus/MQTT/CoAP validation
- Multi-protocol validation
- Batch validation
- Strictness levels
- Guidance hints
- Invalid samples
- Disabled constraints

### 3. Python REPL Testing

Quick manual testing in Python:

```python
import numpy as np
from src.models.constraints.manager import IoTConstraintManager

# Initialize manager
manager = IoTConstraintManager()
print(manager.summary())

# Validate a sample
sample = np.random.normal(100, 50, (128, 12))
report = manager.validate(sample, protocol='modbus', strictness='moderate')

print(report.summary())
print(f"Valid: {report.is_valid('moderate')}")
print(f"Violations: {len(report.violations)}")
```

## Detailed Testing Scenarios

### Test 1: Basic Initialization

Test that the manager loads correctly from the config file:

```python
from src.models.constraints.manager import IoTConstraintManager

# Load from default config.yaml
manager = IoTConstraintManager()

# Check configuration
assert manager.is_enabled() == True
assert manager.default_protocol == 'modbus'
assert 'modbus' in manager.get_available_protocols()
assert 'mqtt' in manager.get_available_protocols()
assert 'coap' in manager.get_available_protocols()

print(manager.summary())
```

### Test 2: Custom Configuration

Test with a custom configuration dictionary:

```python
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
```

### Test 3: Modbus Validation

Validate a Modbus-compliant sample:

```python
import numpy as np
from src.models.constraints.manager import IoTConstraintManager
from src.models.constraints.types import FEATURE_IDX_FWD_BYTS_B_AVG

manager = IoTConstraintManager()

# Create a Modbus-typical sample
sample = np.random.normal(64, 32, (128, 12))
sample[:, FEATURE_IDX_FWD_BYTS_B_AVG] = np.random.uniform(30, 100, 128)

report = manager.validate(sample, protocol='modbus', strictness='moderate')

print(f"Protocol: {report.protocol}")
print(f"Valid: {report.is_valid('moderate')}")
print(f"Violations: {len(report.violations)}")

# Print detailed violations
if report.violations:
    print("\nViolations:")
    for v in report.violations:
        print(f"  - [{v.severity}] {v.constraint_name}")
        print(f"    {v.suggestion}")
```

### Test 4: MQTT Validation

Validate an MQTT sample with lognormal distribution:

```python
import numpy as np
from src.models.constraints.manager import IoTConstraintManager

manager = IoTConstraintManager()

# MQTT typically has lognormal packet size distribution
sample = np.abs(np.random.lognormal(np.log(128), 0.5, (128, 12)))

report = manager.validate(sample, protocol='mqtt', strictness='moderate')
print(report.summary())
```

### Test 5: Multi-Protocol Validation

Validate the same sample against multiple protocols:

```python
import numpy as np
from src.models.constraints.manager import IoTConstraintManager

manager = IoTConstraintManager()

sample = np.random.uniform(50, 150, (128, 12))

# Validate against all protocols
reports = manager.validate_multi_protocol(sample)

for protocol, report in reports.items():
    status = "PASS" if report.is_valid('moderate') else "FAIL"
    print(f"{protocol}: {status} ({len(report.violations)} violations)")
```

### Test 6: Batch Validation

Validate multiple samples and aggregate statistics:

```python
import numpy as np
from src.models.constraints.manager import IoTConstraintManager

manager = IoTConstraintManager()

# Create batch of 20 samples
batch = np.random.normal(100, 50, (20, 128, 12))

# Validate batch
reports = manager.validate_batch(batch, protocol='modbus')

# Aggregate violations
stats = manager.aggregate_violations(reports)

print(f"Total samples: {stats['total_samples']}")
print(f"Total violations: {stats['total_violations']}")
print(f"Pass rate (moderate): {stats['pass_rates']['moderate']:.1%}")
print(f"\nSeverity breakdown:")
for severity, count in stats['severity_counts'].items():
    print(f"  {severity}: {count}")
```

### Test 7: Strictness Levels

Test how different strictness levels affect validation:

```python
import numpy as np
from src.models.constraints.manager import IoTConstraintManager

manager = IoTConstraintManager()

sample = np.random.uniform(80, 120, (128, 12))

for strictness in ['strict', 'moderate', 'permissive']:
    report = manager.validate(sample, protocol='modbus', strictness=strictness)
    valid = report.is_valid(strictness)
    print(f"{strictness}: {'PASS' if valid else 'FAIL'} ({len(report.violations)} violations)")
```

### Test 8: Guidance Hints

Get protocol-specific guidance for generation:

```python
from src.models.constraints.manager import IoTConstraintManager

manager = IoTConstraintManager()

for protocol in ['modbus', 'mqtt', 'coap']:
    hints = manager.get_guidance_hints(protocol=protocol)
    print(f"\n{protocol.upper()} Guidance:")
    print(f"  Target mean: {hints.get('target_mean')}")
    print(f"  Target std: {hints.get('target_std')}")
    print(f"  Port: {hints.get('port')}")
    print(f"  Transport: {hints.get('transport')}")
```

### Test 9: Invalid Samples

Test with intentionally invalid samples to verify detection:

```python
import numpy as np
from src.models.constraints.manager import IoTConstraintManager

manager = IoTConstraintManager()

# Test 1: Oversized packets (>260 bytes for Modbus)
sample_large = np.random.uniform(1000, 2000, (128, 12))
report = manager.validate(sample_large, protocol='modbus', strictness='moderate')
print(f"Oversized packets: {len(report.violations)} violations")

# Test 2: Undersized packets (<4 bytes for CoAP)
sample_small = np.random.uniform(0, 3, (128, 12))
report = manager.validate(sample_small, protocol='coap', strictness='moderate')
print(f"Undersized packets: {len(report.violations)} violations")

# Test 3: Negative values (invalid for lognormal)
sample_negative = np.random.uniform(-50, 50, (128, 12))
report = manager.validate(sample_negative, protocol='mqtt', strictness='moderate')
print(f"Negative values: {len(report.violations)} violations")
```

### Test 10: Disabled Constraints

Test behavior when constraints are disabled:

```python
from src.models.constraints.manager import IoTConstraintManager

config = {
    'constraints': {
        'enabled': False,
        'protocols': {'modbus': {'enabled': True}}
    }
}

manager = IoTConstraintManager(config_dict=config)

# Even invalid sample should pass
sample = np.random.uniform(5000, 10000, (128, 12))
report = manager.validate(sample, protocol='modbus')

print(f"Constraints enabled: {manager.is_enabled()}")
print(f"Valid: {report.is_valid('moderate')}")  # Should be True
print(f"Violations: {len(report.violations)}")  # Should be 0
```

## Expected Test Results

### Automated Test Suite

Running `pytest tests/test_constraint_manager.py -v` should show:

```
TestManagerInitialization::test_manager_creation_with_config_file PASSED
TestManagerInitialization::test_manager_creation_with_config_dict PASSED
TestManagerInitialization::test_manager_with_protocol_override PASSED
TestManagerInitialization::test_manager_with_strictness_override PASSED
TestManagerInitialization::test_manager_loads_validators PASSED
TestManagerInitialization::test_manager_disabled_protocol PASSED
TestValidation::test_validate_single_sample PASSED
TestValidation::test_validate_with_protocol_override PASSED
TestValidation::test_validate_with_strictness_override PASSED
TestValidation::test_validate_invalid_protocol PASSED
TestValidation::test_validate_disabled_constraints PASSED
TestBatchValidation::test_validate_batch PASSED
TestMultiProtocolValidation::test_validate_multi_protocol PASSED
TestMultiProtocolValidation::test_validate_specific_protocols PASSED
TestViolationAggregation::test_aggregate_violations_empty PASSED
TestViolationAggregation::test_aggregate_violations_batch PASSED
TestGuidanceHints::test_get_guidance_hints PASSED
TestGuidanceHints::test_get_guidance_hints_with_target_stats PASSED
TestGuidanceHints::test_get_guidance_hints_default_protocol PASSED
TestUtilityMethods::test_is_enabled PASSED
TestUtilityMethods::test_get_available_protocols PASSED
TestUtilityMethods::test_get_validator PASSED
TestUtilityMethods::test_get_validator_invalid PASSED
TestUtilityMethods::test_summary PASSED

======================== 24 passed in 0.71s ========================
```

### Full Test Suite

Running `pytest tests/ -v` should show **145 tests passing**:
- 54 original tests (diffusion, loader, preprocessor, device)
- 67 protocol validator tests (base, modbus, mqtt, coap)
- 24 constraint manager tests

## Troubleshooting

### Issue: Config file not found

**Error:** `Config file not found: /path/to/config.yaml`

**Solution:** Either:
1. Ensure you're running from the project root directory
2. Specify the config path explicitly:
   ```python
   manager = IoTConstraintManager(config_path='config/config.yaml')
   ```
3. Use a config dictionary instead of a file

### Issue: Import errors

**Error:** `ModuleNotFoundError: No module named 'src.models.constraints'`

**Solution:** Make sure you're in the project root and the virtual environment is activated:
```bash
cd /path/to/iotsf_demo
source .venv12/bin/activate
```

### Issue: Validator not found

**Error:** `No validator found for protocol: xyz`

**Solution:** Check that:
1. The protocol is spelled correctly (case-sensitive)
2. The protocol is enabled in config.yaml
3. The protocol validator is implemented (currently: modbus, mqtt, coap)

### Issue: YAML parsing error

**Error:** `yaml.scanner.ScannerError`

**Solution:** Check config.yaml syntax:
- Proper indentation (2 or 4 spaces, no tabs)
- Valid YAML formatting
- Arrays use `[item1, item2]` format

## Verification Checklist

After testing, verify these items:

- [ ] All 24 constraint manager tests pass
- [ ] All 145 total tests pass (backward compatibility)
- [ ] Manager loads from config.yaml successfully
- [ ] All three protocols (Modbus, MQTT, CoAP) validate samples
- [ ] Batch validation works with multiple samples
- [ ] Multi-protocol validation returns reports for all protocols
- [ ] Strictness levels (strict/moderate/permissive) work correctly
- [ ] Violation aggregation provides accurate statistics
- [ ] Guidance hints return protocol-specific parameters
- [ ] Invalid samples are detected and reported
- [ ] Disabled constraints are respected

## Performance Benchmarks

Expected validation performance:

- **Single sample validation:** < 10ms
- **Batch of 100 samples:** < 100ms
- **Multi-protocol validation:** < 30ms

To benchmark:

```python
import time
import numpy as np
from src.models.constraints.manager import IoTConstraintManager

manager = IoTConstraintManager()
sample = np.random.normal(100, 50, (128, 12))

# Single validation
start = time.time()
report = manager.validate(sample, protocol='modbus')
print(f"Single validation: {(time.time() - start) * 1000:.2f}ms")

# Batch validation
batch = np.random.normal(100, 50, (100, 128, 12))
start = time.time()
reports = manager.validate_batch(batch, protocol='modbus')
print(f"Batch (100 samples): {(time.time() - start) * 1000:.2f}ms")
```

## Next Steps

Once testing is complete, you can:

1. **Integrate with generation:** Use the manager to validate generated samples
2. **Add custom protocols:** Extend the validator registry
3. **Tune constraints:** Adjust thresholds in config.yaml
4. **Monitor production:** Use violation statistics for quality metrics

For more information, see:
- `src/models/constraints/manager.py` - Implementation
- `tests/test_constraint_manager.py` - Test suite
- `config/config.yaml` - Configuration reference
- `scripts/test_constraint_manager.py` - Interactive testing
