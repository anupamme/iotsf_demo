"""
Modbus TCP Protocol Validator

Implements protocol-specific constraints for Modbus TCP (industrial SCADA protocol).
Modbus TCP runs over TCP/IP, typically on port 502, with a request-response model.

Protocol Structure:
- MBAP Header: 7 bytes (Transaction ID, Protocol ID, Length, Unit ID)
- PDU: 1-253 bytes (Function Code + Data)
- Total: 7-260 bytes per packet

Reference: Modbus Application Protocol Specification V1.1b3
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger

from .base import ProtocolValidator
from ..types import (
    HardConstraint,
    SoftConstraint,
    MODBUS_FEATURE_MAP,
    FEATURE_IDX_FWD_BYTS_B_AVG,
    FEATURE_IDX_BWD_BYTS_B_AVG,
    FEATURE_IDX_FWD_IAT_MEAN,
    FEATURE_IDX_BWD_IAT_MEAN,
    FEATURE_IDX_FWD_PKTS_PER_SEC,
    FEATURE_IDX_BWD_PKTS_PER_SEC
)


# Modbus TCP Protocol Constants
MODBUS_PORT = 502
MODBUS_MBAP_HEADER_SIZE = 7  # bytes
MODBUS_PDU_MAX_SIZE = 253  # bytes
MODBUS_MIN_PACKET_SIZE = MODBUS_MBAP_HEADER_SIZE + 1  # 8 bytes (header + min PDU)
MODBUS_MAX_PACKET_SIZE = MODBUS_MBAP_HEADER_SIZE + MODBUS_PDU_MAX_SIZE  # 260 bytes

# Valid Modbus function codes (commonly used subset)
MODBUS_VALID_FUNCTION_CODES = [
    0x01,  # Read Coils
    0x02,  # Read Discrete Inputs
    0x03,  # Read Holding Registers
    0x04,  # Read Input Registers
    0x05,  # Write Single Coil
    0x06,  # Write Single Register
    0x0F,  # Write Multiple Coils
    0x10,  # Write Multiple Registers
    0x14,  # Read File Record
    0x15,  # Write File Record
    0x16,  # Mask Write Register
    0x17,  # Read/Write Multiple Registers
    0x18,  # Read FIFO Queue
    0x2B,  # Encapsulated Interface Transport
]

# Timing constraints (milliseconds)
MODBUS_MIN_RESPONSE_TIME_MS = 1
MODBUS_MAX_RESPONSE_TIME_MS = 100
MODBUS_TIMEOUT_MS = 1000
MODBUS_MIN_INTER_REQUEST_MS = 10
MODBUS_MAX_INTER_REQUEST_MS = 500

# Statistical properties for benign Modbus traffic
MODBUS_TYPICAL_PACKET_SIZE_MEAN = 64  # bytes
MODBUS_TYPICAL_PACKET_SIZE_STD = 32  # bytes
MODBUS_TYPICAL_REQUEST_RATE = 10  # requests per second (lambda = 0.1)
MODBUS_REQUEST_RESPONSE_RATIO = 1.0  # 1:1 ratio


class ModbusValidator(ProtocolValidator):
    """
    Validator for Modbus TCP protocol constraints.

    Validates both hard constraints (protocol rules) and soft constraints
    (statistical properties) for Modbus TCP traffic flows.
    """

    def _initialize_constraints(self):
        """Initialize Modbus-specific constraints."""
        # Get configuration or use defaults
        config = self.config.get('modbus', {})

        # Hard Constraints
        self._init_packet_size_constraints(config)
        self._init_timing_constraints(config)

        # Soft Constraints
        self._init_statistical_constraints(config)

    def _init_packet_size_constraints(self, config: Dict):
        """Initialize packet size hard constraints."""
        min_size = config.get('packet_size_range', [MODBUS_MIN_PACKET_SIZE, MODBUS_MAX_PACKET_SIZE])[0]
        max_size = config.get('packet_size_range', [MODBUS_MIN_PACKET_SIZE, MODBUS_MAX_PACKET_SIZE])[1]

        # Constraint on forward (request) packet sizes
        self._hard_constraints.append(
            HardConstraint(
                name="modbus_fwd_packet_size",
                constraint_type="range",
                validation_fn=lambda x: self._validate_packet_size_range(x, min_size, max_size),
                feature_indices=[FEATURE_IDX_FWD_BYTS_B_AVG],
                parameters={'range': (min_size, max_size)},
                severity='error',
                description=f"Forward packet sizes must be {min_size}-{max_size} bytes (MBAP + PDU)"
            )
        )

        # Constraint on backward (response) packet sizes
        self._hard_constraints.append(
            HardConstraint(
                name="modbus_bwd_packet_size",
                constraint_type="range",
                validation_fn=lambda x: self._validate_packet_size_range(x, min_size, max_size),
                feature_indices=[FEATURE_IDX_BWD_BYTS_B_AVG],
                parameters={'range': (min_size, max_size)},
                severity='error',
                description=f"Backward packet sizes must be {min_size}-{max_size} bytes"
            )
        )

    def _init_timing_constraints(self, config: Dict):
        """Initialize timing hard constraints."""
        timing_min = config.get('timing_min_ms', MODBUS_MIN_RESPONSE_TIME_MS)
        timing_max = config.get('timing_max_ms', MODBUS_MAX_RESPONSE_TIME_MS)

        # Inter-arrival time constraints (convert ms to seconds for comparison)
        # CICIoT2023 features are in seconds
        timing_min_sec = timing_min / 1000.0
        timing_max_sec = timing_max / 1000.0

        self._hard_constraints.append(
            HardConstraint(
                name="modbus_fwd_timing",
                constraint_type="timing",
                validation_fn=lambda x: self._validate_timing_range(x, timing_min_sec, timing_max_sec),
                feature_indices=[FEATURE_IDX_FWD_IAT_MEAN],
                parameters={'range': (timing_min_sec, timing_max_sec)},
                severity='warning',
                description=f"Forward inter-arrival time should be {timing_min}-{timing_max}ms"
            )
        )

        self._hard_constraints.append(
            HardConstraint(
                name="modbus_bwd_timing",
                constraint_type="timing",
                validation_fn=lambda x: self._validate_timing_range(x, timing_min_sec, timing_max_sec),
                feature_indices=[FEATURE_IDX_BWD_IAT_MEAN],
                parameters={'range': (timing_min_sec, timing_max_sec)},
                severity='warning',
                description=f"Backward inter-arrival time should be {timing_min}-{timing_max}ms"
            )
        )

    def _init_statistical_constraints(self, config: Dict):
        """Initialize soft statistical constraints."""
        soft_config = config.get('soft_constraints', {})

        packet_size_mean = soft_config.get('packet_size_mean', MODBUS_TYPICAL_PACKET_SIZE_MEAN)
        packet_size_std = soft_config.get('packet_size_std', MODBUS_TYPICAL_PACKET_SIZE_STD)

        # Packet size distribution
        self._soft_constraints.append(
            SoftConstraint(
                name="modbus_packet_size_distribution",
                target_distribution="normal",
                target_params={'mean': packet_size_mean, 'std': packet_size_std},
                feature_indices=[FEATURE_IDX_FWD_BYTS_B_AVG, FEATURE_IDX_BWD_BYTS_B_AVG],
                tolerance=0.3,
                weight=1.0,
                description=f"Packet sizes should follow Normal({packet_size_mean}, {packet_size_std})"
            )
        )

        # Inter-arrival time (exponential distribution for polling patterns)
        inter_arrival_lambda = soft_config.get('inter_arrival_lambda', 0.1)
        # For exponential distribution, mean = 1/lambda
        inter_arrival_mean = 1.0 / inter_arrival_lambda

        self._soft_constraints.append(
            SoftConstraint(
                name="modbus_inter_arrival_time",
                target_distribution="exponential",
                target_params={'lambda': inter_arrival_lambda, 'mean': inter_arrival_mean},
                feature_indices=[FEATURE_IDX_FWD_IAT_MEAN],
                tolerance=0.4,  # More tolerance for timing
                weight=0.8,
                description=f"Inter-arrival times should follow Exponential(λ={inter_arrival_lambda})"
            )
        )

        # Request/response symmetry
        self._soft_constraints.append(
            SoftConstraint(
                name="modbus_request_response_ratio",
                target_distribution="uniform",
                target_params={'min': 0.8, 'max': 1.2},  # Allow some asymmetry
                feature_indices=[FEATURE_IDX_FWD_PKTS_PER_SEC, FEATURE_IDX_BWD_PKTS_PER_SEC],
                tolerance=0.3,
                weight=0.7,
                description="Request/response packet rates should be approximately 1:1"
            )
        )

    def _validate_packet_size_range(self, data: np.ndarray, min_size: float, max_size: float) -> bool:
        """Validate that packet sizes are within protocol limits."""
        # Data is average packet size from flow features
        mean_size = np.mean(data)
        # Allow some tolerance since we're looking at averages
        # Use 90th percentile to account for outliers
        p90_size = np.percentile(data, 90)

        # Check if the typical packet size is within range
        return min_size <= mean_size <= max_size and p90_size <= max_size * 1.1

    def _validate_timing_range(self, data: np.ndarray, min_time: float, max_time: float) -> bool:
        """Validate that timing is within expected range."""
        # For Modbus, timing should be relatively consistent
        mean_time = np.mean(data)
        median_time = np.median(data)

        # Both mean and median should be in reasonable range
        # Allow broader range since timing can vary, but not too lenient
        # Minimum: 1/2 of specified min (0.5ms for typical 1ms min)
        # Maximum: 10x of specified max (1s for typical 100ms max)
        extended_min = min_time / 2
        extended_max = max_time * 10  # Allow up to 1 second (10x normal max)
        return (extended_min <= mean_time <= extended_max and
                extended_min <= median_time <= extended_max)

    def get_protocol_name(self) -> str:
        """Return protocol name."""
        return "modbus"

    def _generate_suggestion(self, constraint: HardConstraint, actual_value: float) -> str:
        """Generate protocol-specific suggestions for violations."""
        if "packet_size" in constraint.name:
            expected_range = constraint.parameters.get('range')
            if expected_range:
                min_size, max_size = expected_range
                if actual_value < min_size:
                    return (f"Packet size {actual_value:.0f} bytes is below Modbus minimum. "
                            f"Ensure MBAP header (7 bytes) + PDU (≥1 byte) is present.")
                elif actual_value > max_size:
                    return (f"Packet size {actual_value:.0f} bytes exceeds Modbus maximum of {max_size} bytes. "
                            f"PDU is limited to {MODBUS_PDU_MAX_SIZE} bytes.")

        elif "timing" in constraint.name:
            expected_range = constraint.parameters.get('range')
            if expected_range:
                min_time, max_time = expected_range
                min_ms = min_time * 1000
                max_ms = max_time * 1000
                actual_ms = actual_value * 1000
                if actual_value < min_time:
                    return (f"Inter-arrival time {actual_ms:.1f}ms is unusually fast for Modbus. "
                            f"Typical range is {min_ms:.0f}-{max_ms:.0f}ms.")
                elif actual_value > max_time:
                    return (f"Inter-arrival time {actual_ms:.1f}ms is slow. "
                            f"Consider checking for network delays or timeout issues.")

        return f"Adjust value {actual_value:.2f} to meet Modbus protocol requirements"

    def get_guidance_hints(self, target_statistics: Optional[Dict] = None) -> Dict:
        """
        Get Modbus-specific guidance hints for generation.

        Returns:
            Dict with guidance parameters optimized for Modbus traffic
        """
        hints = {
            'target_mean': MODBUS_TYPICAL_PACKET_SIZE_MEAN,
            'target_std': MODBUS_TYPICAL_PACKET_SIZE_STD,
            'feature_ranges': {
                FEATURE_IDX_FWD_BYTS_B_AVG: (MODBUS_MIN_PACKET_SIZE, MODBUS_MAX_PACKET_SIZE),
                FEATURE_IDX_BWD_BYTS_B_AVG: (MODBUS_MIN_PACKET_SIZE, MODBUS_MAX_PACKET_SIZE),
                FEATURE_IDX_FWD_IAT_MEAN: (MODBUS_MIN_RESPONSE_TIME_MS/1000, MODBUS_MAX_RESPONSE_TIME_MS/1000),
                FEATURE_IDX_BWD_IAT_MEAN: (MODBUS_MIN_RESPONSE_TIME_MS/1000, MODBUS_MAX_RESPONSE_TIME_MS/1000),
            },
            'protocol': 'modbus',
            'port': MODBUS_PORT,
            'request_response_model': True
        }

        # Override with user-provided statistics if available
        if target_statistics:
            hints.update(target_statistics)

        return hints

    def validate_function_code_distribution(self, sample: np.ndarray) -> bool:
        """
        Validate that the traffic pattern is consistent with typical Modbus function code usage.

        This is an advanced validation that looks at traffic patterns to infer
        whether the function code distribution seems realistic.

        Note: Since CICIoT2023 doesn't have function codes directly, we infer
        from packet size patterns (different function codes have different response sizes).
        """
        # Check if packet size variance suggests multiple function codes
        fwd_sizes = sample[:, FEATURE_IDX_FWD_BYTS_B_AVG]
        bwd_sizes = sample[:, FEATURE_IDX_BWD_BYTS_B_AVG]

        # Typical Modbus operations have distinct sizes:
        # - Read operations: Small request (12-15 bytes), larger response (varies)
        # - Write operations: Larger request (varies), small response (12 bytes)

        # Check for reasonable variance (indicating multiple operation types)
        fwd_std = np.std(fwd_sizes)
        bwd_std = np.std(bwd_sizes)

        # Some variance is expected, but not too much
        reasonable_variance = (5 < fwd_std < 50) and (5 < bwd_std < 50)

        return reasonable_variance
