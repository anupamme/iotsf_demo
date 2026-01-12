"""
CoAP Protocol Validator

Implements protocol-specific constraints for CoAP (Constrained Application Protocol).
CoAP is a RESTful protocol for resource-constrained devices, typically on UDP port 5683.

Protocol Structure:
- Fixed Header: 4 bytes (version, type, token length, code, message ID)
- Token: 0-8 bytes
- Options: Variable length (delta-encoded)
- Payload: 0-1024 bytes (typical, fits in UDP datagram)

Reference: RFC 7252 - The Constrained Application Protocol
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger

from .base import ProtocolValidator
from ..types import (
    HardConstraint,
    SoftConstraint,
    COAP_FEATURE_MAP,
    FEATURE_IDX_FWD_BYTS_B_AVG,
    FEATURE_IDX_BWD_BYTS_B_AVG,
    FEATURE_IDX_FWD_IAT_MEAN,
    FEATURE_IDX_BWD_IAT_MEAN
)


# CoAP Protocol Constants
COAP_PORT = 5683
COAP_PORT_DTLS = 5684
COAP_HEADER_SIZE = 4  # Fixed 4-byte header
COAP_MIN_PACKET_SIZE = COAP_HEADER_SIZE  # Header only
COAP_MAX_PACKET_SIZE = 1280  # To fit in single UDP datagram (IPv6 MTU)
COAP_TOKEN_MIN_LENGTH = 0
COAP_TOKEN_MAX_LENGTH = 8

# CoAP Message Types
COAP_TYPE_CON = 0  # Confirmable
COAP_TYPE_NON = 1  # Non-confirmable
COAP_TYPE_ACK = 2  # Acknowledgement
COAP_TYPE_RST = 3  # Reset

# CoAP Method Codes
COAP_METHOD_GET = 1
COAP_METHOD_POST = 2
COAP_METHOD_PUT = 3
COAP_METHOD_DELETE = 4

# Timing (milliseconds)
COAP_ACK_TIMEOUT_MS = 2000  # 2-3 seconds typical
COAP_MAX_RETRANSMIT = 4
COAP_OBSERVE_MIN_INTERVAL = 1  # seconds
COAP_OBSERVE_MAX_INTERVAL = 300  # seconds

# Statistical properties
COAP_TYPICAL_PACKET_SIZE_MEAN = 128  # Bimodal: small requests, larger responses
COAP_TYPICAL_PACKET_SIZE_STD = 96
COAP_TYPE_DISTRIBUTION = [0.3, 0.2, 0.3, 0.2]  # CON, NON, ACK, RST


class CoAPValidator(ProtocolValidator):
    """Validator for CoAP protocol constraints."""

    def _initialize_constraints(self):
        """Initialize CoAP-specific constraints."""
        config = self.config.get('coap', {})

        self._init_packet_size_constraints(config)
        self._init_timing_constraints(config)
        self._init_statistical_constraints(config)

    def _init_packet_size_constraints(self, config: Dict):
        """Initialize packet size constraints."""
        min_size = config.get('packet_size_range', [COAP_MIN_PACKET_SIZE, COAP_MAX_PACKET_SIZE])[0]
        max_size = config.get('packet_size_range', [COAP_MIN_PACKET_SIZE, COAP_MAX_PACKET_SIZE])[1]

        self._hard_constraints.append(
            HardConstraint(
                name="coap_fwd_packet_size",
                constraint_type="range",
                validation_fn=lambda x: self._validate_packet_size_range(x, min_size, max_size),
                feature_indices=[FEATURE_IDX_FWD_BYTS_B_AVG],
                parameters={'range': (min_size, max_size)},
                severity='error',
                description=f"Forward packet sizes must be {min_size}-{max_size} bytes"
            )
        )

        self._hard_constraints.append(
            HardConstraint(
                name="coap_bwd_packet_size",
                constraint_type="range",
                validation_fn=lambda x: self._validate_packet_size_range(x, min_size, max_size),
                feature_indices=[FEATURE_IDX_BWD_BYTS_B_AVG],
                parameters={'range': (min_size, max_size)},
                severity='error',
                description=f"Backward packet sizes must be {min_size}-{max_size} bytes"
            )
        )

    def _init_timing_constraints(self, config: Dict):
        """Initialize timing constraints."""
        # CoAP ACK timeout and retransmission
        ack_timeout_sec = COAP_ACK_TIMEOUT_MS / 1000.0

        self._hard_constraints.append(
            HardConstraint(
                name="coap_response_timing",
                constraint_type="timing",
                validation_fn=lambda x: self._validate_timing_range(x, 0.001, ack_timeout_sec * 5),
                feature_indices=[FEATURE_IDX_FWD_IAT_MEAN, FEATURE_IDX_BWD_IAT_MEAN],
                parameters={'range': (0.001, ack_timeout_sec * 5)},
                severity='warning',
                description="CoAP response timing should be within reasonable bounds"
            )
        )

    def _init_statistical_constraints(self, config: Dict):
        """Initialize soft statistical constraints."""
        soft_config = config.get('soft_constraints', {})

        packet_size_mean = soft_config.get('packet_size_mean', COAP_TYPICAL_PACKET_SIZE_MEAN)
        packet_size_std = soft_config.get('packet_size_std', COAP_TYPICAL_PACKET_SIZE_STD)

        # Bimodal distribution: Small requests, larger responses
        self._soft_constraints.append(
            SoftConstraint(
                name="coap_packet_size_distribution",
                target_distribution="normal",  # Simplified from bimodal
                target_params={'mean': packet_size_mean, 'std': packet_size_std},
                feature_indices=[FEATURE_IDX_FWD_BYTS_B_AVG, FEATURE_IDX_BWD_BYTS_B_AVG],
                tolerance=0.4,
                weight=0.9,
                description=f"Packet sizes should be centered around {packet_size_mean}±{packet_size_std} bytes"
            )
        )

        # Request-response timing
        self._soft_constraints.append(
            SoftConstraint(
                name="coap_request_response_timing",
                target_distribution="exponential",
                target_params={'lambda': 0.2, 'mean': 5.0},  # 5 second mean interval
                feature_indices=[FEATURE_IDX_FWD_IAT_MEAN],
                tolerance=0.5,
                weight=0.7,
                description="CoAP requests follow exponential distribution for polling"
            )
        )

    def _validate_packet_size_range(self, data: np.ndarray, min_size: float, max_size: float) -> bool:
        """Validate packet sizes within CoAP limits."""
        mean_size = np.mean(data)
        p95_size = np.percentile(data, 95)

        # Check mean and 95th percentile
        return min_size <= mean_size <= max_size and p95_size <= max_size * 1.1

    def _validate_timing_range(self, data: np.ndarray, min_time: float, max_time: float) -> bool:
        """Validate timing is within expected range."""
        mean_time = np.mean(data)
        median_time = np.median(data)

        # Allow broader range for CoAP (UDP, retransmissions)
        extended_min = min_time / 2
        extended_max = max_time * 2

        return extended_min <= mean_time <= extended_max and extended_min <= median_time <= extended_max

    def get_protocol_name(self) -> str:
        """Return protocol name."""
        return "coap"

    def _generate_suggestion(self, constraint: HardConstraint, actual_value: float) -> str:
        """Generate protocol-specific suggestions."""
        if "packet_size" in constraint.name:
            expected_range = constraint.parameters.get('range')
            if expected_range:
                min_size, max_size = expected_range
                if actual_value < min_size:
                    return f"Packet size {actual_value:.0f} bytes is below CoAP minimum. Ensure 4-byte header is present."
                elif actual_value > max_size:
                    return f"Packet size {actual_value:.0f} bytes exceeds typical UDP datagram limit ({max_size} bytes)."

        elif "timing" in constraint.name:
            return f"Response timing {actual_value * 1000:.1f}ms seems unusual for CoAP request-response pattern."

        return f"Adjust value {actual_value:.2f} to meet CoAP protocol requirements"

    def get_guidance_hints(self, target_statistics: Optional[Dict] = None) -> Dict:
        """Get CoAP-specific guidance hints."""
        hints = {
            'target_mean': COAP_TYPICAL_PACKET_SIZE_MEAN,
            'target_std': COAP_TYPICAL_PACKET_SIZE_STD,
            'feature_ranges': {
                FEATURE_IDX_FWD_BYTS_B_AVG: (COAP_MIN_PACKET_SIZE, COAP_MAX_PACKET_SIZE),
                FEATURE_IDX_BWD_BYTS_B_AVG: (COAP_MIN_PACKET_SIZE, COAP_MAX_PACKET_SIZE),
                FEATURE_IDX_FWD_IAT_MEAN: (0.001, 10.0),
                FEATURE_IDX_BWD_IAT_MEAN: (0.001, 10.0),
            },
            'protocol': 'coap',
            'port': COAP_PORT,
            'transport': 'udp',
            'pattern': 'request-response'
        }

        if target_statistics:
            hints.update(target_statistics)

        return hints
