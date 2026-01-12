"""
MQTT Protocol Validator

Implements protocol-specific constraints for MQTT (Message Queuing Telemetry Transport).
MQTT is a lightweight publish-subscribe protocol for IoT devices, typically on port 1883.

Protocol Structure:
- Fixed Header: 2-5 bytes (message type, flags, remaining length)
- Variable Header: 0-10 bytes (depends on message type)
- Payload: 0-268,435,455 bytes (practical limit ~256KB)

Reference: MQTT Version 3.1.1 Specification
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger

from .base import ProtocolValidator
from ..types import (
    HardConstraint,
    SoftConstraint,
    MQTT_FEATURE_MAP,
    FEATURE_IDX_FWD_BYTS_B_AVG,
    FEATURE_IDX_BWD_BYTS_B_AVG,
    FEATURE_IDX_FWD_IAT_MEAN,
    FEATURE_IDX_BWD_IAT_MEAN,
    FEATURE_IDX_FLOW_DURATION,
    FEATURE_IDX_FWD_PKTS_PER_SEC,
    FEATURE_IDX_BWD_PKTS_PER_SEC
)


# MQTT Protocol Constants
MQTT_PORT_STANDARD = 1883
MQTT_PORT_TLS = 8883
MQTT_MIN_PACKET_SIZE = 2  # Fixed header minimum
MQTT_TYPICAL_MAX_PACKET_SIZE = 1024  # Practical limit for most IoT
MQTT_PROTOCOL_MAX_PACKET_SIZE = 268435455  # Theoretical max

# Valid MQTT message types (control packet types)
MQTT_VALID_MESSAGE_TYPES = [
    1,   # CONNECT
    2,   # CONNACK
    3,   # PUBLISH
    4,   # PUBACK
    5,   # PUBREC
    6,   # PUBREL
    7,   # PUBCOMP
    8,   # SUBSCRIBE
    9,   # SUBACK
    10,  # UNSUBSCRIBE
    11,  # UNSUBACK
    12,  # PINGREQ
    13,  # PINGRESP
    14,  # DISCONNECT
]

# QoS Levels
MQTT_QOS_LEVELS = [0, 1, 2]  # At most once, At least once, Exactly once

# Keep-alive timing (seconds)
MQTT_MIN_KEEPALIVE = 60
MQTT_MAX_KEEPALIVE = 300
MQTT_KEEPALIVE_DISABLED = 0

# Statistical properties for benign MQTT traffic
MQTT_TYPICAL_PACKET_SIZE_MEAN = 128  # bytes (log-normal distribution)
MQTT_TYPICAL_PACKET_SIZE_STD = 256   # Large variance due to payload variability
MQTT_QOS_DISTRIBUTION = [0.7, 0.25, 0.05]  # QoS 0, 1, 2 probabilities


class MQTTValidator(ProtocolValidator):
    """
    Validator for MQTT protocol constraints.

    Validates both hard constraints (protocol rules) and soft constraints
    (statistical properties) for MQTT pub-sub traffic flows.
    """

    def _initialize_constraints(self):
        """Initialize MQTT-specific constraints."""
        config = self.config.get('mqtt', {})

        # Hard Constraints
        self._init_packet_size_constraints(config)
        self._init_keepalive_constraints(config)

        # Soft Constraints
        self._init_statistical_constraints(config)

    def _init_packet_size_constraints(self, config: Dict):
        """Initialize packet size hard constraints."""
        min_size = config.get('packet_size_range', [MQTT_MIN_PACKET_SIZE, MQTT_TYPICAL_MAX_PACKET_SIZE])[0]
        max_size = config.get('packet_size_range', [MQTT_MIN_PACKET_SIZE, MQTT_TYPICAL_MAX_PACKET_SIZE])[1]

        # Forward (client to broker) packet sizes
        self._hard_constraints.append(
            HardConstraint(
                name="mqtt_fwd_packet_size",
                constraint_type="range",
                validation_fn=lambda x: self._validate_packet_size_range(x, min_size, max_size),
                feature_indices=[FEATURE_IDX_FWD_BYTS_B_AVG],
                parameters={'range': (min_size, max_size)},
                severity='error',
                description=f"Forward packet sizes must be {min_size}-{max_size} bytes"
            )
        )

        # Backward (broker to client) packet sizes
        self._hard_constraints.append(
            HardConstraint(
                name="mqtt_bwd_packet_size",
                constraint_type="range",
                validation_fn=lambda x: self._validate_packet_size_range(x, min_size, max_size),
                feature_indices=[FEATURE_IDX_BWD_BYTS_B_AVG],
                parameters={'range': (min_size, max_size)},
                severity='error',
                description=f"Backward packet sizes must be {min_size}-{max_size} bytes"
            )
        )

    def _init_keepalive_constraints(self, config: Dict):
        """Initialize keep-alive timing constraints."""
        keepalive_range = config.get('keep_alive_range', [MQTT_MIN_KEEPALIVE, MQTT_MAX_KEEPALIVE])

        # Connection duration should reflect keep-alive pattern
        # MQTT connections are typically long-lived
        self._hard_constraints.append(
            HardConstraint(
                name="mqtt_connection_duration",
                constraint_type="timing",
                validation_fn=lambda x: self._validate_connection_duration(x),
                feature_indices=[FEATURE_IDX_FLOW_DURATION],
                parameters={'min_duration': 10.0},  # At least 10 seconds for meaningful connection
                severity='warning',
                description="MQTT connections typically persist for extended periods"
            )
        )

    def _init_statistical_constraints(self, config: Dict):
        """Initialize soft statistical constraints."""
        soft_config = config.get('soft_constraints', {})

        packet_size_mean = soft_config.get('packet_size_mean', MQTT_TYPICAL_PACKET_SIZE_MEAN)
        packet_size_std = soft_config.get('packet_size_std', MQTT_TYPICAL_PACKET_SIZE_STD)

        # Packet size distribution (log-normal due to high variance)
        # MQTT can have very small messages (PINGREQ) or large payloads (PUBLISH)
        self._soft_constraints.append(
            SoftConstraint(
                name="mqtt_packet_size_distribution",
                target_distribution="lognormal",
                target_params={'mean': packet_size_mean, 'std': packet_size_std},
                feature_indices=[FEATURE_IDX_FWD_BYTS_B_AVG, FEATURE_IDX_BWD_BYTS_B_AVG],
                tolerance=0.5,  # Higher tolerance for MQTT due to variability
                weight=0.8,
                description=f"Packet sizes should follow log-normal distribution (μ={packet_size_mean})"
            )
        )

        # Keep-alive timing
        keepalive_min = soft_config.get('keep_alive_min', MQTT_MIN_KEEPALIVE)
        keepalive_max = soft_config.get('keep_alive_max', MQTT_MAX_KEEPALIVE)

        self._soft_constraints.append(
            SoftConstraint(
                name="mqtt_keepalive_timing",
                target_distribution="uniform",
                target_params={'min': keepalive_min, 'max': keepalive_max},
                feature_indices=[FEATURE_IDX_FWD_IAT_MEAN],
                tolerance=0.4,
                weight=0.6,
                description=f"Keep-alive intervals typically {keepalive_min}-{keepalive_max}s"
            )
        )

        # Publish/subscribe asymmetry
        # In pub-sub, one publisher can send to many subscribers (1:N pattern)
        # Check the ratio of forward to backward packet rates
        # Typical asymmetry: 0.1 (subscriber-heavy) to 10.0 (publisher-heavy)
        self._hard_constraints.append(
            HardConstraint(
                name="mqtt_pubsub_asymmetry",
                constraint_type="range",
                validation_fn=lambda x: self._validate_pubsub_asymmetry(x),
                feature_indices=[FEATURE_IDX_FWD_PKTS_PER_SEC, FEATURE_IDX_BWD_PKTS_PER_SEC],
                parameters={'min_ratio': 0.1, 'max_ratio': 10.0},
                severity='warning',
                description="Pub-sub pattern allows asymmetric packet rates (0.1x to 10x)"
            )
        )

    def _validate_packet_size_range(self, data: np.ndarray, min_size: float, max_size: float) -> bool:
        """Validate that packet sizes are within protocol limits."""
        mean_size = np.mean(data)
        p90_size = np.percentile(data, 90)

        # MQTT has high variance, so check mean and high percentile
        return min_size <= mean_size <= max_size and p90_size <= max_size * 1.2

    def _validate_connection_duration(self, data: np.ndarray) -> bool:
        """Validate connection duration is consistent with MQTT patterns."""
        mean_duration = np.mean(data)

        # MQTT connections are typically persistent (not one-shot like HTTP)
        # Allow very short connections for testing, but flag suspiciously short
        min_duration = 10.0  # seconds
        return mean_duration >= min_duration or mean_duration == 0  # 0 means ongoing

    def _validate_pubsub_asymmetry(self, data: np.ndarray) -> bool:
        """
        Validate pub-sub asymmetry by comparing forward and backward packet rates.

        In MQTT pub-sub:
        - Publishers send data (forward packets)
        - Subscribers receive data (backward packets)
        - The ratio can be asymmetric (e.g., 1 publisher -> N subscribers)

        Args:
            data: Array with shape (seq_length, 2) where:
                  column 0 = forward packets per second
                  column 1 = backward packets per second

        Returns:
            True if asymmetry ratio is within acceptable range (0.1x to 10x)
        """
        # Extract forward and backward rates
        fwd_rate = data[:, 0]
        bwd_rate = data[:, 1]

        # Compute mean rates
        mean_fwd = np.mean(fwd_rate)
        mean_bwd = np.mean(bwd_rate)

        # Handle edge cases
        if mean_fwd < 1e-6 and mean_bwd < 1e-6:
            # Both very close to zero - acceptable (idle connection)
            return True

        if mean_bwd < 1e-6:
            # Backward rate is zero but forward isn't - likely publisher-only
            # This is acceptable for pub-sub pattern
            return True

        # Compute asymmetry ratio (forward / backward)
        ratio = mean_fwd / (mean_bwd + 1e-8)

        # Check if ratio is within acceptable range
        # 0.1x means subscriber-heavy (10x more backward)
        # 10x means publisher-heavy (10x more forward)
        min_ratio = 0.1
        max_ratio = 10.0

        return min_ratio <= ratio <= max_ratio

    def get_protocol_name(self) -> str:
        """Return protocol name."""
        return "mqtt"

    def _generate_suggestion(self, constraint: HardConstraint, actual_value: float) -> str:
        """Generate protocol-specific suggestions for violations."""
        if "packet_size" in constraint.name:
            expected_range = constraint.parameters.get('range')
            if expected_range:
                min_size, max_size = expected_range
                if actual_value < min_size:
                    return (f"Packet size {actual_value:.0f} bytes is below MQTT minimum. "
                            f"Ensure fixed header (2-5 bytes) is present.")
                elif actual_value > max_size:
                    return (f"Packet size {actual_value:.0f} bytes exceeds typical MQTT payload. "
                            f"Consider message fragmentation or check for abnormal traffic.")

        elif "connection_duration" in constraint.name:
            if actual_value < 10.0:
                return (f"Connection duration {actual_value:.1f}s is very short for MQTT. "
                        f"MQTT typically maintains persistent connections with keep-alive.")

        elif "pubsub_asymmetry" in constraint.name:
            return (f"Pub-sub asymmetry ratio {actual_value:.2f} is outside acceptable range (0.1x to 10x). "
                    f"Adjust forward/backward packet rates to reflect realistic pub-sub patterns.")

        return f"Adjust value {actual_value:.2f} to meet MQTT protocol requirements"

    def get_guidance_hints(self, target_statistics: Optional[Dict] = None) -> Dict:
        """
        Get MQTT-specific guidance hints for generation.

        Returns:
            Dict with guidance parameters optimized for MQTT traffic
        """
        hints = {
            'target_mean': MQTT_TYPICAL_PACKET_SIZE_MEAN,
            'target_std': MQTT_TYPICAL_PACKET_SIZE_STD,
            'feature_ranges': {
                FEATURE_IDX_FWD_BYTS_B_AVG: (MQTT_MIN_PACKET_SIZE, MQTT_TYPICAL_MAX_PACKET_SIZE),
                FEATURE_IDX_BWD_BYTS_B_AVG: (MQTT_MIN_PACKET_SIZE, MQTT_TYPICAL_MAX_PACKET_SIZE),
                FEATURE_IDX_FLOW_DURATION: (60.0, 3600.0),  # 1 minute to 1 hour typical
            },
            'protocol': 'mqtt',
            'port': MQTT_PORT_STANDARD,
            'connection_model': 'persistent',
            'pattern': 'pub-sub'
        }

        if target_statistics:
            hints.update(target_statistics)

        return hints

    def validate_qos_consistency(self, sample: np.ndarray) -> bool:
        """
        Validate QoS-related traffic patterns.

        QoS 0: Fire and forget (PUBLISH only)
        QoS 1: At least once (PUBLISH → PUBACK)
        QoS 2: Exactly once (PUBLISH → PUBREC → PUBREL → PUBCOMP)

        Higher QoS means more packet exchanges and potentially larger flows.
        """
        # Infer QoS from packet patterns
        fwd_packets = sample[:, FEATURE_IDX_FWD_BYTS_B_AVG]
        bwd_packets = sample[:, FEATURE_IDX_BWD_BYTS_B_AVG]

        # QoS 0: Minimal backward traffic
        # QoS 1/2: More symmetric traffic pattern
        fwd_mean = np.mean(fwd_packets)
        bwd_mean = np.mean(bwd_packets)

        # Check if ratio seems reasonable for mixed QoS
        ratio = bwd_mean / (fwd_mean + 1e-8)
        reasonable_ratio = 0.1 <= ratio <= 10.0  # Allow wide range

        return reasonable_ratio

    def validate_message_type_diversity(self, sample: np.ndarray) -> bool:
        """
        Validate that traffic shows diversity consistent with MQTT operations.

        Real MQTT traffic includes connection management (CONNECT, PINGREQ),
        subscriptions (SUBSCRIBE), and data (PUBLISH).
        """
        # Check packet size variance as proxy for message type diversity
        fwd_sizes = sample[:, FEATURE_IDX_FWD_BYTS_B_AVG]
        fwd_std = np.std(fwd_sizes)

        # Some variance expected (different message types have different sizes)
        # But not too much (all messages are relatively small)
        reasonable_variance = 10 < fwd_std < 500

        return reasonable_variance
