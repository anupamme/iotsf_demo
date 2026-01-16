"""Shared pytest fixtures and configuration."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import yaml


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return data directory path."""
    return project_root / "data"


@pytest.fixture(scope="session")
def config_dir(project_root):
    """Return config directory path."""
    return project_root / "config"


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def sample_config(tmp_path):
    """Create a temporary test configuration."""
    config = {
        "project": {"name": "Test Project", "version": "0.1.0"},
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "synthetic_dir": "data/synthetic",
            "ciciot_subset_size": 1000
        },
        "models": {
            "diffusion_ts": {
                "seq_length": 32,
                "feature_dim": 12,
                "n_diffusion_steps": 10
            },
            "moirai": {
                "model_size": "small",
                "context_length": 256,
                "prediction_length": 32
            },
            "baseline": {"method": "threshold"}
        },
        "device": {"use_gpu": False, "gpu_id": 0},
        "demo": {
            "n_benign_samples": 2,
            "n_attack_samples": 2,
            "attack_types": ["slow_exfiltration", "beacon"]
        }
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def default_config(config_dir):
    """Load the actual default config.yaml."""
    from src.utils.config import Config
    return Config(config_dir / "config.yaml")


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_traffic_data():
    """Generate synthetic IoT traffic data for testing."""
    n_samples = 100

    data = pd.DataFrame({
        'flow_duration': np.random.randn(n_samples),
        'fwd_pkts_tot': np.random.randn(n_samples),
        'bwd_pkts_tot': np.random.randn(n_samples),
        'fwd_data_pkts_tot': np.random.randn(n_samples),
        'bwd_data_pkts_tot': np.random.randn(n_samples),
        'fwd_pkts_per_sec': np.random.randn(n_samples),
        'bwd_pkts_per_sec': np.random.randn(n_samples),
        'flow_pkts_per_sec': np.random.randn(n_samples),
        'fwd_byts_b_avg': np.random.randn(n_samples),
        'bwd_byts_b_avg': np.random.randn(n_samples),
        'fwd_iat_mean': np.random.randn(n_samples),
        'bwd_iat_mean': np.random.randn(n_samples),
    })

    return data


@pytest.fixture
def sample_sequences():
    """Generate sample time-series sequences."""
    # Shape: (n_samples, seq_length, n_features)
    return np.random.randn(10, 128, 12)


@pytest.fixture
def sample_benign_sequence():
    """Generate single benign sequence."""
    return np.random.randn(128, 12)


@pytest.fixture
def sample_attack_sequence():
    """Generate single attack sequence (with anomalies)."""
    sequence = np.random.randn(128, 12)
    # Inject obvious anomaly
    sequence[64:80, :] *= 3.0
    return sequence


@pytest.fixture
def mixed_batch():
    """Generate mixed batch of benign and attack samples."""
    benign = np.random.randn(50, 128, 12)
    attack = np.random.randn(50, 128, 12) * 1.5  # Slightly amplified

    data = np.vstack([benign, attack])
    labels = np.array([0] * 50 + [1] * 50)  # 0=benign, 1=attack

    # Shuffle
    indices = np.random.permutation(100)
    return data[indices], labels[indices]


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def diffusion_generator():
    """Create and initialize a Diffusion-TS generator."""
    from src.models.diffusion_ts import IoTDiffusionGenerator

    generator = IoTDiffusionGenerator(
        seq_length=64,
        feature_dim=12,
        device='cpu'
    )
    generator.initialize()

    return generator


@pytest.fixture
def preprocessor():
    """Create a fitted preprocessor."""
    from src.data.preprocessor import TrafficPreprocessor

    prep = TrafficPreprocessor('standard')
    # Fit on random data
    data = np.random.randn(100, 12)
    prep.fit(data)

    return prep


# =============================================================================
# File System Fixtures
# =============================================================================

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "synthetic").mkdir(parents=True)

    return data_dir


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file for testing loader."""
    data = pd.DataFrame({
        'flow_duration': np.random.randn(200),
        'fwd_pkts_tot': np.random.randn(200),
        'bwd_pkts_tot': np.random.randn(200),
        'fwd_data_pkts_tot': np.random.randn(200),
        'bwd_data_pkts_tot': np.random.randn(200),
        'fwd_pkts_per_sec': np.random.randn(200),
        'bwd_pkts_per_sec': np.random.randn(200),
        'flow_pkts_per_sec': np.random.randn(200),
        'fwd_byts_b_avg': np.random.randn(200),
        'bwd_byts_b_avg': np.random.randn(200),
        'fwd_iat_mean': np.random.randn(200),
        'bwd_iat_mean': np.random.randn(200),
        'label': ['Benign'] * 100 + ['DDoS-TCP_Flood'] * 100
    })

    csv_path = tmp_path / "test_traffic.csv"
    data.to_csv(csv_path, index=False)

    return csv_path


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add project root to sys.path
    import sys
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))


def pytest_collection_modifyitems(config, items):
    """Modify test collection (e.g., auto-add markers)."""
    for item in items:
        # Auto-mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark UI tests
        if "test_app" in item.name or "test_components" in item.name:
            item.add_marker(pytest.mark.ui)

        # Auto-mark slow tests
        if "large" in item.name or "batch" in item.name:
            item.add_marker(pytest.mark.slow)
