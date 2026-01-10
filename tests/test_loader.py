"""Tests for CICIoT2023 data loader."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.loader import CICIoT2023Loader


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create temporary sample data for testing."""
    # Create benign data
    benign_data = pd.DataFrame({
        'flow_duration': np.random.randn(100),
        'fwd_pkts_tot': np.random.randn(100),
        'bwd_pkts_tot': np.random.randn(100),
        'fwd_data_pkts_tot': np.random.randn(100),
        'bwd_data_pkts_tot': np.random.randn(100),
        'fwd_pkts_per_sec': np.random.randn(100),
        'bwd_pkts_per_sec': np.random.randn(100),
        'flow_pkts_per_sec': np.random.randn(100),
        'fwd_byts_b_avg': np.random.randn(100),
        'bwd_byts_b_avg': np.random.randn(100),
        'fwd_iat_mean': np.random.randn(100),
        'bwd_iat_mean': np.random.randn(100),
        'label': 'Benign'
    })
    benign_data.to_csv(tmp_path / 'benign_traffic.csv', index=False)

    # Create attack data
    attack_data = benign_data.copy()
    attack_data['label'] = 'DDoS-TCP_Flood'
    attack_data.to_csv(tmp_path / 'ddos_attack.csv', index=False)

    return tmp_path


def test_loader_initialization(sample_data_dir):
    """Test loader can be initialized."""
    loader = CICIoT2023Loader(sample_data_dir)
    assert loader.data_dir == sample_data_dir


def test_load_benign_samples(sample_data_dir):
    """Test loading benign samples."""
    loader = CICIoT2023Loader(sample_data_dir)
    benign = loader.load_benign_samples(50)

    assert len(benign) == 50
    assert len(benign.columns) == 12
    assert all(col in benign.columns for col in loader.FEATURE_COLUMNS)


def test_load_attack_samples(sample_data_dir):
    """Test loading attack samples."""
    loader = CICIoT2023Loader(sample_data_dir)
    attack = loader.load_attack_samples('DDoS-TCP_Flood', 50)

    assert len(attack) == 50
    assert len(attack.columns) == 12


def test_get_mixed_batch(sample_data_dir):
    """Test creating mixed batch of benign and attack samples."""
    loader = CICIoT2023Loader(sample_data_dir)
    data, labels = loader.get_mixed_batch(
        n_benign=30,
        n_attack=20,
        attack_types=['DDoS-TCP_Flood']
    )

    assert len(data) == 50
    assert len(labels) == 50
    assert np.sum(labels == 0) == 30  # benign
    assert np.sum(labels == 1) == 20  # attack


def test_get_statistics(sample_data_dir):
    """Test getting data statistics."""
    loader = CICIoT2023Loader(sample_data_dir)
    stats = loader.get_statistics()

    assert 'data_dir' in stats
    assert 'files' in stats
    assert len(stats['files']) == 2  # benign and attack


def test_load_benign_samples_insufficient_data(sample_data_dir):
    """Test loading when requested samples exceed available data."""
    loader = CICIoT2023Loader(sample_data_dir)
    benign = loader.load_benign_samples(200)  # Request more than available

    assert len(benign) == 100  # Should return all available


def test_load_attack_samples_with_filter(sample_data_dir):
    """Test loading attack samples with label filtering."""
    loader = CICIoT2023Loader(sample_data_dir)
    attack = loader.load_attack_samples('DDoS-TCP_Flood', 30)

    assert len(attack) == 30


def test_caching(sample_data_dir):
    """Test that data caching works."""
    loader = CICIoT2023Loader(sample_data_dir)

    # First load
    benign1 = loader.load_benign_samples(10)

    # Second load (should use cache)
    benign2 = loader.load_benign_samples(10)

    # Both should have same data (same random seed)
    pd.testing.assert_frame_equal(benign1, benign2)
