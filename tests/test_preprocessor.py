"""Tests for traffic data preprocessor."""

import pytest
import numpy as np
from src.data.preprocessor import (
    TrafficPreprocessor,
    create_sequences,
    train_val_test_split
)


def test_preprocessor_standard_scaler():
    """Test StandardScaler normalization."""
    data = np.random.randn(1000, 12) * 10 + 5

    preprocessor = TrafficPreprocessor('standard')
    normalized = preprocessor.fit_transform(data)

    # Check zero mean and unit variance
    assert np.abs(normalized.mean()) < 0.1
    assert np.abs(normalized.std() - 1.0) < 0.1


def test_preprocessor_minmax_scaler():
    """Test MinMaxScaler normalization."""
    data = np.random.randn(1000, 12) * 10 + 5

    preprocessor = TrafficPreprocessor('minmax')
    normalized = preprocessor.fit_transform(data)

    # Check range [0, 1] (with small tolerance for floating point precision)
    assert normalized.min() >= -1e-10
    assert normalized.max() <= 1 + 1e-10


def test_preprocessor_3d_data():
    """Test preprocessing 3D time-series data."""
    data = np.random.randn(100, 128, 12)  # (n_samples, seq_len, n_features)

    preprocessor = TrafficPreprocessor('standard')
    normalized = preprocessor.fit_transform(data)

    assert normalized.shape == data.shape


def test_preprocessor_inverse_transform():
    """Test inverse transformation."""
    data = np.random.randn(100, 12)

    preprocessor = TrafficPreprocessor('standard')
    normalized = preprocessor.fit_transform(data)
    recovered = preprocessor.inverse_transform(normalized)

    # Should be close to original
    assert np.allclose(data, recovered, rtol=1e-5)


def test_preprocessor_save_load(tmp_path):
    """Test saving and loading scaler."""
    data = np.random.randn(100, 12)

    # Fit and save
    preprocessor1 = TrafficPreprocessor('standard')
    preprocessor1.fit(data)
    save_path = tmp_path / 'scaler.pkl'
    preprocessor1.save(str(save_path))

    # Load and compare
    preprocessor2 = TrafficPreprocessor()
    preprocessor2.load(str(save_path))

    test_data = np.random.randn(10, 12)
    result1 = preprocessor1.transform(test_data)
    result2 = preprocessor2.transform(test_data)

    assert np.allclose(result1, result2)


def test_preprocessor_not_fitted_error():
    """Test that transform raises error when not fitted."""
    preprocessor = TrafficPreprocessor('standard')
    data = np.random.randn(10, 12)

    with pytest.raises(RuntimeError, match="not fitted"):
        preprocessor.transform(data)


def test_create_sequences():
    """Test sequence creation from flow records."""
    data = np.random.randn(1000, 12)
    seq_length = 128
    stride = 64

    sequences = create_sequences(data, seq_length, stride)

    expected_n_seq = (1000 - 128) // 64 + 1
    assert sequences.shape == (expected_n_seq, 128, 12)


def test_create_sequences_stride_1():
    """Test maximum overlap (stride=1)."""
    data = np.random.randn(200, 12)
    sequences = create_sequences(data, seq_length=128, stride=1)

    # Should have 200 - 128 + 1 = 73 sequences
    assert sequences.shape[0] == 73


def test_create_sequences_insufficient_data():
    """Test error when data is too short."""
    data = np.random.randn(50, 12)

    with pytest.raises(ValueError, match="less than sequence length"):
        create_sequences(data, seq_length=128, stride=64)


def test_train_val_test_split():
    """Test data splitting."""
    data = np.random.randn(1000, 12)
    labels = np.random.randint(0, 2, 1000)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        data, labels, train_ratio=0.7, val_ratio=0.15
    )

    # Check sizes
    assert len(X_train) == 700
    assert len(X_val) == 150
    assert len(X_test) == 150

    # Check labels match
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)


def test_train_val_test_split_without_labels():
    """Test splitting without labels."""
    data = np.random.randn(1000, 12)

    X_train, X_val, X_test = train_val_test_split(
        data, train_ratio=0.7, val_ratio=0.15
    )

    # Check sizes
    assert len(X_train) == 700
    assert len(X_val) == 150
    assert len(X_test) == 150


def test_train_val_test_split_no_shuffle():
    """Test splitting without shuffling."""
    data = np.arange(100).reshape(100, 1)

    X_train, X_val, X_test = train_val_test_split(
        data, train_ratio=0.7, val_ratio=0.15, shuffle=False
    )

    # Check that data is in order
    assert X_train[0, 0] == 0
    assert X_train[-1, 0] == 69
    assert X_val[0, 0] == 70


def test_sequence_overlap():
    """Test that sequences have correct overlap."""
    data = np.arange(300).reshape(300, 1)
    sequences = create_sequences(data, seq_length=10, stride=5)

    # Check first two sequences overlap correctly
    # First sequence: 0-9, second sequence: 5-14
    assert sequences[0, 0, 0] == 0
    assert sequences[0, -1, 0] == 9
    assert sequences[1, 0, 0] == 5
    assert sequences[1, -1, 0] == 14
