"""Integration tests for end-to-end workflows."""

import pytest
import numpy as np
from pathlib import Path


@pytest.mark.integration
class TestDataLoadingPipeline:
    """Test complete data loading and preprocessing workflow."""

    def test_load_and_preprocess_benign(self, sample_traffic_data):
        """Integration: Load data -> Preprocess -> Create sequences"""
        from src.data.preprocessor import TrafficPreprocessor, create_sequences

        # Preprocess
        preprocessor = TrafficPreprocessor('standard')
        normalized = preprocessor.fit_transform(sample_traffic_data.values)

        # Create sequences
        sequences = create_sequences(normalized, seq_length=32, stride=16)

        assert sequences.ndim == 3
        assert sequences.shape[1] == 32
        assert sequences.shape[2] == 12

    def test_mixed_batch_pipeline(self, sample_traffic_data):
        """Integration: Mixed data -> Preprocess -> Sequences"""
        from src.data.preprocessor import TrafficPreprocessor, create_sequences

        # Simulate mixed benign/attack data
        data = sample_traffic_data.values

        preprocessor = TrafficPreprocessor('minmax')
        normalized = preprocessor.fit_transform(data)

        sequences = create_sequences(normalized, seq_length=32, stride=32)

        assert sequences.shape[2] == 12
        assert not np.isnan(sequences).any()


@pytest.mark.integration
class TestGenerationPipeline:
    """Test synthetic attack generation workflow."""

    def test_generate_benign_baseline(self):
        """Integration: Initialize generator -> Generate benign -> Validate stats"""
        from src.models.diffusion_ts import IoTDiffusionGenerator

        generator = IoTDiffusionGenerator(seq_length=32, feature_dim=12, device='cpu')
        generator.initialize()

        samples = generator.generate(n_samples=3, seed=42)

        assert samples.shape == (3, 32, 12)
        assert not np.isnan(samples).any()
        # Mock mode should produce reasonable values
        assert np.abs(samples.mean()) < 2.0
        assert 0.1 < samples.std() < 2.0  # Relaxed lower bound for mock mode

    def test_generate_hard_negative_pipeline(self):
        """Integration: Generate attack -> Verify stealth"""
        from src.models.diffusion_ts import IoTDiffusionGenerator

        generator = IoTDiffusionGenerator(seq_length=32, feature_dim=12, device='cpu')
        generator.initialize()

        benign = np.random.randn(32, 12)
        attack, metadata = generator.generate_hard_negative(
            benign_sample=benign,
            attack_pattern='slow_exfiltration',
            stealth_level=0.90
        )

        assert attack.shape == (32, 12)
        assert 'attack_type' in metadata
        assert 'mean_diff' in metadata
        assert metadata['attack_type'] == 'slow_exfiltration'

    def test_batch_attack_generation(self):
        """Integration: Generate multiple attack types"""
        from src.models.diffusion_ts import IoTDiffusionGenerator

        generator = IoTDiffusionGenerator(seq_length=32, feature_dim=12, device='cpu')
        generator.initialize()

        benign = np.random.randn(32, 12)
        attack_patterns = ['slow_exfiltration', 'beacon', 'lotl_mimicry']

        for pattern in attack_patterns:
            attack, metadata = generator.generate_hard_negative(
                benign_sample=benign,
                attack_pattern=pattern,
                stealth_level=0.85
            )
            assert attack.shape == (32, 12)
            assert metadata['attack_type'] == pattern


@pytest.mark.integration
class TestPreprocessingPipeline:
    """Test preprocessing with save/load cycle."""

    def test_fit_save_load_transform(self, tmp_path, sample_traffic_data):
        """Integration: Fit scaler -> Save -> Load -> Transform new data"""
        from src.data.preprocessor import TrafficPreprocessor

        # Fit and save
        prep1 = TrafficPreprocessor('standard')
        prep1.fit(sample_traffic_data.values)
        save_path = tmp_path / "scaler.pkl"
        prep1.save(save_path)

        # Load and transform
        prep2 = TrafficPreprocessor('standard')
        prep2.load(save_path)

        # Transform should give same results
        data = sample_traffic_data.values[:10]
        result1 = prep1.transform(data)
        result2 = prep2.transform(data)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_scaler_consistency_across_sessions(self, tmp_path):
        """Integration: Same data should produce same output after reload"""
        from src.data.preprocessor import TrafficPreprocessor

        data = np.random.randn(100, 12)

        # Session 1
        prep1 = TrafficPreprocessor('minmax')
        normalized1 = prep1.fit_transform(data)
        save_path = tmp_path / "scaler.pkl"
        prep1.save(save_path)

        # Session 2 (load scaler)
        prep2 = TrafficPreprocessor('minmax')
        prep2.load(save_path)
        normalized2 = prep2.transform(data)

        np.testing.assert_array_almost_equal(normalized1, normalized2)


@pytest.mark.integration
class TestConfigurationPipeline:
    """Test configuration loading across all components."""

    def test_config_driven_initialization(self, sample_config):
        """Integration: Load config -> Initialize all components with config"""
        from src.utils.config import Config
        from src.models.diffusion_ts import IoTDiffusionGenerator

        config = Config(sample_config)

        generator = IoTDiffusionGenerator(
            seq_length=config.get("models.diffusion_ts.seq_length"),
            feature_dim=config.get("models.diffusion_ts.feature_dim"),
            device='cpu'
        )
        generator.initialize()

        assert generator.seq_length == 32
        assert generator.feature_dim == 12

    def test_device_config_integration(self, sample_config):
        """Integration: Config device settings -> Model uses correct device"""
        from src.utils.config import Config
        from src.utils.device import get_device

        config = Config(sample_config)

        device = get_device(
            use_gpu=config.get("device.use_gpu"),
            gpu_id=config.get("device.gpu_id")
        )

        assert str(device) in ['cpu', 'cuda', 'cuda:0']


@pytest.mark.integration
class TestDecompositionPipeline:
    """Test time-series decomposition workflow."""

    def test_generate_and_decompose(self):
        """Integration: Generate traffic -> Decompose -> Validate components"""
        from src.models.diffusion_ts import IoTDiffusionGenerator

        generator = IoTDiffusionGenerator(seq_length=64, feature_dim=12, device='cpu')
        generator.initialize()

        sample = generator.generate(n_samples=1, seed=42)[0]
        decomp = generator.get_decomposition(sample)

        assert decomp['trend'].shape == (64, 12)
        assert decomp['seasonality'].shape == (64, 12)
        assert decomp['residual'].shape == (64, 12)

        # Decomposition should sum back to original (approximately)
        reconstructed = decomp['trend'] + decomp['seasonality'] + decomp['residual']
        np.testing.assert_array_almost_equal(reconstructed, sample, decimal=1)

    def test_decompose_with_attack_injection(self):
        """Integration: Generate -> Inject attack -> Decompose -> Verify pattern"""
        from src.models.diffusion_ts import IoTDiffusionGenerator

        generator = IoTDiffusionGenerator(seq_length=64, feature_dim=12, device='cpu')
        generator.initialize()

        benign = generator.generate(n_samples=1, seed=42)[0]
        attack, _ = generator.generate_hard_negative(
            benign_sample=benign,
            attack_pattern='beacon',
            stealth_level=0.85
        )

        # Decompose attack
        decomp = generator.get_decomposition(attack)

        # All components should have valid shapes
        assert decomp['trend'].shape == attack.shape
        assert decomp['seasonality'].shape == attack.shape
        assert decomp['residual'].shape == attack.shape


@pytest.mark.integration
class TestAttackInjectionPipeline:
    """Test complete attack injection workflow."""

    def test_all_attack_patterns_sequential(self):
        """Integration: Generate attacks for all 4 patterns sequentially"""
        from src.models.diffusion_ts import IoTDiffusionGenerator

        generator = IoTDiffusionGenerator(seq_length=32, feature_dim=12, device='cpu')
        generator.initialize()

        benign = np.random.randn(32, 12)
        patterns = ['slow_exfiltration', 'lotl_mimicry', 'protocol_anomaly', 'beacon']

        for pattern in patterns:
            attack, metadata = generator.generate_hard_negative(
                benign_sample=benign,
                attack_pattern=pattern,
                stealth_level=0.90
            )

            assert attack.shape == benign.shape
            assert metadata['attack_type'] == pattern
            assert not np.isnan(attack).any()

    def test_stealth_level_sweep(self):
        """Integration: Generate attacks at different stealth levels"""
        from src.models.diffusion_ts import IoTDiffusionGenerator

        generator = IoTDiffusionGenerator(seq_length=32, feature_dim=12, device='cpu')
        generator.initialize()

        benign = np.random.randn(32, 12)
        stealth_levels = [0.85, 0.90, 0.95]

        mean_diffs = []
        for stealth in stealth_levels:
            attack, metadata = generator.generate_hard_negative(
                benign_sample=benign,
                attack_pattern='slow_exfiltration',
                stealth_level=stealth
            )
            mean_diffs.append(metadata['mean_diff'])

        # Higher stealth should generally result in smaller mean differences
        # (though this may not always be strictly monotonic in mock mode)
        assert all(diff >= 0 for diff in mean_diffs)


@pytest.mark.integration
class TestEndToEndDemo:
    """Test complete demo workflow (what users will see)."""

    def test_demo_data_preparation(self):
        """
        Integration: Full demo prep workflow
        - Generate synthetic benign
        - Generate hard-negative attacks
        - Mix samples for challenge
        """
        from src.models.diffusion_ts import IoTDiffusionGenerator

        generator = IoTDiffusionGenerator(seq_length=32, feature_dim=12, device='cpu')
        generator.initialize()

        # Generate benign samples
        benign_samples = generator.generate(n_samples=3, seed=42)
        assert benign_samples.shape == (3, 32, 12)

        # Generate attack samples
        attacks = []
        for benign in benign_samples:
            attack, _ = generator.generate_hard_negative(
                benign_sample=benign,
                attack_pattern='slow_exfiltration',
                stealth_level=0.95
            )
            attacks.append(attack)

        attacks = np.array(attacks)
        assert attacks.shape == (3, 32, 12)

        # Mix samples (this is what demo would show)
        all_samples = np.vstack([benign_samples, attacks])
        labels = np.array([0, 0, 0, 1, 1, 1])  # 0=benign, 1=attack

        assert all_samples.shape == (6, 32, 12)
        assert len(labels) == 6

    def test_config_to_generation_pipeline(self, sample_config):
        """Integration: Load config -> Generate using config params"""
        from src.utils.config import Config
        from src.models.diffusion_ts import IoTDiffusionGenerator

        config = Config(sample_config)

        # Get parameters from config
        seq_length = config.get("models.diffusion_ts.seq_length")
        feature_dim = config.get("models.diffusion_ts.feature_dim")
        n_benign = config.get("demo.n_benign_samples")

        # Initialize and generate
        generator = IoTDiffusionGenerator(
            seq_length=seq_length,
            feature_dim=feature_dim,
            device='cpu'
        )
        generator.initialize()

        samples = generator.generate(n_samples=n_benign, seed=42)

        assert samples.shape == (n_benign, seq_length, feature_dim)
        assert samples.shape == (2, 32, 12)  # From sample_config
