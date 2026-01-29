"""
Tests for supervised contrastive learning components.

Tests:
- SupervisedContrastiveLoss
- ProjectionHead
- MoiraiSupervisedDataset
- Integration with MoiraiAnomalyDetector.fine_tune_supervised()
"""

import pytest
import torch
import numpy as np

from src.models.losses import SupervisedContrastiveLoss, CombinedLoss
from src.models.projection_head import ProjectionHead, AttentionPooling, ProjectionHeadWithPooling
from src.data.torch_dataset import MoiraiSupervisedDataset


class TestSupervisedContrastiveLoss:
    """Tests for SupervisedContrastiveLoss."""

    def test_loss_initialization(self):
        """Test loss function initializes correctly."""
        loss_fn = SupervisedContrastiveLoss(temperature=0.07)
        assert loss_fn.temperature == 0.07

    def test_loss_output_shape(self):
        """Test loss function returns scalar."""
        loss_fn = SupervisedContrastiveLoss()
        embeddings = torch.randn(8, 128)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        loss = loss_fn(embeddings, labels)

        assert loss.dim() == 0  # Scalar
        assert loss.requires_grad

    def test_loss_same_class_lower(self):
        """Test that same-class samples produce lower loss."""
        loss_fn = SupervisedContrastiveLoss(temperature=0.1)

        # Create embeddings where same-class samples are similar
        embeddings_similar = torch.tensor([
            [1.0, 0.0], [0.9, 0.1],  # Class 0 - similar
            [0.0, 1.0], [0.1, 0.9],  # Class 1 - similar
        ])
        labels = torch.tensor([0, 0, 1, 1])

        # Create embeddings where same-class samples are dissimilar
        embeddings_dissimilar = torch.tensor([
            [1.0, 0.0], [0.0, 1.0],  # Class 0 - dissimilar
            [-1.0, 0.0], [0.0, -1.0],  # Class 1 - dissimilar
        ])

        loss_similar = loss_fn(embeddings_similar, labels)
        loss_dissimilar = loss_fn(embeddings_dissimilar, labels)

        # Similar same-class embeddings should have lower loss
        assert loss_similar < loss_dissimilar

    def test_loss_batch_size_mismatch(self):
        """Test that mismatched batch sizes raise error."""
        loss_fn = SupervisedContrastiveLoss()
        embeddings = torch.randn(8, 128)
        labels = torch.tensor([0, 0, 0, 1])  # Wrong size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            loss_fn(embeddings, labels)

    def test_loss_no_positive_pairs(self):
        """Test handling when no positive pairs exist."""
        loss_fn = SupervisedContrastiveLoss()
        embeddings = torch.randn(4, 128)
        labels = torch.tensor([0, 1, 2, 3])  # All different classes

        loss = loss_fn(embeddings, labels)

        # Should return zero loss when no positive pairs
        assert loss.item() == 0.0

    def test_loss_numerical_stability(self):
        """Test loss is stable with large embeddings."""
        loss_fn = SupervisedContrastiveLoss()
        embeddings = torch.randn(32, 256) * 100  # Large values
        labels = torch.randint(0, 2, (32,))

        loss = loss_fn(embeddings, labels)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestProjectionHead:
    """Tests for ProjectionHead."""

    def test_projection_head_initialization(self):
        """Test projection head initializes correctly."""
        proj = ProjectionHead(input_dim=384, hidden_dim=256, output_dim=128)

        assert proj.input_dim == 384
        assert proj.hidden_dim == 256
        assert proj.output_dim == 128

    def test_projection_head_forward(self):
        """Test forward pass produces correct output shape."""
        proj = ProjectionHead(input_dim=384, hidden_dim=256, output_dim=128)
        x = torch.randn(8, 384)

        output = proj(x)

        assert output.shape == (8, 128)

    def test_projection_head_with_dropout(self):
        """Test projection head with dropout."""
        proj = ProjectionHead(input_dim=384, hidden_dim=256, output_dim=128, dropout=0.1)
        x = torch.randn(8, 384)

        proj.train()
        output = proj(x)

        assert output.shape == (8, 128)

    def test_get_output_dim(self):
        """Test get_output_dim method."""
        proj = ProjectionHead(input_dim=384, hidden_dim=256, output_dim=128)
        assert proj.get_output_dim() == 128


class TestAttentionPooling:
    """Tests for AttentionPooling."""

    def test_attention_pooling_initialization(self):
        """Test attention pooling initializes correctly."""
        pool = AttentionPooling(embed_dim=384)
        assert pool.attention is not None

    def test_attention_pooling_forward(self):
        """Test forward pass produces correct output shape."""
        pool = AttentionPooling(embed_dim=384)
        x = torch.randn(8, 48, 384)  # (batch, seq_len, embed_dim)

        output = pool(x)

        assert output.shape == (8, 384)

    def test_attention_pooling_with_mask(self):
        """Test attention pooling with mask."""
        pool = AttentionPooling(embed_dim=384)
        x = torch.randn(8, 48, 384)
        mask = torch.ones(8, 48, dtype=torch.bool)
        mask[:, 40:] = False  # Mask out last 8 positions

        output = pool(x, mask)

        assert output.shape == (8, 384)


class TestProjectionHeadWithPooling:
    """Tests for ProjectionHeadWithPooling."""

    def test_with_mean_pooling(self):
        """Test with mean pooling (default)."""
        module = ProjectionHeadWithPooling(
            input_dim=384,
            hidden_dim=256,
            output_dim=128,
            use_attention_pooling=False
        )
        x = torch.randn(8, 48, 384)

        output = module(x)

        assert output.shape == (8, 128)

    def test_with_attention_pooling(self):
        """Test with attention pooling."""
        module = ProjectionHeadWithPooling(
            input_dim=384,
            hidden_dim=256,
            output_dim=128,
            use_attention_pooling=True
        )
        x = torch.randn(8, 48, 384)

        output = module(x)

        assert output.shape == (8, 128)


class TestMoiraiSupervisedDataset:
    """Tests for MoiraiSupervisedDataset."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_samples = 20
        seq_length = 128
        n_features = 12

        data = np.random.randn(n_samples, seq_length, n_features).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)  # 10 benign, 10 attack

        return data, labels

    def test_dataset_initialization(self, sample_data):
        """Test dataset initializes correctly."""
        data, labels = sample_data
        dataset = MoiraiSupervisedDataset(data, labels, context_length=96, prediction_length=32)

        assert len(dataset) == 20
        assert dataset.n_features == 12

    def test_dataset_getitem(self, sample_data):
        """Test __getitem__ returns correct structure."""
        data, labels = sample_data
        dataset = MoiraiSupervisedDataset(data, labels, context_length=96, prediction_length=32)

        item = dataset[0]

        assert 'context' in item
        assert 'target' in item
        assert 'label' in item
        assert 'past_is_pad' in item
        assert 'past_observed_target' in item

        assert item['context'].shape == (96, 12)
        assert item['target'].shape == (32, 12)
        assert item['label'].dim() == 0  # Scalar

    def test_dataset_label_values(self, sample_data):
        """Test labels are correctly returned."""
        data, labels = sample_data
        dataset = MoiraiSupervisedDataset(data, labels, context_length=96, prediction_length=32)

        # Check benign samples
        for i in range(10):
            assert dataset[i]['label'].item() == 0

        # Check attack samples
        for i in range(10, 20):
            assert dataset[i]['label'].item() == 1

    def test_dataset_invalid_data_shape(self):
        """Test error on invalid data shape."""
        data_2d = np.random.randn(20, 128).astype(np.float32)
        labels = np.zeros(20, dtype=np.int64)

        with pytest.raises(ValueError, match="must be 3D"):
            MoiraiSupervisedDataset(data_2d, labels)

    def test_dataset_mismatched_lengths(self, sample_data):
        """Test error on mismatched data and labels lengths."""
        data, _ = sample_data
        labels_wrong = np.zeros(10, dtype=np.int64)  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            MoiraiSupervisedDataset(data, labels_wrong)

    def test_dataset_sequence_too_short(self, sample_data):
        """Test error when sequence is too short."""
        data, labels = sample_data

        with pytest.raises(ValueError, match="must be >="):
            MoiraiSupervisedDataset(data, labels, context_length=100, prediction_length=50)

    def test_get_sample_info(self, sample_data):
        """Test get_sample_info returns correct info."""
        data, labels = sample_data
        dataset = MoiraiSupervisedDataset(data, labels, context_length=96, prediction_length=32)

        info = dataset.get_sample_info()

        assert info['n_samples'] == 20
        assert info['n_benign'] == 10
        assert info['n_attack'] == 10
        assert info['context_length'] == 96
        assert info['prediction_length'] == 32
        assert info['n_features'] == 12


class TestCombinedLoss:
    """Tests for CombinedLoss."""

    def test_combined_loss_initialization(self):
        """Test combined loss initializes correctly."""
        loss_fn = CombinedLoss(contrastive_weight=0.5, temperature=0.07)
        assert loss_fn.contrastive_weight == 0.5

    def test_combined_loss_forward(self):
        """Test forward pass returns three components."""
        loss_fn = CombinedLoss(contrastive_weight=0.5)

        nll_loss = torch.tensor(2.5, requires_grad=True)
        embeddings = torch.randn(8, 128)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        total, nll, cont = loss_fn(nll_loss, embeddings, labels)

        assert total.dim() == 0
        assert nll.dim() == 0
        assert cont.dim() == 0

        # Total should be approximately nll + 0.5 * cont
        expected = nll + 0.5 * cont
        assert torch.allclose(total, expected)


class TestIntegration:
    """Integration tests for supervised contrastive learning pipeline."""

    def test_full_training_step(self):
        """Test a full training step with all components."""
        # Setup
        batch_size = 8
        d_model = 384
        context_length = 96
        prediction_length = 32
        n_features = 12

        projection_head = ProjectionHead(input_dim=d_model, hidden_dim=256, output_dim=128)
        contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.07)

        # Simulate batch
        context = torch.randn(batch_size, context_length, n_features)
        target = torch.randn(batch_size, prediction_length, n_features)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        # Simulate encoder output (what Moirai encoder would produce)
        encoder_output = torch.randn(batch_size, 48, d_model)

        # Mean pool embeddings
        embeddings = encoder_output.mean(dim=1)

        # Project embeddings
        projected = projection_head(embeddings)

        # Compute contrastive loss
        cont_loss = contrastive_loss_fn(projected, labels)

        # Simulate NLL loss
        nll_loss = torch.tensor(2.5, requires_grad=True)

        # Combined loss
        total_loss = nll_loss + 0.5 * cont_loss

        # Backward pass
        total_loss.backward()

        # Verify gradients flow
        assert projection_head.net[0].weight.grad is not None

    def test_dataloader_integration(self):
        """Test DataLoader integration with supervised dataset."""
        from torch.utils.data import DataLoader

        # Create sample data
        n_samples = 32
        data = np.random.randn(n_samples, 128, 12).astype(np.float32)
        labels = np.array([0] * 16 + [1] * 16, dtype=np.int64)

        dataset = MoiraiSupervisedDataset(data, labels, context_length=96, prediction_length=32)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        for batch in dataloader:
            assert batch['context'].shape == (8, 96, 12)
            assert batch['target'].shape == (8, 32, 12)
            assert batch['label'].shape == (8,)
            break  # Just test first batch
