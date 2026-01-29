"""
Projection head for contrastive learning.

Maps Moirai encoder embeddings to a lower-dimensional space optimized
for contrastive learning. Following SimCLR and SupCon conventions,
the projection head is an MLP that is discarded after training.
"""

import torch
import torch.nn as nn
from typing import Optional


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.

    Projects high-dimensional encoder embeddings to a lower-dimensional
    space where contrastive loss is applied. This follows the architecture
    from SimCLR (Chen et al., 2020) and is used in supervised contrastive
    learning as well.

    The projection head is typically discarded after training - only the
    encoder embeddings are used for downstream tasks (anomaly detection).

    Architecture:
        Input (d_model) -> Linear -> ReLU -> Linear -> Output (output_dim)

    For Moirai small model:
        384 -> 256 -> ReLU -> 128

    Args:
        input_dim: Dimension of input embeddings (d_model from Moirai).
                  Default: 384 (Moirai small)
        hidden_dim: Dimension of hidden layer. Default: 256
        output_dim: Dimension of output embeddings. Default: 128

    Example:
        >>> proj_head = ProjectionHead(input_dim=384)
        >>> encoder_embeddings = moirai.get_embeddings(batch)  # (B, 384)
        >>> projected = proj_head(encoder_embeddings)  # (B, 128)
        >>> loss = contrastive_loss(projected, labels)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.0
    ):
        """
        Initialize projection head.

        Args:
            input_dim: Input embedding dimension (from Moirai encoder)
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension for contrastive loss
            dropout: Dropout probability (default: 0.0, no dropout)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to contrastive space.

        Args:
            x: Input embeddings of shape (B, input_dim)

        Returns:
            Projected embeddings of shape (B, output_dim)
        """
        return self.net(x)

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence embeddings.

    Instead of simple mean/max pooling over sequence tokens,
    this module learns to weight tokens based on their importance.

    This can be used as an alternative to mean pooling when
    converting Moirai encoder output (B, seq_tokens, d_model)
    to a single embedding per sample (B, d_model).

    Args:
        embed_dim: Embedding dimension (d_model)

    Example:
        >>> pool = AttentionPooling(embed_dim=384)
        >>> encoder_output = moirai.encoder(x)  # (B, 48, 384)
        >>> pooled = pool(encoder_output)  # (B, 384)
    """

    def __init__(self, embed_dim: int):
        """
        Initialize attention pooling.

        Args:
            embed_dim: Embedding dimension
        """
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply attention pooling.

        Args:
            x: Sequence embeddings of shape (B, seq_len, embed_dim)
            mask: Optional boolean mask of shape (B, seq_len).
                 True = valid token, False = padding to ignore.

        Returns:
            Pooled embeddings of shape (B, embed_dim)
        """
        # Compute attention scores: (B, seq_len, 1)
        scores = self.attention(x)

        # Apply mask if provided
        if mask is not None:
            # mask: (B, seq_len) -> (B, seq_len, 1)
            mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax over sequence dimension
        weights = torch.softmax(scores, dim=1)  # (B, seq_len, 1)

        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # (B, embed_dim)

        return pooled


class ProjectionHeadWithPooling(nn.Module):
    """
    Combined attention pooling and projection head.

    Takes raw encoder output (B, seq_tokens, d_model), applies
    attention pooling, then projects to contrastive space.

    Args:
        input_dim: Encoder embedding dimension (d_model)
        hidden_dim: Projection head hidden dimension
        output_dim: Final output dimension for contrastive loss
        use_attention_pooling: If True, use attention pooling.
                              If False, use mean pooling.

    Example:
        >>> module = ProjectionHeadWithPooling(input_dim=384)
        >>> encoder_output = moirai.encoder(x)  # (B, 48, 384)
        >>> projected = module(encoder_output)  # (B, 128)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        use_attention_pooling: bool = False
    ):
        """
        Initialize module.

        Args:
            input_dim: Encoder embedding dimension
            hidden_dim: Projection hidden dimension
            output_dim: Output dimension for contrastive loss
            use_attention_pooling: Whether to use attention pooling
        """
        super().__init__()

        self.use_attention_pooling = use_attention_pooling

        if use_attention_pooling:
            self.pooling = AttentionPooling(input_dim)
        else:
            self.pooling = None  # Will use mean pooling

        self.projection = ProjectionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool and project encoder output.

        Args:
            x: Encoder output of shape (B, seq_len, d_model)
            mask: Optional mask for attention pooling

        Returns:
            Projected embeddings of shape (B, output_dim)
        """
        # Pool sequence to single vector
        if self.use_attention_pooling:
            pooled = self.pooling(x, mask)
        else:
            # Mean pooling
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)

        # Project to contrastive space
        projected = self.projection(pooled)

        return projected
