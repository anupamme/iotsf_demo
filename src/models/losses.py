"""
Loss functions for Moirai fine-tuning.

This module contains custom loss functions for training, including:
- SupervisedContrastiveLoss: Pushes same-class samples together, different-class apart
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon) from Khosla et al., 2020.

    This loss pulls samples with the same label together in embedding space
    while pushing samples with different labels apart. Unlike standard
    contrastive learning (e.g., SimCLR), this uses label information to
    define positive pairs.

    For IoT anomaly detection:
    - Benign samples (label=0) are pulled together
    - Attack samples (label=1) are pulled together
    - Benign and attack samples are pushed apart

    This creates a clear decision boundary in embedding space, making
    anomaly detection more reliable than relying solely on NLL scores.

    Reference:
        Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
        https://arxiv.org/abs/2004.11362

    Args:
        temperature: Temperature parameter for scaling similarities.
                    Lower values make the loss more sensitive to differences.
                    Default: 0.07 (as recommended in the paper)

    Example:
        >>> loss_fn = SupervisedContrastiveLoss(temperature=0.07)
        >>> embeddings = model.get_embeddings(batch)  # (B, D)
        >>> labels = batch['label']  # (B,)
        >>> loss = loss_fn(embeddings, labels)
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize the loss function.

        Args:
            temperature: Scaling factor for similarity scores.
                        Lower = sharper distinctions but harder to train.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            embeddings: Tensor of shape (B, D) containing normalized embeddings.
                       Will be L2-normalized internally if not already.
            labels: Tensor of shape (B,) containing integer class labels.
                   For binary classification: 0=benign, 1=attack
            mask: Optional tensor of shape (B,) indicating valid samples.
                 If provided, invalid samples (mask=0) are excluded.

        Returns:
            Scalar loss value (mean over all valid anchors)

        Raises:
            ValueError: If embeddings and labels have mismatched batch sizes
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if labels.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: embeddings={batch_size}, labels={labels.shape[0]}"
            )

        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute pairwise similarity matrix: (B, B)
        # similarity[i,j] = dot(embeddings[i], embeddings[j]) / temperature
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs (same label, excluding self)
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)

        # Remove diagonal (self-similarity should not count as positive)
        mask_self = torch.eye(batch_size, dtype=torch.float32, device=device)
        mask_positive = mask_positive - mask_self

        # Create mask for all valid pairs (excluding self)
        mask_all = torch.ones_like(mask_positive) - mask_self

        # Apply optional validity mask
        if mask is not None:
            validity = mask.view(-1, 1).float() * mask.view(1, -1).float()
            mask_positive = mask_positive * validity
            mask_all = mask_all * validity

        # For numerical stability, subtract max from similarities
        # (equivalent to log-sum-exp trick)
        similarity_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        similarity_matrix = similarity_matrix - similarity_max.detach()

        # Compute log-softmax over all valid pairs (denominator)
        exp_similarity = torch.exp(similarity_matrix) * mask_all
        log_sum_exp = torch.log(exp_similarity.sum(dim=1, keepdim=True) + 1e-8)

        # Compute log-probability for each pair
        log_prob = similarity_matrix - log_sum_exp

        # Average log-probability over positive pairs for each anchor
        # mask_positive.sum(1) = number of positive pairs for each anchor
        num_positives = mask_positive.sum(dim=1)

        # Only compute loss for anchors that have at least one positive pair
        valid_anchors = num_positives > 0

        if valid_anchors.sum() == 0:
            # No valid positive pairs - return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Mean log-prob over positives for each valid anchor
        mean_log_prob = (mask_positive * log_prob).sum(dim=1) / num_positives.clamp(min=1)

        # Loss is negative mean log-probability (we want to maximize log-prob)
        loss = -mean_log_prob[valid_anchors].mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for Moirai fine-tuning with NLL and contrastive components.

    Total Loss = NLL_loss + lambda * Contrastive_loss

    The NLL loss teaches the model to forecast IoT traffic patterns.
    The contrastive loss teaches the model to separate benign from attack
    embeddings, making anomaly detection more reliable.

    Args:
        contrastive_weight: Weight (lambda) for the contrastive loss term.
                          Higher values emphasize separation over forecasting.
                          Default: 0.5 (equal weighting)
        temperature: Temperature for contrastive loss. Default: 0.07

    Example:
        >>> combined_loss = CombinedLoss(contrastive_weight=0.5)
        >>> nll_loss = model.compute_nll(batch)
        >>> embeddings = model.get_embeddings(batch)
        >>> labels = batch['label']
        >>> total_loss = combined_loss(nll_loss, embeddings, labels)
    """

    def __init__(
        self,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07
    ):
        """
        Initialize combined loss.

        Args:
            contrastive_weight: Lambda weight for contrastive loss
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)

    def forward(
        self,
        nll_loss: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            nll_loss: Scalar or (B,) tensor of NLL losses
            embeddings: (B, D) tensor of embeddings
            labels: (B,) tensor of labels

        Returns:
            Tuple of (total_loss, nll_component, contrastive_component)
        """
        # Ensure nll_loss is scalar
        if nll_loss.dim() > 0:
            nll_loss = nll_loss.mean()

        # Compute contrastive loss
        cont_loss = self.contrastive_loss(embeddings, labels)

        # Combined loss
        total_loss = nll_loss + self.contrastive_weight * cont_loss

        return total_loss, nll_loss, cont_loss
