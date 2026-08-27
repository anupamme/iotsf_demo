"""Diagnostic metrics: value gate, forgetting, weight drift."""

import numpy as np
import torch


def compute_r2_task(zs_mse: float, linear_mse: float) -> float:
    """Value-gate metric: R^2_task = 1 - MSE_ZS / MSE_Linear.

    Positive means pre-trained model beats linear baseline (gate pass).
    """
    if linear_mse < 1e-12:
        return 0.0
    return 1.0 - zs_mse / linear_mse


def compute_forgetting_pct(zeroshot_mse: float, final_mse: float) -> float:
    """Catastrophic forgetting percentage.

    Positive = degradation after fine-tuning.
    Negative = improvement (beneficial restructuring).
    """
    if abs(zeroshot_mse) < 1e-12:
        return 0.0
    return 100.0 * (final_mse - zeroshot_mse) / zeroshot_mse


def compute_weight_drift(model: torch.nn.Module, pretrained_params: dict) -> float:
    """L2 distance between current and pre-trained weights."""
    total_drift = 0.0
    for name, param in model.named_parameters():
        if name in pretrained_params:
            diff = (param.data - pretrained_params[name]).float()
            total_drift += diff.norm().item() ** 2
    return float(np.sqrt(total_drift))
