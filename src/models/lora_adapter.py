"""LoRA adapter for Moirai encoder via HuggingFace PEFT."""

import torch
from loguru import logger

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("peft not installed. LoRA fine-tuning unavailable.")


def apply_lora(model, rank: int = 8, alpha: int = 16, target_modules=None) -> torch.nn.Module:
    """Wrap a MoiraiForecast model's encoder with LoRA adapters.

    Defaults target q_proj, v_proj, and out_proj in all encoder self-attention
    layers.  Pass a list of strings to override for LoRA-HP ablation (reviewer Q4).

    Args:
        model: MoiraiForecast instance (model.module is the MoiraiModule)
        rank: LoRA rank (number of low-rank dimensions)
        alpha: LoRA alpha scaling factor
        target_modules: optional list of attention module names (default q_proj, v_proj, out_proj)
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")

    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "out_proj"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )

    model.module = get_peft_model(model.module, lora_config)

    n_trainable = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.module.parameters())
    logger.info(
        f"LoRA applied: {n_trainable:,} trainable / {n_total:,} total params "
        f"({100 * n_trainable / n_total:.1f}%), rank={rank}, alpha={alpha}"
    )

    return model
