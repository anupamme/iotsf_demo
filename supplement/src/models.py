"""Model loading, uni2ts gradient patch, and representation extraction."""

import numpy as np
import torch
import torch.nn as nn

MODEL_SIZE_MAP = {
    "small": "Salesforce/moirai-1.1-R-small",
    "base": "Salesforce/moirai-1.1-R-base",
    "large": "Salesforce/moirai-1.1-R-large",
}

_UNI2TS_PATCHED = False


def apply_uni2ts_gradient_patch():
    """Patch uni2ts PackedStdScaler to fix in-place operation bug.

    The original _get_loc_scale uses in-place assignment:
        loc[sample_id == 0] = 0
        scale[sample_id == 0] = 1
    This breaks gradient computation. We replace with torch.where().
    """
    global _UNI2TS_PATCHED
    if _UNI2TS_PATCHED:
        return

    from einops import reduce
    from uni2ts.module.packed_scaler import PackedStdScaler
    from uni2ts.common.torch_util import safe_div

    def _get_loc_scale_fixed(self, target, observed_mask, sample_id, variate_id):
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = reduce(
            id_mask
            * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = safe_div(loc, tobs)
        var = reduce(
            id_mask
            * reduce(
                ((target - loc) ** 2) * observed_mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        var = safe_div(var, (tobs - self.correction))
        scale = torch.sqrt(var + self.minimum_scale)

        sample_id_mask = (sample_id == 0).unsqueeze(-1)
        loc = torch.where(sample_id_mask, torch.zeros_like(loc), loc)
        scale = torch.where(sample_id_mask, torch.ones_like(scale), scale)

        return loc, scale

    PackedStdScaler._get_loc_scale = _get_loc_scale_fixed
    _UNI2TS_PATCHED = True


def load_moirai(
    model_size: str = "small",
    context_length: int = 96,
    prediction_length: int = 96,
    target_dim: int = 7,
    num_samples: int = 20,
    device: str = "cpu",
    patch_size: int = 32,
    random_init: bool = False,
):
    """Load a Moirai forecast model and apply gradient patch.

    Returns:
        model: MoiraiForecast instance on device.
    """
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    apply_uni2ts_gradient_patch()

    model_id = MODEL_SIZE_MAP[model_size]

    if random_init:
        import huggingface_hub
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        module = MoiraiModule(config)
    else:
        module = MoiraiModule.from_pretrained(model_id)

    model = MoiraiForecast(
        module=module,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        target_dim=target_dim,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        num_samples=num_samples,
    )
    model = model.to(device)
    return model


def get_encoder(model):
    """Get encoder module, handling both normal and LoRA-wrapped models."""
    module = model.module
    if hasattr(module, "base_model"):
        return module.base_model.model.encoder
    return module.encoder


def extract_representations(
    model,
    data: torch.Tensor,
    batch_size: int = 32,
    device: str = "cpu",
    max_samples: int = 500,
    keep_sequence: bool = False,
) -> np.ndarray:
    """Extract encoder representations for CKA / probing.

    Args:
        model: MoiraiForecast model.
        data: (N, seq_len, D) input tensor.
        keep_sequence: If True return (N, T, D); else mean-pool to (N, D).

    Returns:
        Numpy array of representations.
    """
    model.eval()
    captured = {}

    def hook(module, input, output):
        captured["out"] = output

    encoder = get_encoder(model)
    handle = encoder.register_forward_hook(hook)

    all_reps = []
    n = min(len(data), max_samples)
    data_subset = data[:n]

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = data_subset[i : i + batch_size].to(device)
            b = batch.shape[0]
            seq_len = batch.shape[1]
            past_obs = torch.ones_like(batch, dtype=torch.bool)
            past_pad = torch.zeros(b, seq_len, dtype=torch.bool, device=device)

            try:
                model.forward(
                    past_target=batch,
                    past_observed_target=past_obs,
                    past_is_pad=past_pad,
                    num_samples=2,
                )
            except Exception:
                pass

            if "out" in captured:
                rep = captured["out"]
                if isinstance(rep, tuple):
                    rep = rep[0]
                if keep_sequence:
                    all_reps.append(rep.cpu().numpy())
                else:
                    all_reps.append(rep.mean(dim=1).cpu().numpy())

    handle.remove()

    if all_reps:
        return np.concatenate(all_reps, axis=0)
    return np.zeros((0, 1))
