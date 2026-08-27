#!/usr/bin/env python3
"""
Shared TimesFM 2.5 machinery: loading, a DIFFERENTIABLE native forecast, and the scoring
conventions the Chronos arm already uses.

WHY THIS EXISTS
---------------
The released torch API cannot be trained through. `TimesFM_2p5_200M_torch.forecast()` calls
`compiled_decode`, which calls `module.decode()`, which is wrapped in `with torch.no_grad()` and
ends with `.detach().cpu().numpy()`. So fine-tuning TimesFM requires re-implementing the forward
path. Re-implementing an inference stack is exactly where a silent protocol mismatch gets born, so
this module does it once, and `verify_native_path()` asserts the re-implementation reproduces
`model.forecast()` on real windows before anything is trained.

WHAT THE PATH IS (from timesfm_2p5_torch.py: TimesFM_2p5_200M_torch.compile / module.decode)
  outer   mu, sigma over the whole context; inputs <- revin(inputs, mu, sigma)
  prefill patch to (B, nP, 32); per-patch RUNNING stats -> context_mu/sigma (cumulative, not
          per-patch); normed <- revin(patched, context_mu, context_sigma); module.forward();
          renorm the point-head output by context_mu/sigma; take the LAST patch's o=128 outputs
  flip    force_flip_invariance: rerun on -inputs, reverse quantile channels 1..9, average
          (pf - pf_flipped) / 2
  slice   [:, :horizon, :]
  outer   revin(..., mu, sigma, reverse=True); point forecast = quantile channel 5

The autoregressive loop in `decode` runs `num_decode_steps = (max_horizon - 1) // o` times. With
o = 128 and max_horizon = 128 that is ZERO, so at h = 24 a single prefill forward IS the whole
forecast -- which is why this arm is trainable in one differentiable pass at all. `assert_no_ar()`
enforces that precondition rather than trusting it.

Channel layout (timesfm_2p5_base.py:92): q = len(quantiles) + 1 = 10, channel 0 is the mean and
channels 1..9 are the deciles 0.1..0.9, so channel 5 is the median -- the point forecast.
`use_continuous_quantile_head` and `fix_quantile_crossing` rewrite channels 1-4 and 6-9 ONLY, so
neither touches the point forecast and neither is needed on the training path.
"""
import glob
import os

import numpy as np
import torch

import timesfm
from timesfm.timesfm_2p5.timesfm_2p5_torch import revin
from timesfm.torch import util

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
POINT_CHANNEL = 5          # median; what forecast() returns as the point forecast
MEAN_CHANNEL = 0

# The stack is the "encoder" for the purposes of conditions D and H. Named here so the freeze
# logic and the parameter counts in the paper come from one place.
STACK_ATTR = "stacked_xf"          # 20 transformer layers, 196.71M
TOKENIZER_ATTR = "tokenizer"       # input residual block, 1.81M
HEAD_ATTRS = ("output_projection_point", "output_projection_quantiles")   # 4.92M + 27.85M


def load_timesfm(max_context: int, max_horizon: int, device: str = "cpu"):
    """Load the released checkpoint and compile it, preferring the local snapshot.

    from_pretrained() hits the hub for a config even when the weights are cached, which breaks
    under HF_HUB_OFFLINE; scripts/timesfm_etth2.py sidesteps it the same way.
    """
    snaps = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--timesfm-2.5-200m-pytorch/snapshots/*")))
    src = snaps[-1] if snaps else MODEL_ID
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(src)
    fc = timesfm.ForecastConfig(
        max_context=max_context,
        max_horizon=max_horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    )
    model.compile(fc)
    model.model.to(device)
    # The module hardcodes self.device to cuda:0 or cpu at construction (timesfm_2p5_torch.py:73-77)
    # and compiled_decode moves its inputs to it, so .to(device) alone leaves forecast() sending CPU
    # tensors into MPS weights. Keep the attribute in step with where the weights actually are.
    model.model.device = torch.device(device)
    return model, model.forecast_config


def assert_no_ar(model):
    """The differentiable path is prefill-only. Refuse to run if decode would recurse."""
    m, fc = model.model, model.forecast_config
    steps = (fc.max_horizon - 1) // m.o
    if steps != 0:
        raise ValueError(
            f"prefill-only path requires num_decode_steps == 0, got {steps} "
            f"(max_horizon={fc.max_horizon}, o={m.o}). Use max_horizon <= {m.o}.")
    if fc.max_context % m.p != 0:
        raise ValueError(f"max_context {fc.max_context} is not a multiple of the patch size {m.p}")


def _flip_quantiles(x):
    """compile()'s flip_quantile_fn: keep the mean channel, reverse the deciles."""
    return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=(-1,))], dim=-1)


def _prefill(module, normed_context, masks):
    """module.decode()'s prefill, differentiable. Returns (pf_outputs, output_embeddings).

    pf_outputs: (B, nP, o, q) renormalised point-head output.
    output_embeddings: (B, nP, d) final-layer hidden states -- what CKA is computed on.
    """
    B = normed_context.shape[0]
    patched = torch.reshape(normed_context, (B, -1, module.p))
    patched_masks = torch.reshape(masks, (B, -1, module.p))

    n = torch.zeros(B, device=normed_context.device)
    mu = torch.zeros(B, device=normed_context.device)
    sigma = torch.zeros(B, device=normed_context.device)
    patch_mu, patch_sigma = [], []
    for i in range(patched.shape[1]):
        (n, mu, sigma), _ = util.update_running_stats(
            n, mu, sigma, patched[:, i], patched_masks[:, i])
        patch_mu.append(mu)
        patch_sigma.append(sigma)
    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)

    normed_inputs = revin(patched, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, torch.zeros_like(normed_inputs), normed_inputs)
    (_, output_embeddings, normed_outputs, _), _ = module(normed_inputs, patched_masks, None)
    pf = torch.reshape(revin(normed_outputs, context_mu, context_sigma, reverse=True),
                       (B, -1, module.o, module.q))
    return pf, output_embeddings


def native_forecast(model, context, horizon, return_reps=False):
    """Differentiable equivalent of model.forecast()'s full_forecast, minus the quantile-only
    post-processing (which cannot reach the point channel).

    context: (B, max_context) float32 tensor, finite, on the model's device.
    Returns (B, horizon, q); [..., POINT_CHANNEL] is the point forecast.
    """
    module, fc = model.model, model.forecast_config
    if context.shape[1] != fc.max_context:
        raise ValueError(f"context length {context.shape[1]} != max_context {fc.max_context}; "
                         "pad or trim before calling (this path does not re-pad).")
    masks = torch.zeros_like(context, dtype=torch.bool)

    mu = torch.mean(context, dim=-1, keepdim=True)
    sigma = torch.std(context, dim=-1, keepdim=True)
    normed = revin(context, mu, sigma, reverse=False)

    pf, reps = _prefill(module, normed, masks)
    ff = pf[:, -1, ...]
    if fc.force_flip_invariance:
        pf_f, _ = _prefill(module, -normed, masks)
        ff = (ff - _flip_quantiles(pf_f)[:, -1, ...]) / 2
    ff = ff[:, :horizon, :]
    ff = revin(ff, mu, sigma, reverse=True)
    if return_reps:
        return ff, reps
    return ff


def native_point(model, context, horizon):
    return native_forecast(model, context, horizon)[..., POINT_CHANNEL]


# ---------------------------------------------------------------------------
# Loss: TimesFM's own output head, no head attached
# ---------------------------------------------------------------------------

def native_loss(forecast, target):
    """TimesFM's own point-head output structure: MSE on the mean channel plus pinball loss on
    the nine decile channels. No head is attached to the backbone -- this is the loss the released
    output_projection_point is shaped for.

    The continuous quantile head (output_projection_quantiles, 27.85M) only rewrites channels
    1-4/6-9 at inference and receives no gradient here; the appendix states this.

    forecast: (B, H, q); target: (B, H).
    """
    tgt = target.unsqueeze(-1)
    mse = torch.mean((forecast[..., MEAN_CHANNEL] - target) ** 2)
    err = tgt - forecast[..., 1:]
    lv = torch.tensor(QUANTILE_LEVELS, device=forecast.device, dtype=forecast.dtype)
    pinball = torch.mean(torch.maximum(lv * err, (lv - 1.0) * err))
    return mse + pinball


# ---------------------------------------------------------------------------
# Scoring: identical convention to the Chronos arm (chronos_mse_finetune.chronos_zs_mse)
# ---------------------------------------------------------------------------

def window_norm_mse(preds, contexts, targets):
    """Per-window z-score by the CONTEXT's own mean/std, then MSE. Matches the Chronos arm so the
    TimesFM cells are directly comparable to the five Chronos cells and to the gate's linear
    baseline in gate_all_cells.chronos_gates().
    """
    out = []
    for i in range(len(contexts)):
        mu = contexts[i].mean()
        sd = contexts[i].std() + 1e-8
        out.append(float(np.mean(((preds[i] - mu) / sd - (targets[i] - mu) / sd) ** 2)))
    return float(np.mean(out))


def batched_point_mse(model, contexts, targets, horizon, device, batch_size=16, grad=False):
    """Score window_norm_mse over a window set through the differentiable native path."""
    preds = []
    ctx = np.asarray(contexts, dtype=np.float32)
    was_training = model.model.training
    model.model.eval()
    with torch.set_grad_enabled(grad):
        for s in range(0, len(ctx), batch_size):
            b = torch.from_numpy(ctx[s:s + batch_size]).to(device)
            preds.append(native_point(model, b, horizon).detach().cpu().numpy())
    if was_training:
        model.model.train()
    return window_norm_mse(np.concatenate(preds), contexts, targets)


def pooled_reps(model, contexts, horizon, device, batch_size=16):
    """Mean-pooled final-layer hidden states over patches, for linear_CKA. Same recipe as the
    Moirai arm (mean pooling over the token axis), computed on a fixed held-out window set.
    """
    reps = []
    ctx = np.asarray(contexts, dtype=np.float32)
    model.model.eval()
    with torch.no_grad():
        for s in range(0, len(ctx), batch_size):
            b = torch.from_numpy(ctx[s:s + batch_size]).to(device)
            _, r = native_forecast(model, b, horizon, return_reps=True)
            reps.append(r.mean(dim=1).cpu().numpy())
    return np.concatenate(reps)


# ---------------------------------------------------------------------------
# Verification -- run before training, not after
# ---------------------------------------------------------------------------

def verify_native_path(model, contexts, horizon, device, tol=1e-4, n=8):
    """Assert the differentiable path reproduces model.forecast() on real windows."""
    ctx = np.asarray(contexts[:n], dtype=np.float32)
    ref, _ = model.forecast(horizon=horizon, inputs=[c for c in ctx])
    ref = np.asarray(ref)[:, :horizon]
    with torch.no_grad():
        got = native_point(model, torch.from_numpy(ctx).to(device), horizon).cpu().numpy()
    dev = float(np.max(np.abs(ref - got)))
    scale = float(np.mean(np.abs(ref))) + 1e-12
    print(f"  native-path check: max |forecast - differentiable| = {dev:.3e} "
          f"(relative {dev / scale:.3e}) over {len(ctx)} windows x h={horizon}")
    if dev / scale > tol:
        raise AssertionError(
            f"differentiable path deviates from model.forecast() by {dev:.3e} "
            f"(relative {dev / scale:.3e} > {tol}); do not train through it")
    return dev


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chronos_mse_finetune import build_windows, load_series

    HZ, LB = 24, 96
    dev = "cpu"
    model, fc = load_timesfm(max_context=LB, max_horizon=128, device=dev)
    assert_no_ar(model)
    _, _, test_s = load_series("ETTh1")
    c, t = build_windows(test_s, LB, HZ, max_windows=16, seed=0)
    verify_native_path(model, c, HZ, dev)
    print("  point MSE (16 windows):", batched_point_mse(model, c, t, HZ, dev))
    print("  reps:", pooled_reps(model, c[:8], HZ, dev).shape)
