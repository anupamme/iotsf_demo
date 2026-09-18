#!/usr/bin/env python3
"""CKA floor: pre-trained Moirai-Small encoder against a randomly re-initialised
encoder of the same architecture.

WHY: app:cka_calibration reported this floor as 0.0000 +- 0.0001 over 5 seeds with
Xavier-uniform init, but no script in the repo performs a random re-initialisation
(no occurrence of "xavier" anywhere) and no run record exists. The floor is worth
having -- it is what makes CKA 0.09-0.17 on Chronos readable as "near the floor"
rather than merely small -- so it is measured here instead of asserted.

Uses extract_representations() from finetune_forecasting.py, so the number is on
exactly the same footing as every other CKA in the paper: forward hook on the
whole encoder, mean-pooled over the sequence.

  CKA_DEV   torch device (default mps)
  CKA_OUT   path to write JSON (required in practice; warns if unset)
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, ".")
from src.data.forecasting_loader import get_forecasting_loader
from src.models.moirai_detector import MoiraiAnomalyDetector, _apply_uni2ts_gradient_patch
from scripts.finetune_forecasting import (
    extract_representations, linear_CKA, _patch_packed_scaler_for_mps,
)

DEV = os.environ.get("CKA_DEV", "mps")
OUT = os.environ.get("CKA_OUT")
INIT = os.environ.get("CKA_INIT", "reset")
assert INIT in ("reset", "xavier"), INIT
LB, H, NWIN = 96, 96, 200
SEEDS = [42, 123, 456, 789, 999]

_apply_uni2ts_gradient_patch()
if DEV == "mps":
    _patch_packed_scaler_for_mps()

loader = get_forecasting_loader("data/forecasting/ETTh2.csv", lookback_window=LB,
                                forecast_horizon=H, features="M")
_, val_df, _ = loader.get_splits()
va = val_df[loader.FEATURE_COLUMNS].values
Xval = np.array([va[i:i + LB + H] for i in range(len(va) - LB - H + 1)])[:NWIN]
Xt = torch.from_numpy(Xval).float()
print(f"val windows: {len(Xval)}")

det = MoiraiAnomalyDetector(model_size="small", context_length=LB, prediction_length=H,
                            target_dim=len(loader.FEATURE_COLUMNS), num_samples=20, device=DEV)
det.initialize()
model = det.model
module = model.module
encoder = module.base_model.model.encoder if hasattr(module, "base_model") else module.encoder

reps_pt = extract_representations(model, Xt, None, device=DEV, max_samples=NWIN)
assert reps_pt.shape[0] == NWIN, f"pretrained reps empty/short: {reps_pt.shape}"
print(f"pretrained reps: {reps_pt.shape}")

pretrained_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}


def randomise(seed):
    """Re-initialise the encoder randomly.

    CKA_INIT=reset (default) calls each submodule's own reset_parameters(), which
    is what "a randomly initialised encoder of the same architecture" means: every
    layer gets its library-default init, including LayerNorm at gamma=1, beta=0.

    CKA_INIT=xavier is the rule the earlier appendix described -- Xavier-uniform
    for weight tensors, zeros for biases -- with one necessary correction:
    normalisation parameters are left alone. Zeroing every 1-D tensor also zeroes
    LayerNorm's gamma, which annihilates the signal at the first norm and collapses
    the encoder to a constant output. CKA is then 0 for a degenerate reason rather
    than because two unrelated encoders share no structure, so such a run measures
    nothing. The rep-variance assertion below is what catches it.
    """
    torch.manual_seed(seed)
    if DEV == "mps":
        torch.mps.manual_seed(seed)
    touched = 0
    with torch.no_grad():
        if INIT == "reset":
            for m in encoder.modules():
                if m is not encoder and hasattr(m, "reset_parameters"):
                    m.reset_parameters(); touched += 1
        else:
            for name, p in encoder.named_parameters():
                if "norm" in name.lower():
                    continue
                if p.dim() >= 2:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    p.zero_()
                touched += 1
    return touched


pt_var = float(np.mean(np.var(reps_pt, axis=0)))
print(f"pretrained rep variance: {pt_var:.4e}")

ckas, sanity = [], []
for seed in SEEDS:
    touched = randomise(seed)
    reps_rand = extract_representations(model, Xt, None, device=DEV, max_samples=NWIN)
    assert reps_rand.shape == reps_pt.shape, f"random reps {reps_rand.shape} != {reps_pt.shape}"
    c = linear_CKA(reps_pt, reps_rand)
    var = float(np.mean(np.var(reps_rand, axis=0)))
    # A collapsed encoder gives CKA ~ 0 for a trivial reason and is not a floor.
    assert var > 1e-6 * pt_var, (
        f"seed {seed}: randomised encoder output is constant (var={var:.3e} vs "
        f"pretrained {pt_var:.3e}); CKA=0 here is degenerate, not a floor")
    ckas.append(c); sanity.append(var)
    print(f"seed {seed}: init={INIT}, {touched} tensors/modules, CKA={c:.6f}, rep var={var:.4e}")
    encoder.load_state_dict(pretrained_state)

# Re-check that restoring the pre-trained state reproduces CKA 1.0, i.e. that the
# comparison is measuring re-initialisation and not hook/order drift.
reps_back = extract_representations(model, Xt, None, device=DEV, max_samples=NWIN)
round_trip = linear_CKA(reps_pt, reps_back)
print(f"round-trip CKA after restore: {round_trip:.6f}")

m, sd = float(np.mean(ckas)), float(np.std(ckas, ddof=1))
print(f"\nrandom-init CKA floor: {m:.6f} +- {sd:.6f} (n={len(ckas)} seeds, ddof=1)")
print(f"max over seeds: {max(ckas):.6f}")

if OUT:
    with open(OUT, "w") as fh:
        json.dump({"device": DEV, "seeds": SEEDS, "n_val_windows": NWIN,
                   "lookback": LB, "horizon": H, "init": INIT,
                   "pretrained_rep_variance": pt_var,
                   "cka_values": ckas, "cka_mean": m, "cka_std_ddof1": sd,
                   "rep_variance_per_seed": sanity, "round_trip_cka": round_trip,
                   "rep_dim": int(reps_pt.shape[1])}, fh, indent=2)
    print(f"wrote {OUT}")
else:
    print("WARNING: CKA_OUT unset -- this leaves no run record.")
