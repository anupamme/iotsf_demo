#!/usr/bin/env python3
"""Per-layer CKA from converged early-stopped checkpoints (no retraining).

Computes per-layer CKA BOTH mean-pooled and token-flattened, since the two
scales differ and the paper must name which one it reports.

Configure with the environment:
  CKA_DEV    torch device (default cuda; use mps on Apple silicon)
  CKA_CKPTS  glob for the fine-tuned encoder checkpoints
  CKA_OUT    path to write the per-layer results as JSON

CKA_OUT matters: earlier invocations of this script printed to stdout only, so
the per-layer numbers that reached the paper had no surviving run record and
could not be re-checked. Always pass it.
"""
import sys, os, glob, re, json, numpy as np, torch
sys.path.insert(0, ".")
from src.data.forecasting_loader import get_forecasting_loader
from src.models.moirai_detector import MoiraiAnomalyDetector, _apply_uni2ts_gradient_patch

DEV, LB, H = os.environ.get("CKA_DEV", "cuda"), 96, 96
CKPT_GLOB = os.environ.get("CKA_CKPTS", "results/v19_cuda_etth2_n10k/seed*/best_encoder.pt")
CKPTS = sorted(glob.glob(CKPT_GLOB))
assert CKPTS, f"no checkpoints matched {CKPT_GLOB}"

def linear_CKA(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    XtX = X.T @ X; YtY = Y.T @ Y
    num = np.trace(XtX @ YtY); den = np.sqrt(np.trace(XtX @ XtX) * np.trace(YtY @ YtY))
    return float(num / den) if den > 1e-10 else 0.0

torch.manual_seed(0); np.random.seed(0)
_apply_uni2ts_gradient_patch()
if DEV == "mps":
    # Same patch finetune_forecasting.py applies: PackedStdScaler casts to
    # float64, which MPS cannot represent, so every forward pass raises and the
    # representation buffers come back empty.
    from scripts.finetune_forecasting import _patch_packed_scaler_for_mps
    _patch_packed_scaler_for_mps()

loader = get_forecasting_loader("data/forecasting/ETTh2.csv", lookback_window=LB,
                                forecast_horizon=H, features="M")
_, val_df, _ = loader.get_splits()
cols = loader.FEATURE_COLUMNS
va = val_df[cols].values
Xval = np.array([va[i:i+LB+H] for i in range(len(va) - LB - H + 1)])[:200]
print(f"val_reps={len(Xval)}")

det = MoiraiAnomalyDetector(model_size="small", context_length=LB, prediction_length=H,
                            target_dim=len(cols), num_samples=20, device=DEV)
det.initialize()
model = det.model
enc = model.module.encoder if not hasattr(model.module, "base_model") else model.module.base_model.model.encoder
layers = enc.layers
nL = len(layers)
print(f"encoder layers: {nL}")

def per_layer_reps(Xnp):
    """Return {layer: (pooled [N,D], flat [N*P,D])}."""
    model.eval()
    data = torch.from_numpy(Xnp).float()
    pooled = {i: [] for i in range(nL)}; flat = {i: [] for i in range(nL)}
    for j in range(0, len(data), 32):
        b = data[j:j+32].to(DEV); bb, sl = b.shape[0], b.shape[1]
        po = torch.ones_like(b, dtype=torch.bool)
        pp = torch.zeros(bb, sl, dtype=torch.bool, device=DEV)
        cap = {}
        def mk(i):
            def h(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                cap[i] = o.detach()
            return h
        hs = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
        with torch.no_grad():
            try:
                model.forward(past_target=b, past_observed_target=po, past_is_pad=pp, num_samples=2)
            except Exception as e:
                print("fwd warn:", str(e)[:70])
        for hh in hs: hh.remove()
        for i in range(nL):
            if i in cap:
                o = cap[i]                       # [B, P, D]
                pooled[i].append(o.mean(dim=1).cpu().numpy())
                flat[i].append(o.reshape(-1, o.shape[-1]).cpu().numpy())
    return {i: (np.concatenate(pooled[i], 0), np.concatenate(flat[i], 0)) for i in range(nL)}

print("extracting pretrained per-layer reps...")
reps_pt = per_layer_reps(Xval)

pooled_all = {i: [] for i in range(nL)}; flat_all = {i: [] for i in range(nL)}
for ck in CKPTS:
    seed = re.search(r"seed(\d+)", ck).group(1)
    sd = torch.load(ck, map_location=DEV, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # strict=False silently loads nothing when the key namespaces disagree, which
    # would return CKA ~= 1.0 at every layer and look like "no drift".
    assert not missing and not unexpected, (
        f"{ck}: {len(missing)} missing / {len(unexpected)} unexpected keys; "
        f"first missing={missing[:2]} first unexpected={unexpected[:2]}")
    reps_ft = per_layer_reps(Xval)
    line = []
    for i in range(nL):
        cp = linear_CKA(reps_pt[i][0], reps_ft[i][0])
        cf = linear_CKA(reps_pt[i][1], reps_ft[i][1])
        pooled_all[i].append(cp); flat_all[i].append(cf)
        line.append(f"L{i}:{cf:.3f}")
    print(f"seed {seed} (flat CKA): " + " ".join(line))

print("\n=== W4 Part A: per-layer CKA across %d seeds (index 0 = bottom) ===" % len(CKPTS))
print(f"{'layer':>6s} {'CKA_pooled':>18s} {'CKA_flat(token)':>18s}")
for i in range(nL):
    p = np.array(pooled_all[i]); f = np.array(flat_all[i])
    print(f"{i:>6d} {p.mean():>8.3f}±{p.std():.3f}      {f.mean():>8.3f}±{f.std():.3f}")
bp = np.mean([np.mean(pooled_all[i]) for i in range(3)]); tp = np.mean([np.mean(pooled_all[i]) for i in range(3, nL)])
bf = np.mean([np.mean(flat_all[i]) for i in range(3)]); tf = np.mean([np.mean(flat_all[i]) for i in range(3, nL)])
print(f"\nbottom-3 vs top-3 : pooled {bp:.3f}/{tp:.3f}   flat {bf:.3f}/{tf:.3f}")
print(f"global (mean over layers): pooled {np.mean([np.mean(pooled_all[i]) for i in range(nL)]):.3f}  "
      f"flat {np.mean([np.mean(flat_all[i]) for i in range(nL)]):.3f}")

out_path = os.environ.get("CKA_OUT")
if out_path:
    payload = {
        "device": DEV, "ckpt_glob": CKPT_GLOB, "n_ckpts": len(CKPTS),
        "seeds": [re.search(r"seed(\d+)", c).group(1) for c in CKPTS],
        "n_val_windows": int(len(Xval)), "lookback": LB, "horizon": H,
        "per_layer_pooled": {str(i): pooled_all[i] for i in range(nL)},
        "per_layer_flat": {str(i): flat_all[i] for i in range(nL)},
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {out_path}")
else:
    print("WARNING: CKA_OUT unset -- these numbers leave no run record.")
