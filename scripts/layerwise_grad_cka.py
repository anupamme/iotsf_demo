#!/usr/bin/env python3
"""W4: per-layer gradient norms vs per-layer CKA drift on Moirai-Small/ETTh2.

Reviewer W4: the "bottom layers drift more" pattern needs gradient-norm
measurements to tell gradient-flow apart from functional specialisation.
This logs, per encoder layer: (a) mean gradient norm during fine-tuning,
(b) CKA(pretrained, fine-tuned) representation drift. If bottom layers have
BOTH larger gradients and lower CKA, gradient flow is sufficient; if the CKA
pattern is not tracked by the gradient-norm profile, that points to functional
specialisation.
"""
import sys, numpy as np, torch, pandas as pd
sys.path.insert(0, ".")
from torch.utils.data import DataLoader, TensorDataset
from src.data.forecasting_loader import get_forecasting_loader
from src.models.moirai_detector import MoiraiAnomalyDetector, _apply_uni2ts_gradient_patch

DEV = "cuda"
SEED, N, EPOCHS, LB, H = 42, 10000, 6, 96, 96

def linear_CKA(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    XtX = X.T @ X; YtY = Y.T @ Y
    num = np.trace(XtX @ YtY); den = np.sqrt(np.trace(XtX @ XtX) * np.trace(YtY @ YtY))
    return float(num / den) if den > 1e-10 else 0.0

torch.manual_seed(SEED); np.random.seed(SEED)
_apply_uni2ts_gradient_patch()

loader = get_forecasting_loader("data/forecasting/ETTh2.csv", lookback_window=LB,
                                forecast_horizon=H, features="M")
train_df, val_df, _ = loader.get_splits()
cols = loader.FEATURE_COLUMNS
tr, va = train_df[cols].values, val_df[cols].values

def mk_train(d, c, h):
    X, y = [], []
    for i in range(len(d) - c - h + 1):
        X.append(d[i:i+c]); y.append(d[i+c:i+c+h])
    return np.array(X), np.array(y)

Xtr, ytr = mk_train(tr, LB, H)
idx = np.random.choice(len(Xtr), min(N, len(Xtr)), replace=False)
Xtr, ytr = Xtr[idx], ytr[idx]
# val reps use 192-length (lookback+horizon) windows, matching the main script
Xval = np.array([va[i:i+LB+H] for i in range(len(va) - LB - H + 1)])[:200]
print(f"train={len(Xtr)}  val_reps={len(Xval)}")

det = MoiraiAnomalyDetector(model_size="small", context_length=LB,
                            prediction_length=H, target_dim=len(cols),
                            num_samples=20, device=DEV)
det.initialize()
model = det.model
enc = model.module.encoder if not hasattr(model.module, "base_model") else model.module.base_model.model.encoder
layers = enc.layers
nL = len(layers)
print(f"encoder layers: {nL}")

def extract_layer_reps(Xnp):
    model.eval()
    data = torch.from_numpy(Xnp).float()
    per = [[] for _ in range(nL)]
    for j in range(0, len(data), 32):
        b = data[j:j+32].to(DEV); bb, sl = b.shape[0], b.shape[1]
        po = torch.ones_like(b, dtype=torch.bool)
        pp = torch.zeros(bb, sl, dtype=torch.bool, device=DEV)
        cap = {}
        def mk(i):
            def h(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                cap[i] = o.mean(dim=1).detach().cpu().numpy()
            return h
        hs = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
        with torch.no_grad():
            try:
                model.forward(past_target=b, past_observed_target=po, past_is_pad=pp, num_samples=2)
            except Exception as e:
                print("fwd warn:", str(e)[:80])
        for h in hs: h.remove()
        for i in range(nL):
            if i in cap: per[i].append(cap[i])
    return {i: np.concatenate(per[i], 0) for i in range(nL) if per[i]}

print("extracting pretrained per-layer reps...")
reps_pt = extract_layer_reps(Xval)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).float())
dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
gacc = {i: 0.0 for i in range(nL)}; nstep = 0
model.train()
for ep in range(EPOCHS):
    last = 0.0
    for ctx, tgt in dl:
        ctx, tgt = ctx.to(DEV), tgt.to(DEV); b = ctx.shape[0]
        full = torch.cat([ctx, tgt], dim=1); sl, nf = full.shape[1], full.shape[2]
        obs = torch.ones(b, sl, nf, dtype=torch.bool, device=DEV)
        pad = torch.zeros(b, sl, dtype=torch.bool, device=DEV)
        try:
            nll = model._val_loss(patch_size=32, target=full, observed_target=obs, is_pad=pad).mean()
        except Exception as e:
            print("step warn:", str(e)[:80]); continue
        opt.zero_grad(); nll.backward()
        for i, l in enumerate(layers):
            gn = sum(p.grad.pow(2).sum().item() for p in l.parameters() if p.grad is not None)
            gacc[i] += gn ** 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); nstep += 1; last = nll.item()
    print(f"epoch {ep}: nll={last:.4f}")

print("extracting fine-tuned per-layer reps...")
reps_ft = extract_layer_reps(Xval)

print("\n=== W4 RESULT: per-layer gradient norm vs CKA drift (Moirai-Small/ETTh2, n=%d) ===" % N)
print(f"{'layer':>5s} {'mean_grad_norm':>15s} {'CKA(PT,FT)':>12s}  (lower CKA = more drift)")
gnorms, ckas = [], []
for i in range(nL):
    g = gacc[i] / max(nstep, 1)
    c = linear_CKA(reps_pt[i], reps_ft[i]) if i in reps_pt and i in reps_ft else float("nan")
    gnorms.append(g); ckas.append(c)
    print(f"{i:>5d} {g:>15.4f} {c:>12.4f}")
# correlation between grad-norm profile and drift (1-CKA)
drift = [1 - c for c in ckas]
if nL > 2:
    r = np.corrcoef(gnorms, drift)[0, 1]
    print(f"\ncorr(grad_norm, drift=1-CKA) across layers = {r:+.3f}")
    print("high positive corr -> gradient-flow explains the drift pattern;")
    print("weak/negative corr -> drift is not tracked by gradient magnitude (functional specialisation).")
