#!/usr/bin/env python3
"""W4 re-verification, second cell (ETTm2): fine-tune to convergence with
val-NLL early stopping, then per-layer CKA (same pooled method as Part A).
Self-contained (no checkpoint upload). Reports whole-encoder CKA for
calibration against v19 ETTm2, plus per-layer pattern across seeds.
"""
import sys, copy, numpy as np, torch
sys.path.insert(0, ".")
from torch.utils.data import DataLoader, TensorDataset
from src.data.forecasting_loader import get_forecasting_loader
from src.models.moirai_detector import MoiraiAnomalyDetector, _apply_uni2ts_gradient_patch

DEV, LB, H, N, EPOCHS = "cuda", 96, 96, 10000, 15
SEEDS = [42, 123, 202]

def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    A = X.T @ X; B = Y.T @ Y
    return float(np.trace(A @ B) / (np.sqrt(np.trace(A @ A)) * np.sqrt(np.trace(B @ B)) + 1e-12))

_apply_uni2ts_gradient_patch()
loader = get_forecasting_loader("data/forecasting/ETTm2.csv", lookback_window=LB, forecast_horizon=H, features="M")
train_df, val_df, _ = loader.get_splits(); cols = loader.FEATURE_COLUMNS
tr, va = train_df[cols].values, val_df[cols].values
def mk(d, c, h):
    X, y = [], []
    for i in range(len(d) - c - h + 1): X.append(d[i:i+c]); y.append(d[i+c:i+c+h])
    return np.array(X), np.array(y)
Xtr0, ytr0 = mk(tr, LB, H)
Xvw, yvw = mk(va, LB, H)                       # val windows for ES-NLL (context+target)
Xrep = np.array([va[i:i+LB+H] for i in range(len(va) - LB - H + 1)])[:200]  # 192-len for CKA reps

def layer_reps(model, layers, nL):
    model.eval(); data = torch.from_numpy(Xrep).float(); pooled = {i: [] for i in range(nL)}
    for j in range(0, len(data), 32):
        b = data[j:j+32].to(DEV); bb, sl = b.shape[0], b.shape[1]
        po = torch.ones_like(b, dtype=torch.bool); pp = torch.zeros(bb, sl, dtype=torch.bool, device=DEV)
        cap = {}
        def mk2(i):
            def h(m, i_, o): cap[i] = (o[0] if isinstance(o, tuple) else o).detach()
            return h
        hs = [l.register_forward_hook(mk2(i)) for i, l in enumerate(layers)]
        with torch.no_grad():
            try: model.forward(past_target=b, past_observed_target=po, past_is_pad=pp, num_samples=2)
            except Exception: pass
        for hh in hs: hh.remove()
        for i in range(nL):
            if i in cap: pooled[i].append(cap[i].mean(1).cpu().numpy())
    return {i: np.concatenate(pooled[i], 0) for i in range(nL)}

def val_nll(model):
    model.eval(); tot = 0.0; nb = 0
    with torch.no_grad():
        for j in range(0, min(len(Xvw), 512), 64):
            xb = torch.from_numpy(Xvw[j:j+64]).float().to(DEV); yb = torch.from_numpy(yvw[j:j+64]).float().to(DEV)
            b = xb.shape[0]; full = torch.cat([xb, yb], dim=1); sl, nf = full.shape[1], full.shape[2]
            obs = torch.ones(b, sl, nf, dtype=torch.bool, device=DEV); pad = torch.zeros(b, sl, dtype=torch.bool, device=DEV)
            try: tot += float(model._val_loss(patch_size=32, target=full, observed_target=obs, is_pad=pad).mean()); nb += 1
            except Exception: pass
    return tot / max(nb, 1)

per_layer = {}; glob = []
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    idx = np.random.choice(len(Xtr0), min(N, len(Xtr0)), replace=False)
    Xtr, ytr = Xtr0[idx], ytr0[idx]
    det = MoiraiAnomalyDetector(model_size="small", context_length=LB, prediction_length=H,
                                target_dim=len(cols), num_samples=20, device=DEV)
    det.initialize(); model = det.model
    enc = model.module.encoder; layers = enc.layers; nL = len(layers)
    reps_pt = layer_reps(model, layers, nL)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).float())
    dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    best_nll = float("inf"); best_state = copy.deepcopy(model.state_dict()); best_ep = 0
    for ep in range(EPOCHS):
        model.train()
        for ctx, tgt in dl:
            ctx, tgt = ctx.to(DEV), tgt.to(DEV); b = ctx.shape[0]
            full = torch.cat([ctx, tgt], dim=1); sl, nf = full.shape[1], full.shape[2]
            obs = torch.ones(b, sl, nf, dtype=torch.bool, device=DEV); pad = torch.zeros(b, sl, dtype=torch.bool, device=DEV)
            try: nll = model._val_loss(patch_size=32, target=full, observed_target=obs, is_pad=pad).mean()
            except Exception: continue
            opt.zero_grad(); nll.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vn = val_nll(model)
        if vn < best_nll: best_nll = vn; best_state = copy.deepcopy(model.state_dict()); best_ep = ep
    model.load_state_dict(best_state)  # ES restore
    reps_ft = layer_reps(model, layers, nL)
    pl = [cka(reps_pt[i], reps_ft[i]) for i in range(nL)]
    ge = cka(np.concatenate([reps_pt[i] for i in [nL-1]], 0), np.concatenate([reps_ft[i] for i in [nL-1]], 0))
    for i in range(nL): per_layer.setdefault(i, []).append(pl[i])
    glob.append(ge)
    print(f"seed {seed}: best_ep={best_ep} per-layer CKA={[round(x,3) for x in pl]}")

nL = len(per_layer)
print("\n=== ETTm2 SECOND-CELL per-layer CKA (%d seeds, index 0=bottom) ===" % len(SEEDS))
for i in range(nL):
    a = np.array(per_layer[i]); print(f"  layer {i}: {a.mean():.3f} ± {a.std():.3f}")
means = [np.mean(per_layer[i]) for i in range(nL)]
print(f"bottom-3 mean {np.mean(means[:3]):.3f}  top-3 mean {np.mean(means[3:]):.3f}")
print(f"argmax-drift layer (min CKA) = L{int(np.argmin(means))}")
print("compare ETTh2 (Part A): non-monotone, middle-peaked (L2-L3 min), bottom L0~1.0")
