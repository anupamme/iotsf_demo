#!/usr/bin/env python3
"""W4 Part B: per-layer gradient norms on a converged Moirai-Small/ETTh2 run.

Logs per-step per-layer gradient L2-norm during fine-tuning (n=10k, ES, multiple
seeds), reporting mean AND median over steps (robust to spikes). Compared against
the validated per-layer CKA drift profile from Part A.
"""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from torch.utils.data import DataLoader, TensorDataset
from src.data.forecasting_loader import get_forecasting_loader
from src.models.moirai_detector import MoiraiAnomalyDetector, _apply_uni2ts_gradient_patch

DEV, LB, H, N, EPOCHS = "cuda", 96, 96, 10000, 15
SEEDS = [42, 123, 202]

_apply_uni2ts_gradient_patch()
loader = get_forecasting_loader("data/forecasting/ETTh2.csv", lookback_window=LB, forecast_horizon=H, features="M")
train_df, _, _ = loader.get_splits(); cols = loader.FEATURE_COLUMNS; tr = train_df[cols].values
def mk(d, c, h):
    X, y = [], []
    for i in range(len(d) - c - h + 1): X.append(d[i:i+c]); y.append(d[i+c:i+c+h])
    return np.array(X), np.array(y)
Xtr0, ytr0 = mk(tr, LB, H)

per_seed_mean, per_seed_med = [], []
dom_param = {}
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    idx = np.random.choice(len(Xtr0), min(N, len(Xtr0)), replace=False)
    Xtr, ytr = Xtr0[idx], ytr0[idx]
    det = MoiraiAnomalyDetector(model_size="small", context_length=LB, prediction_length=H,
                                target_dim=len(cols), num_samples=20, device=DEV)
    det.initialize(); model = det.model
    enc = model.module.encoder
    layers = enc.layers; nL = len(layers)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).float())
    dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    gsteps = {i: [] for i in range(nL)}
    pmax = {}  # track dominant param per layer
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
                continue
            opt.zero_grad(); nll.backward()
            for i, l in enumerate(layers):
                gn = 0.0
                for name, p in l.named_parameters():
                    if p.grad is not None:
                        v = float(p.grad.pow(2).sum())
                        gn += v
                        key = (i, name)
                        if v > pmax.get(key, 0): pmax[key] = v
                gsteps[i].append(gn ** 0.5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); last = float(nll)
    means = [float(np.mean(gsteps[i])) for i in range(nL)]
    meds = [float(np.median(gsteps[i])) for i in range(nL)]
    per_seed_mean.append(means); per_seed_med.append(meds)
    # dominant param in the largest-mean layer
    top_layer = int(np.argmax(means))
    doms = sorted([(v, k) for k, v in pmax.items() if k[0] == top_layer], reverse=True)[:2]
    dom_param[seed] = (top_layer, [(k[1], round(v**0.5, 1)) for v, k in doms])
    print(f"seed {seed} done (final nll {last:.3f}); layer-mean gradnorms: {[round(m,1) for m in means]}")

mean_arr = np.array(per_seed_mean); med_arr = np.array(per_seed_med)
print("\n=== W4 Part B: per-layer gradient norm (Moirai-Small/ETTh2 n=10k, %d seeds, %d ep) ===" % (len(SEEDS), EPOCHS))
print(f"{'layer':>5s} {'grad_mean(±sd)':>18s} {'grad_median(±sd)':>18s}")
for i in range(mean_arr.shape[1]):
    print(f"{i:>5d} {mean_arr[:,i].mean():>10.1f}±{mean_arr[:,i].std():>6.1f}   {med_arr[:,i].mean():>10.1f}±{med_arr[:,i].std():>6.1f}")
print("\ndominant param in largest-grad layer per seed:", dom_param)
# reconcile vs Part A drift profile (validated pooled CKA, 9-seed): middle-concentrated
driftA = [1-0.998, 1-0.981, 1-0.815, 1-0.676, 1-0.945, 1-0.943]  # from Part A pooled
gm = med_arr.mean(0)
r = np.corrcoef(gm, driftA)[0,1]
print(f"\ncorr(median grad norm, PartA drift=1-CKA) = {r:+.3f}")
print("PartA drift profile (1-CKA): ", [round(d,3) for d in driftA])
