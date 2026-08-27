#!/usr/bin/env python3
"""W4 independent re-verification:
(1) per-layer CKA via TWO implementations (feature-form vs Gram/HSIC-form) -> rule out CKA bug.
(2) second cell (ETTm2) -> rule out ETTh2-specificity.
(3) per-layer param-count + weight-norm -> confirm gradient norms are comparable across layers.
"""
import sys, glob, re, numpy as np, torch
sys.path.insert(0, ".")
from src.data.forecasting_loader import get_forecasting_loader
from src.models.moirai_detector import MoiraiAnomalyDetector, _apply_uni2ts_gradient_patch

DEV, LB, H = "cuda", 96, 96

def cka_feature(X, Y):        # ||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    X = X - X.mean(0); Y = Y - Y.mean(0)
    A = X.T @ X; B = Y.T @ Y
    return float(np.trace(A @ B) / (np.sqrt(np.trace(A @ A)) * np.sqrt(np.trace(B @ B)) + 1e-12))

def cka_gram(X, Y):           # HSIC via n x n Gram matrices (independent code path)
    X = X - X.mean(0); Y = Y - Y.mean(0)
    K = X @ X.T; L = Y @ Y.T
    return float((K * L).sum() / (np.sqrt((K * K).sum()) * np.sqrt((L * L).sum()) + 1e-12))

_apply_uni2ts_gradient_patch()

def run_cell(name, csv, ckpt_glob, max_seeds=5):
    loader = get_forecasting_loader(csv, lookback_window=LB, forecast_horizon=H, features="M")
    _, val_df, _ = loader.get_splits(); cols = loader.FEATURE_COLUMNS; va = val_df[cols].values
    Xval = np.array([va[i:i+LB+H] for i in range(len(va) - LB - H + 1)])[:200]
    det = MoiraiAnomalyDetector(model_size="small", context_length=LB, prediction_length=H,
                                target_dim=len(cols), num_samples=20, device=DEV)
    det.initialize(); model = det.model
    enc = model.module.encoder; layers = enc.layers; nL = len(layers)
    # param counts + pretrained weight norms per layer (architecturally identical check)
    pcounts = [sum(p.numel() for p in layers[i].parameters()) for i in range(nL)]
    wnorms = [float(sum(p.detach().float().pow(2).sum() for p in layers[i].parameters()) ** 0.5) for i in range(nL)]

    def per_layer_reps():
        model.eval(); data = torch.from_numpy(Xval).float(); pooled = {i: [] for i in range(nL)}
        for j in range(0, len(data), 32):
            b = data[j:j+32].to(DEV); bb, sl = b.shape[0], b.shape[1]
            po = torch.ones_like(b, dtype=torch.bool); pp = torch.zeros(bb, sl, dtype=torch.bool, device=DEV)
            cap = {}
            def mk(i):
                def h(m, inp, out): cap[i] = (out[0] if isinstance(out, tuple) else out).detach()
                return h
            hs = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
            with torch.no_grad():
                try: model.forward(past_target=b, past_observed_target=po, past_is_pad=pp, num_samples=2)
                except Exception: pass
            for hh in hs: hh.remove()
            for i in range(nL):
                if i in cap: pooled[i].append(cap[i].mean(1).cpu().numpy())
        return {i: np.concatenate(pooled[i], 0) for i in range(nL)}

    reps_pt = per_layer_reps()
    cks = sorted(glob.glob(ckpt_glob))[:max_seeds]
    feat = {i: [] for i in range(nL)}; gram = {i: [] for i in range(nL)}
    for ck in cks:
        model.load_state_dict(torch.load(ck, map_location=DEV, weights_only=False), strict=True)
        reps_ft = per_layer_reps()
        for i in range(nL):
            feat[i].append(cka_feature(reps_pt[i], reps_ft[i]))
            gram[i].append(cka_gram(reps_pt[i], reps_ft[i]))
    print(f"\n=== {name}  ({len(cks)} seeds) ===")
    print(f"param counts per layer: {pcounts}  (identical? {len(set(pcounts))==1})")
    print(f"pretrained weight-norm per layer: {[round(w,1) for w in wnorms]}")
    print(f"{'layer':>5s} {'CKA_feature':>13s} {'CKA_gram':>13s} {'max|diff|':>10s}")
    for i in range(nL):
        f = np.mean(feat[i]); g = np.mean(gram[i]); d = max(abs(np.array(feat[i]) - np.array(gram[i])))
        print(f"{i:>5d} {f:>13.4f} {g:>13.4f} {d:>10.2e}")

run_cell("ETTh2 (primary)", "data/forecasting/ETTh2.csv",
         "results/v19_cuda_etth2_n10k/seed*/best_encoder.pt", max_seeds=9)
run_cell("ETTm2 (2nd cell)", "data/forecasting/ETTm2.csv",
         "results/v19_cuda_ettm2_n10k/seed*/best_encoder.pt", max_seeds=5)
