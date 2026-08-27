#!/usr/bin/env python3
"""Task-orthogonal probes on saved Moirai encoders (no retraining).

Round-12 reviewer priority: the auxiliary probe set (mean / variance / lag-1
autocorrelation) is too small and too homogeneous to support "no auxiliary probe
increases". This adds three structurally different targets -- trend, lag-24
(seasonal/long-range) autocorrelation, and spectral centroid -- and recomputes
the original three as a pipeline control.

Protocol matches sections/appendix.tex:690-712 exactly:
  Ridge(alpha=1.0) on mean-pooled encoder outputs, h=96, condition B,
  targets are per-window scalars on the OT feature of the INPUT (context) window.
  R2(PT) is reported for every probe: a target with a negative floor is
  uninformative and must NOT be counted toward the asymmetry signature.

NOTE ON REPRODUCIBILITY: this is a re-implementation. The generating code for the
published table (appendix.tex:704-706) is not in the repo, and this pipeline does
not reproduce its absolute R2 values (e.g. lag1 R2(PT) here vs +0.183 published).
The probe train/test split must be SHUFFLED -- a contiguous split puts probe train
and test on different regimes of the ETT val series and drives every R2 negative.
Only the WITHIN-RUN comparison (trained probe vs auxiliary probes, same protocol,
same seeds) is used for the asymmetry claim.
"""
import sys, glob, re, json, os
import numpy as np, torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.data.forecasting_loader import get_forecasting_loader
from src.models.moirai_detector import MoiraiAnomalyDetector, _apply_uni2ts_gradient_patch

DEV, LB, H = "cpu", 96, 96
N_PROBE_TRAIN = 300                      # appendix: n_probe = 300, remainder = test
CKPTS = sorted(glob.glob("results/v19_cuda_etth2_n10k/seed*/best_encoder.pt"))
OUT = "results/v33_orthogonal_probes"

# ---- targets: all computed on the OT context window, shape (N,1) ----
def t_mean(ctx):  return ctx.mean(axis=1, keepdims=True)
def t_var(ctx):   return ctx.var(axis=1, keepdims=True)

def t_lag1(ctx):
    x, xl = ctx[:, 1:], ctx[:, :-1]
    mu = ctx.mean(axis=1, keepdims=True)
    num = ((x - mu) * (xl - mu)).mean(axis=1)
    den = ((ctx - mu) ** 2).mean(axis=1)
    return (num / (den + 1e-8)).reshape(-1, 1)

def t_lag24(ctx):                      # seasonality / long-range (daily cycle, hourly ETT)
    k = 24
    x, xl = ctx[:, k:], ctx[:, :-k]
    mu = ctx.mean(axis=1, keepdims=True)
    num = ((x - mu) * (xl - mu)).mean(axis=1)
    den = ((ctx - mu) ** 2).mean(axis=1)
    return (num / (den + 1e-8)).reshape(-1, 1)

def t_trend(ctx):                      # OLS slope over the window
    T = ctx.shape[1]
    t = np.arange(T, dtype=np.float64); t = t - t.mean()
    xc = ctx - ctx.mean(axis=1, keepdims=True)
    return ((xc * t).sum(axis=1) / (t ** 2).sum()).reshape(-1, 1)

def t_spectral_centroid(ctx):          # normalized spectral centroid
    xc = ctx - ctx.mean(axis=1, keepdims=True)
    P = np.abs(np.fft.rfft(xc, axis=1)) ** 2
    f = np.arange(P.shape[1], dtype=np.float64)
    return ((P * f).sum(axis=1) / (P.sum(axis=1) + 1e-12) / max(P.shape[1] - 1, 1)).reshape(-1, 1)

TARGETS = {  # name -> (fn, is_original)
    "lag1": (t_lag1, True), "mean": (t_mean, True), "var": (t_var, True),
    "trend": (t_trend, False), "lag24": (t_lag24, False),
    "spectral_centroid": (t_spectral_centroid, False),
}

def main():
    torch.manual_seed(0); np.random.seed(0)
    _apply_uni2ts_gradient_patch()
    loader = get_forecasting_loader("data/forecasting/ETTh2.csv", lookback_window=LB,
                                    forecast_horizon=H, features="M")
    _, val_df, test_df = loader.get_splits()
    cols = loader.FEATURE_COLUMNS
    ot = cols.index("OT")
    # appendix.tex:363-366: representations are extracted on the held-out
    # VALIDATION AND TEST sets (192-timestep windows).
    va = np.concatenate([val_df[cols].values, test_df[cols].values], axis=0)
    X = np.array([va[i:i+LB+H] for i in range(len(va)-LB-H+1)])
    N_WIN = len(X)
    ctx_ot = X[:, :LB, ot].astype(np.float64)          # context window, OT feature
    print(f"windows {X.shape}  context {ctx_ot.shape}")

    det = MoiraiAnomalyDetector(model_size="small", context_length=LB, prediction_length=H,
                                target_dim=len(cols), num_samples=20, device=DEV)
    det.initialize(); model = det.model
    enc = model.module.encoder if not hasattr(model.module, "base_model") else model.module.base_model.model.encoder

    def reps(Xnp):
        model.eval(); data = torch.from_numpy(Xnp).float(); out = []; cap = {}
        def h(m, i, o): cap['o'] = (o[0] if isinstance(o, tuple) else o).detach()
        hd = enc.register_forward_hook(h)
        for j in range(0, len(data), 32):
            b = data[j:j+32].to(DEV); bb, sl = b.shape[0], b.shape[1]
            po = torch.ones_like(b, dtype=torch.bool)
            pp = torch.zeros(bb, sl, dtype=torch.bool, device=DEV)
            cap.clear()
            with torch.no_grad():
                try: model.forward(past_target=b, past_observed_target=po, past_is_pad=pp, num_samples=2)
                except Exception: pass
            if 'o' in cap: out.append(cap['o'].mean(dim=1).cpu().numpy())
        hd.remove()
        return np.concatenate(out, 0)

    SPLIT = os.environ.get("PROBE_SPLIT", "contiguous")
    idx = np.arange(N_WIN)
    if SPLIT == "shuffled":
        np.random.RandomState(42).shuffle(idx)
    tr, te = idx[:N_PROBE_TRAIN], idx[N_PROBE_TRAIN:]
    print(f"split={SPLIT}  probe-train={len(tr)}  probe-test={len(te)}")
    tgts = {n: f(ctx_ot) for n, (f, _) in TARGETS.items()}
    tgts["forecast96_TRAINED"] = X[:, LB:, ot].astype(np.float64)   # the trained objective

    def probe_all(R):
        sc = StandardScaler().fit(R[tr]); Rs = sc.transform(R)
        return {n: float(Ridge(alpha=1.0).fit(Rs[tr], t[tr]).score(Rs[te], t[te]))
                for n, t in tgts.items()}

    print("extracting pretrained (PT reference)...")
    r2_pt = probe_all(reps(X))
    for n in r2_pt: print(f"  R2(PT) {n:20s} {r2_pt[n]:+.4f}")

    per_seed = {}
    for ck in CKPTS:
        seed = re.search(r"seed(\d+)", ck).group(1)
        sd = torch.load(ck, map_location=DEV, weights_only=False)
        miss, unexp = model.load_state_dict(sd, strict=False)
        assert not miss and not unexp, (len(miss), len(unexp))
        per_seed[seed] = probe_all(reps(X))
        print(f"seed {seed}: " + " ".join(f"{n}={per_seed[seed][n]-r2_pt[n]:+.3f}" for n in r2_pt))

    print(f"\n=== delta-R2 over {len(per_seed)} seeds (FT - PT), ETTh2 n=10k, h=96, cond B ===")
    print(f"{'probe':<20s}{'R2(PT)':>9s}{'R2(FT)':>9s}{'dR2 mean':>11s}{'sd':>8s}{'neg':>7s}  floor")
    summary = {}
    for n in r2_pt:
        d = np.array([per_seed[s][n] - r2_pt[n] for s in per_seed])
        ft = np.array([per_seed[s][n] for s in per_seed])
        floor = "positive" if r2_pt[n] > 0 else "NEGATIVE (uninformative)"
        summary[n] = {"r2_pt": r2_pt[n], "r2_ft_mean": float(ft.mean()),
                      "dr2_mean": float(d.mean()), "dr2_std": float(d.std()),
                      "n_negative": int((d < 0).sum()), "n_seeds": len(d),
                      "positive_floor": bool(r2_pt[n] > 0),
                      "original": TARGETS[n][1] if n in TARGETS else None,
                      "trained_objective": n.endswith("_TRAINED")}
        print(f"{n:<20s}{r2_pt[n]:>+9.4f}{ft.mean():>+9.4f}{d.mean():>+11.4f}{d.std():>8.4f}"
              f"{(d<0).sum():>4d}/{len(d)}  {floor}")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"summary": summary, "per_seed": per_seed, "r2_pt": r2_pt,
               "n_windows": int(X.shape[0]), "n_probe_train": N_PROBE_TRAIN, "split": SPLIT},
              open(f"{OUT}/etth2_probes_{SPLIT}.json", "w"), indent=2)
    print(f"\nwrote {OUT}/etth2_probes_{SPLIT}.json")
    print("paper control (appendix.tex:704-706): lag1 -0.052+/-0.041 (9/10), "
          "mean -0.033+/-0.019 (10/10), var -0.048+/-0.030 (9/10)")

if __name__ == "__main__":
    main()
