#!/usr/bin/env python3
"""Auxiliary probes reproducing finetune_forecasting.py's EXACT probe protocol.

Recovered from scripts/finetune_forecasting.py:1029-1034 and :709-730:
  extended_lookback = lookback + horizon = 192      -> the model's INPUT window is 192 steps
  make_eval_sequences: X = data[i:i+192], y = data[i+192:i+288]
  n_probe_train = min(300, len(X_val_eval))   -> probe TRAIN = first 300 windows of VALIDATION
  n_probe_val   = min(200, len(X_test_eval))  -> probe TEST  = first 200 windows of TEST
  Ridge(alpha=1.0) on mean-pooled encoder outputs  (linear_probe_r2, :197)

Targets are per-window scalars on the OT feature of the 192-step INPUT window.
"""
import sys, glob, re, json, os
import numpy as np, torch
from sklearn.linear_model import Ridge

sys.path.insert(0, ".")
from src.data.forecasting_loader import get_forecasting_loader
from src.models.moirai_detector import MoiraiAnomalyDetector, _apply_uni2ts_gradient_patch

DEV, LB, H = "cpu", 96, 96
EXT_LB = LB + H                       # 192 -- the input window
N_TR, N_TE = 300, 200
CKPTS = sorted(glob.glob("results/v19_cuda_etth2_n10k/seed*/best_encoder.pt"))
OUT = "results/v34_probe_exact"

def t_mean(c):  return c.mean(axis=1, keepdims=True)
def t_var(c):   return c.var(axis=1, keepdims=True)
def t_lag1(c):
    x, xl = c[:, 1:], c[:, :-1]; mu = c.mean(axis=1, keepdims=True)
    return (((x-mu)*(xl-mu)).mean(axis=1) / (((c-mu)**2).mean(axis=1)+1e-8)).reshape(-1,1)
def t_lag24(c):
    k=24; x, xl = c[:, k:], c[:, :-k]; mu = c.mean(axis=1, keepdims=True)
    return (((x-mu)*(xl-mu)).mean(axis=1) / (((c-mu)**2).mean(axis=1)+1e-8)).reshape(-1,1)
def t_trend(c):
    T=c.shape[1]; t=np.arange(T,dtype=np.float64); t=t-t.mean()
    xc=c-c.mean(axis=1,keepdims=True)
    return ((xc*t).sum(axis=1)/(t**2).sum()).reshape(-1,1)
def t_spec(c):
    xc=c-c.mean(axis=1,keepdims=True); P=np.abs(np.fft.rfft(xc,axis=1))**2
    f=np.arange(P.shape[1],dtype=np.float64)
    return ((P*f).sum(axis=1)/(P.sum(axis=1)+1e-12)/max(P.shape[1]-1,1)).reshape(-1,1)

ORIG = {"lag1": t_lag1, "mean": t_mean, "var": t_var}
NEW  = {"trend": t_trend, "lag24": t_lag24, "spectral_centroid": t_spec}

def make_eval_sequences(data, ext_lb, hz):
    X, y = [], []
    for i in range(len(data) - (ext_lb+hz) + 1):
        X.append(data[i:i+ext_lb]); y.append(data[i+ext_lb:i+ext_lb+hz])
    return np.array(X), np.array(y)

def main():
    torch.manual_seed(0); np.random.seed(0); _apply_uni2ts_gradient_patch()
    loader = get_forecasting_loader("data/forecasting/ETTh2.csv", lookback_window=LB,
                                    forecast_horizon=H, features="M")
    _, val_df, test_df = loader.get_splits()
    cols = loader.FEATURE_COLUMNS; ot = cols.index("OT")
    Xv, yv = make_eval_sequences(val_df[cols].values, EXT_LB, H)
    Xt, yt = make_eval_sequences(test_df[cols].values, EXT_LB, H)
    AUXSPLIT = os.environ.get("AUX_SPLIT", "val_to_test")
    if AUXSPLIT == "within_val":
        Xtr, Xte = Xv[:N_TR], Xv[N_TR:N_TR+N_TE]
        ytr, yte = yv[:N_TR], yv[N_TR:N_TR+N_TE]
    else:
        Xtr, Xte = Xv[:N_TR], Xt[:N_TE]
        ytr, yte = yv[:N_TR], yt[:N_TE]
    print(f"aux split mode: {AUXSPLIT}")
    print(f"probe train {Xtr.shape} (val)   probe test {Xte.shape} (test)")

    ctx_tr = Xtr[:, :, ot].astype(np.float64)     # full 192-step input window
    ctx_te = Xte[:, :, ot].astype(np.float64)
    tg = {}
    for n, f in {**ORIG, **NEW}.items():
        tg[n] = (f(ctx_tr), f(ctx_te))
    tg["forecast96_OT"] = (ytr[:, :, ot].astype(np.float64), yte[:, :, ot].astype(np.float64))
    tg["forecast96_ALLFEAT"] = (ytr.reshape(len(ytr),-1).astype(np.float64), yte.reshape(len(yte),-1).astype(np.float64))

    det = MoiraiAnomalyDetector(model_size="small", context_length=LB, prediction_length=H,
                                target_dim=len(cols), num_samples=20, device=DEV)
    det.initialize(); model = det.model
    enc = model.module.encoder if not hasattr(model.module,"base_model") else model.module.base_model.model.encoder

    def reps(Xnp):
        model.eval(); d = torch.from_numpy(Xnp).float(); out=[]; cap={}
        def h(m,i,o): cap['o']=(o[0] if isinstance(o,tuple) else o).detach()
        hd = enc.register_forward_hook(h)
        for j in range(0, len(d), 32):
            b=d[j:j+32].to(DEV); bb,sl=b.shape[0],b.shape[1]
            po=torch.ones_like(b,dtype=torch.bool); pp=torch.zeros(bb,sl,dtype=torch.bool,device=DEV)
            cap.clear()
            with torch.no_grad():
                try: model.forward(past_target=b,past_observed_target=po,past_is_pad=pp,num_samples=2)
                except Exception: pass
            if 'o' in cap: out.append(cap['o'].mean(dim=1).cpu().numpy())
        hd.remove(); return np.concatenate(out,0)

    def probe_all():
        Rtr, Rte = reps(Xtr), reps(Xte)
        return {n: float(Ridge(alpha=1.0).fit(Rtr, a).score(Rte, b)) for n,(a,b) in tg.items()}

    print("pretrained reference...")
    r2_pt = probe_all()
    for n,v in r2_pt.items(): print(f"  R2(PT) {n:20s} {v:+.4f}")

    per_seed={}
    for ck in CKPTS:
        seed=re.search(r"seed(\d+)",ck).group(1)
        sd=torch.load(ck,map_location=DEV,weights_only=False)
        miss,unexp=model.load_state_dict(sd,strict=False)
        assert not miss and not unexp
        per_seed[seed]=probe_all()
        print(f"seed {seed}: " + " ".join(f"{n}={per_seed[seed][n]-r2_pt[n]:+.3f}" for n in r2_pt))

    print(f"\n{'probe':<20s}{'R2(PT)':>9s}{'R2(FT)':>9s}{'dR2':>10s}{'sd':>8s}{'neg':>7s}")
    summ={}
    for n in r2_pt:
        d=np.array([per_seed[s][n]-r2_pt[n] for s in per_seed])
        ft=np.array([per_seed[s][n] for s in per_seed])
        summ[n]={"r2_pt":r2_pt[n],"r2_ft_mean":float(ft.mean()),"dr2_mean":float(d.mean()),
                 "dr2_std":float(d.std()),"n_negative":int((d<0).sum()),"n_seeds":len(d),
                 "original":n in ORIG,"new":n in NEW}
        print(f"{n:<20s}{r2_pt[n]:>+9.4f}{ft.mean():>+9.4f}{d.mean():>+10.4f}{d.std():>8.4f}{(d<0).sum():>4d}/{len(d)}")
    os.makedirs(OUT,exist_ok=True)
    json.dump({"summary":summ,"per_seed":per_seed,"r2_pt":r2_pt,
               "protocol":"train=val[:300], test=test[:200], input window=192, Ridge a=1.0"},
              open(f"{OUT}/etth2_exact_{AUXSPLIT}.json","w"),indent=2)
    print(f"\nwrote {OUT}/etth2_exact_{AUXSPLIT}.json")
    print("TARGET (appendix.tex:704-706): lag1 R2(PT)=+0.183 dR2=-0.052 | mean +0.094 -0.033 | var +0.127 -0.048")

if __name__=="__main__": main()
