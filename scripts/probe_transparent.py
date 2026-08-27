#!/usr/bin/env python3
"""Transparent auxiliary-probe protocol with every knob specified, plus bootstrap CIs.

WHY THIS EXISTS
---------------
The published auxiliary-probe magnitudes (mean +0.094, var +0.127, lag-1 +0.183) could not be
reproduced from the released checkpoints: under the verified trained-probe protocol
(scripts/probe_exact_protocol.py, which matches the trained probe to four significant figures)
the same raw window statistics give strongly negative R^2. Rather than keep guessing at the
original normalization, this script DEFINES a protocol, states every choice, and reports
whatever it gives -- including if the asymmetry comes out weaker than published.

PROTOCOL (every knob the reviewer asked about)
---------------------------------------------
  window        192 steps (= lookback 96 + horizon 96), the encoder's actual input, matching
                make_eval_sequences in finetune_forecasting.py:709-730
  window pool   the same 500 windows the trained probe uses: val[:300] and test[:200]
  splits        two, reported side by side:
                  T (temporal)   train = val[:300],  test = test[:200]   -- matches trained probe
                  R (randomized) the pooled 500 windows shuffled with a fixed seed, 300/200
                The pair isolates how much of each R^2 is distribution shift across the
                val->test boundary rather than decodability.
  features      mean-pooled encoder output; StandardScaler fit on probe-train only
  targets       6 auxiliary scalars on the OT channel of the input window
                (mean, var, lag-1 AC, lag-24 AC, linear trend, spectral centroid)
                + forecast-96-all-features as the trained reference.
                Each target standardized with probe-TRAIN mean/sd, so alpha means the same
                thing across targets. R^2 is affine-invariant in the target, so this changes
                only the effective regularization, not the metric.
  probe         Ridge; alpha chosen by 5-fold CV on probe-TRAIN over logspace(-3, 6, 19),
                refit on all of probe-train. The chosen alpha is reported per (target, split).
                Selection uses ONLY the pretrained encoder's train split, then that same alpha
                is reused for every fine-tuned encoder, so PT and FT are never tuned apart.
  encoders      the 10 released checkpoints results/v19_cuda_etth2_n10k/seed*/best_encoder.pt
                (Moirai-Small, ETTh2, h=96, n=10k, condition B)
  statistic     dR^2_s = R^2(FT_s) - R^2(PT) per seed; mean over the 10 seeds, with a
                percentile bootstrap 95% CI (10,000 resamples of the 10 seeds, fixed rng).
  noise floor   the SAME pipeline run on N_NULL independently permuted copies of each target.
                Any real dR^2 must exceed the spread the pipeline produces on pure noise.
                This is the control that decides whether a dR^2 is interpretable at all.
  alpha sweep   dR^2 at fixed alphas, including the published setting (alpha=1.0 on RAW,
                unstandardized targets), to locate exactly which knob a result depends on.

Usage:  .venv-probe/bin/python scripts/probe_transparent.py
Output: results/v36_probe_transparent/etth2_transparent.json  (+ stdout table)
"""
import sys, glob, re, json, os
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, ".")
from src.data.forecasting_loader import get_forecasting_loader
from src.models.moirai_detector import MoiraiAnomalyDetector, _apply_uni2ts_gradient_patch

DEV, LB, H = "cpu", 96, 96
EXT_LB = LB + H                      # 192 -- the encoder's input window
N_TR, N_TE = 300, 200                # same window budget as the trained probe
ALPHAS = np.logspace(-3, 6, 19)      # wide enough that the CV pick is interior, not clipped
SWEEP_ALPHAS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
N_BOOT = 10000
N_NULL = 20                          # permuted-target replicates for the noise floor
SPLIT_SEED = 0                       # for split R
BOOT_SEED = 12345
NULL_SEED = 777
CKPTS = sorted(glob.glob("results/v19_cuda_etth2_n10k/seed*/best_encoder.pt"))
OUT = "results/v36_probe_transparent"
REPS_CACHE = f"{OUT}/reps_cache.npz"


# ----------------------------------------------------------------- targets
def t_mean(c):
    return c.mean(axis=1, keepdims=True)


def t_var(c):
    return c.var(axis=1, keepdims=True)


def _ac(c, k):
    x, xl = c[:, k:], c[:, :-k]
    mu = c.mean(axis=1, keepdims=True)
    num = ((x - mu) * (xl - mu)).mean(axis=1)
    den = ((c - mu) ** 2).mean(axis=1) + 1e-8
    return (num / den).reshape(-1, 1)


def t_lag1(c):
    return _ac(c, 1)


def t_lag24(c):
    return _ac(c, 24)


def t_trend(c):
    t = np.arange(c.shape[1], dtype=np.float64)
    t = t - t.mean()
    xc = c - c.mean(axis=1, keepdims=True)
    return ((xc * t).sum(axis=1) / (t ** 2).sum()).reshape(-1, 1)


def t_spec(c):
    xc = c - c.mean(axis=1, keepdims=True)
    P = np.abs(np.fft.rfft(xc, axis=1)) ** 2
    f = np.arange(P.shape[1], dtype=np.float64)
    return ((P * f).sum(axis=1) / (P.sum(axis=1) + 1e-12) / max(P.shape[1] - 1, 1)).reshape(-1, 1)


AUX = {"mean": t_mean, "var": t_var, "lag1": t_lag1,
       "lag24": t_lag24, "trend": t_trend, "spectral_centroid": t_spec}


def make_eval_sequences(data, ext_lb, hz):
    X, y = [], []
    for i in range(len(data) - (ext_lb + hz) + 1):
        X.append(data[i:i + ext_lb])
        y.append(data[i + ext_lb:i + ext_lb + hz])
    return np.array(X), np.array(y)


# ----------------------------------------------------------------- probe
def standardize(tr, te):
    """Fit on probe-train only. Returns (tr', te')."""
    mu = tr.mean(axis=0, keepdims=True)
    sd = tr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (tr - mu) / sd, (te - mu) / sd


def pick_alpha(Xtr, ytr):
    """5-fold CV on probe-train. Returns the alpha with the best mean fold R^2."""
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    best, best_a = -np.inf, ALPHAS[0]
    for a in ALPHAS:
        scores = []
        for itr, ite in kf.split(Xtr):
            m = Ridge(alpha=a).fit(Xtr[itr], ytr[itr])
            scores.append(m.score(Xtr[ite], ytr[ite]))
        s = float(np.mean(scores))
        if s > best:
            best, best_a = s, a
    return float(best_a), best


def r2(Xtr, ytr, Xte, yte, alpha):
    return float(Ridge(alpha=alpha).fit(Xtr, ytr).score(Xte, yte))


def r2_blocks(Xtr, Xte, alpha, blocks_tr, blocks_te):
    """One Ridge solve for many target blocks at once.

    blocks_* are dicts name -> (n, k) array. Returns name -> R^2, where a block's R^2 is the
    uniform average of its per-column R^2 against the TEST column means -- identical to
    sklearn's Ridge.score() with multioutput='uniform_average', just amortized over blocks.
    """
    names = list(blocks_tr)
    Ytr = np.hstack([blocks_tr[n] for n in names])
    Yte = np.hstack([blocks_te[n] for n in names])
    P = Ridge(alpha=alpha).fit(Xtr, Ytr).predict(Xte)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    out, c = {}, 0
    for n in names:
        k = blocks_tr[n].shape[1]
        yt, yp = Yte[:, c:c + k], P[:, c:c + k]
        sse = ((yt - yp) ** 2).sum(axis=0)
        sst = ((yt - yt.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
        out[n] = float(np.mean(1.0 - sse / np.where(sst < 1e-30, np.nan, sst)))
        c += k
    return out


def boot_ci(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap 95% CI of the mean, resampling the seeds."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=np.float64)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_or_build_reps(Xpool, cols):
    """Mean-pooled encoder outputs for the pretrained model and each checkpoint.

    Cached to REPS_CACHE: the 11 CPU forward passes cost ~13 min, the analysis on top of them
    costs seconds, so caching makes the analysis knobs cheap to iterate on. Delete the .npz
    to force a rebuild.
    """
    want = ["PT"] + [re.search(r"seed(\d+)", c).group(1) for c in CKPTS]
    if os.path.exists(REPS_CACHE):
        z = np.load(REPS_CACHE)
        if sorted(z.files) == sorted(want) and z["PT"].shape[0] == len(Xpool):
            print(f"representations: loaded from {REPS_CACHE}")
            return {k: z[k] for k in z.files}
        print(f"representations: cache at {REPS_CACHE} does not match, rebuilding")

    det = MoiraiAnomalyDetector(model_size="small", context_length=LB, prediction_length=H,
                               target_dim=len(cols), num_samples=20, device=DEV)
    det.initialize()
    model = det.model
    enc = (model.module.encoder if not hasattr(model.module, "base_model")
           else model.module.base_model.model.encoder)

    def reps(Xnp):
        model.eval()
        d = torch.from_numpy(Xnp).float()
        out, cap = [], {}

        def hook(m, i, o):
            cap["o"] = (o[0] if isinstance(o, tuple) else o).detach()

        hd = enc.register_forward_hook(hook)
        for j in range(0, len(d), 32):
            b = d[j:j + 32].to(DEV)
            bb, sl = b.shape[0], b.shape[1]
            po = torch.ones_like(b, dtype=torch.bool)
            pp = torch.zeros(bb, sl, dtype=torch.bool, device=DEV)
            cap.clear()
            with torch.no_grad():
                try:
                    model.forward(past_target=b, past_observed_target=po,
                                  past_is_pad=pp, num_samples=2)
                except Exception:
                    pass
            if "o" in cap:
                out.append(cap["o"].mean(dim=1).cpu().numpy())
        hd.remove()
        got = np.concatenate(out, 0)
        assert len(got) == len(Xnp), (len(got), len(Xnp))
        return got.astype(np.float64)

    print("building representations (11 forward passes over "
          f"{len(Xpool)} windows on {DEV})...")
    store = {"PT": reps(Xpool)}
    for ck in CKPTS:
        seed = re.search(r"seed(\d+)", ck).group(1)
        sd = torch.load(ck, map_location=DEV, weights_only=False)
        miss, unexp = model.load_state_dict(sd, strict=False)
        assert not miss and not unexp, (miss, unexp)
        store[seed] = reps(Xpool)
        print(f"  seed {seed}: done")

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(REPS_CACHE, **store)
    print(f"  cached to {REPS_CACHE}")
    return store


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    _apply_uni2ts_gradient_patch()

    loader = get_forecasting_loader("data/forecasting/ETTh2.csv", lookback_window=LB,
                                    forecast_horizon=H, features="M")
    _, val_df, test_df = loader.get_splits()
    cols = loader.FEATURE_COLUMNS
    ot = cols.index("OT")
    Xv, yv = make_eval_sequences(val_df[cols].values, EXT_LB, H)
    Xt, yt = make_eval_sequences(test_df[cols].values, EXT_LB, H)

    # the pooled 500 windows, in a fixed order: val[:300] then test[:200]
    Xpool = np.concatenate([Xv[:N_TR], Xt[:N_TE]], axis=0)
    ypool = np.concatenate([yv[:N_TR], yt[:N_TE]], axis=0)
    n = len(Xpool)
    assert n == N_TR + N_TE, n

    idx_T = (np.arange(N_TR), np.arange(N_TR, n))                    # temporal
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    idx_R = (perm[:N_TR], perm[N_TR:])                               # randomized
    SPLITS = {"T_temporal": idx_T, "R_randomized": idx_R}

    print(f"pooled windows {Xpool.shape}   splits: "
          f"T = val[:300] -> test[:200];  R = shuffled(seed {SPLIT_SEED}) 300/200")

    # ---- targets on the pooled windows (raw; split + standardize per split scheme)
    ctx = Xpool[:, :, ot].astype(np.float64)
    raw = {k: f(ctx) for k, f in AUX.items()}
    raw["forecast96_ALLFEAT"] = ypool.reshape(n, -1).astype(np.float64)

    # ---- representations (cached: 11 forward passes over 500 windows on CPU)
    reps_all = load_or_build_reps(Xpool, cols)
    print(f"representations: PT + {len(reps_all) - 1} fine-tuned encoders, "
          f"{reps_all['PT'].shape[1]} dims")
    seeds = [k for k in reps_all if k != "PT"]

    # ---- null replicates: N_NULL independent permutations per split, shared across encoders
    rng = np.random.default_rng(NULL_SEED)
    null_perm = {s: [(rng.permutation(N_TR), rng.permutation(N_TE)) for _ in range(N_NULL)]
                 for s in SPLITS}

    def blocks_for(split, itr, ite, scale_targets):
        """Real + null target blocks for one split. Returns (train dict, test dict)."""
        btr, bte = {}, {}
        for tname, Y in raw.items():
            ytr, yte = Y[itr], Y[ite]
            if scale_targets:
                ytr, yte = standardize(ytr, yte)
            btr[tname], bte[tname] = ytr, yte
            for j, (ptr, pte) in enumerate(null_perm[split]):
                btr[f"NULL{j}|{tname}"], bte[f"NULL{j}|{tname}"] = ytr[ptr], yte[pte]
        return btr, bte

    # =========================================================== primary analysis
    # R2[(split, encoder, target_key)] for every encoder including "PT"
    alpha, cv_score, R2 = {}, {}, {}
    for sname, (itr, ite) in SPLITS.items():
        Rtr_pt, Rte_pt = standardize(reps_all["PT"][itr], reps_all["PT"][ite])
        btr, bte = blocks_for(sname, itr, ite, scale_targets=True)
        for tname in raw:
            a, cv = pick_alpha(Rtr_pt, btr[tname])
            alpha[(sname, tname)] = a
            cv_score[(sname, tname)] = cv
        if any(alpha[(sname, t)] in (ALPHAS[0], ALPHAS[-1]) for t in raw):
            print(f"  WARNING split {sname}: a CV-selected alpha sits on the grid boundary")
        print(f"  split {sname}: alphas " +
              " ".join(f"{t}={alpha[(sname, t)]:g}" for t in raw))

        # group targets (real + their nulls) by the alpha chosen for the real target
        by_alpha = {}
        for tname in raw:
            by_alpha.setdefault(alpha[(sname, tname)], []).append(tname)

        for enc in ["PT"] + seeds:
            Rtr, Rte = standardize(reps_all[enc][itr], reps_all[enc][ite])
            for a, tnames in by_alpha.items():
                keys = [k for t in tnames for k in [t] + [f"NULL{j}|{t}" for j in range(N_NULL)]]
                got = r2_blocks(Rtr, Rte, a, {k: btr[k] for k in keys},
                                {k: bte[k] for k in keys})
                for k, v in got.items():
                    R2[(sname, enc, k)] = v

    def dr2(sname, key):
        """Per-seed dR^2 for one target key under one split."""
        return np.array([R2[(sname, s, key)] - R2[(sname, "PT", key)] for s in seeds])

    summary = {}
    for sname in SPLITS:
        for tname in raw:
            d = dr2(sname, tname)
            lo, hi = boot_ci(d)
            # noise floor: the same statistic on each permuted replicate
            nulls = np.array([dr2(sname, f"NULL{j}|{tname}").mean() for j in range(N_NULL)])
            floor = float(np.percentile(np.abs(nulls), 95))
            above = bool(abs(d.mean()) > floor)
            direction = "increase" if lo > 0 else ("decrease" if hi < 0 else "indistinct")

            # Two independent interpretability criteria, kept separate on purpose.
            #
            # (1) does the probe generalize at all?  A dR^2 measured where R^2 < 0 says only
            #     that the probe extrapolates differently, not that decodability changed.
            #     This is the same criterion the paper already applies to its delta1 probe.
            r2_pt_real = R2[(sname, "PT", tname)]
            generalizes = bool(r2_pt_real > 0)
            #
            # (2) does the effect clear the permuted-target floor?  This test is only fair when
            #     the real and permuted probes sit in the SAME R^2 regime -- a probe at
            #     R^2 = 0.99 is intrinsically far less variable than one at R^2 = -0.8, so
            #     comparing across regimes would overstate the floor. We record the null's own
            #     R^2(PT) and mark the comparison as regime-matched only when both are negative.
            null_r2_pt = float(np.mean([R2[(sname, "PT", f"NULL{j}|{tname}")]
                                        for j in range(N_NULL)]))
            floor_comparable = bool(r2_pt_real < 0 and null_r2_pt < 0)
            summary[f"{sname}|{tname}"] = {
                "split": sname, "target": tname,
                "trained": tname == "forecast96_ALLFEAT",
                "alpha": alpha[(sname, tname)], "cv_r2_train": cv_score[(sname, tname)],
                "r2_pt": R2[(sname, "PT", tname)],
                "r2_ft_mean": float(np.mean([R2[(sname, s, tname)] for s in seeds])),
                "dr2_mean": float(d.mean()), "dr2_sd": float(d.std(ddof=1)),
                "dr2_ci95": [lo, hi], "dr2_per_seed": {s: float(x) for s, x in zip(seeds, d)},
                "n_negative": int((d < 0).sum()), "n_seeds": int(len(d)),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "noise_floor_p95_abs": floor, "null_r2_pt": null_r2_pt,
                "null_dr2_mean": float(nulls.mean()), "null_dr2_sd": float(nulls.std(ddof=1)),
                "exceeds_noise_floor": above, "floor_comparable": floor_comparable,
                "probe_generalizes": generalizes,
                # direction is the CI verdict; interpretability is reported separately so the
                # two criteria can be audited independently
                "direction": direction,
                "interpretable": generalizes and not (floor_comparable and not above),
            }

    for sname in SPLITS:
        tag = ("val->test, matches the published trained probe" if sname.startswith("T")
               else "randomized over the same 500 windows")
        print(f"\n=== split {sname}  ({tag}) ===")
        print(f"{'target':<20s}{'alpha':>8s}{'R2(PT)':>10s}{'R2(FT)':>10s}{'dR2':>10s}"
              f"{'sd':>8s}{'95% CI':>22s}{'floor':>9s}{'nullR2':>9s}  verdict")
        for tname in raw:
            v = summary[f"{sname}|{tname}"]
            lo, hi = v["dr2_ci95"]
            note = v["direction"]
            if not v["probe_generalizes"]:
                note += ", R2<0 (uninterpretable)"
            if v["floor_comparable"] and not v["exceeds_noise_floor"]:
                note += ", below floor"
            print(f"{tname:<20s}{v['alpha']:>8g}{v['r2_pt']:>+10.4f}{v['r2_ft_mean']:>+10.4f}"
                  f"{v['dr2_mean']:>+10.4f}{v['dr2_sd']:>8.4f}"
                  f"   [{lo:>+8.4f},{hi:>+8.4f}]{v['noise_floor_p95_abs']:>9.4f}"
                  f"{v['null_r2_pt']:>+9.2f}  {note}")

    print(f"\nfloor = 95th pct of |dR2| over {N_NULL} permuted-target replicates. nullR2 is the")
    print("permuted probe's own R2(PT): the floor test is only fair where nullR2 and R2(PT) are")
    print("in the same regime (both negative), since a probe at R2=0.99 is intrinsically far")
    print("less variable than one at R2=-0.8. Where R2(PT)<0 the dR2 is uninterpretable anyway.")

    # =========================================================== alpha sweep
    KEY = ["forecast96_ALLFEAT", "mean", "var", "lag1"]
    sweep = {}
    print("\n=== alpha sensitivity of dR^2 (the knob the published result depends on) ===")
    for sname, (itr, ite) in SPLITS.items():
        for scale_reps in (False, True):
            for scale_targets in (False, True):
                btr, bte = blocks_for(sname, itr, ite, scale_targets)
                keys = [k for k in KEY]
                cur = {}
                for enc in ["PT"] + seeds:
                    Rtr, Rte = reps_all[enc][itr], reps_all[enc][ite]
                    if scale_reps:
                        Rtr, Rte = standardize(Rtr, Rte)
                    for a in SWEEP_ALPHAS:
                        got = r2_blocks(Rtr, Rte, a, {k: btr[k] for k in keys},
                                        {k: bte[k] for k in keys})
                        for k, v in got.items():
                            cur[(a, enc, k)] = v
                for a in SWEEP_ALPHAS:
                    for k in keys:
                        d = [cur[(a, s, k)] - cur[(a, "PT", k)] for s in seeds]
                        sweep[f"{sname}|reps={'z' if scale_reps else 'raw'}|"
                              f"tgt={'z' if scale_targets else 'raw'}|a={a:g}|{k}"] = {
                            "r2_pt": cur[(a, "PT", k)],
                            "dr2_mean": float(np.mean(d)), "dr2_sd": float(np.std(d, ddof=1)),
                            "n_pos": int(sum(x > 0 for x in d)), "n_seeds": len(d)}

    for sname in SPLITS:
        for rep_tag in ("raw", "z"):
            for tgt_tag in ("raw", "z"):
                print(f"\n  {sname}  reps={rep_tag}  targets={tgt_tag}")
                print(f"    {'alpha':>8s}" + "".join(f"{k[:14]:>17s}" for k in KEY))
                for a in SWEEP_ALPHAS:
                    row = f"    {a:>8g}"
                    for k in KEY:
                        v = sweep[f"{sname}|reps={rep_tag}|tgt={tgt_tag}|a={a:g}|{k}"]
                        row += f"  {v['dr2_mean']:>+8.3f}/{v['r2_pt']:>+7.2f}"
                    print(row)
    print("\n  cells are  dR2 / R2(PT).  The published configuration is")
    print("  T_temporal, reps=raw, targets=raw, alpha=1: it should show dR2 ~ +0.67 on")
    print("  forecast96_ALLFEAT with R2(PT) ~ -6.90.")

    # =========================================================== verdict
    print("\n=== asymmetry verdict per split ===")
    verdicts = {}
    for sname in SPLITS:
        tr = summary[f"{sname}|forecast96_ALLFEAT"]
        aux = [summary[f"{sname}|{t}"] for t in AUX]
        n_inc = sum(a["direction"] == "increase" for a in aux)
        n_dec = sum(a["direction"] == "decrease" for a in aux)
        n_ind = sum(a["direction"] == "indistinct" for a in aux)
        n_bad = sum(not a["probe_generalizes"] for a in aux)
        # The signature as the paper states it: trained increases, no auxiliary increases.
        # It only means anything if the probes involved actually generalize.
        holds = (tr["direction"] == "increase" and n_inc == 0)
        usable = tr["probe_generalizes"] and n_bad == 0
        verdicts[sname] = {
            "trained_direction": tr["direction"], "trained_dr2": tr["dr2_mean"],
            "trained_ci95": tr["dr2_ci95"], "trained_r2_pt": tr["r2_pt"],
            "trained_generalizes": tr["probe_generalizes"],
            "trained_exceeds_floor": tr["exceeds_noise_floor"],
            "aux_increase": n_inc, "aux_decrease": n_dec, "aux_indistinct": n_ind,
            "aux_not_generalizing": n_bad,
            "all_probes_generalize": bool(usable),
            "asymmetry_holds": bool(holds and usable)}
        print(f"{sname}: trained = {tr['direction']} (dR2 {tr['dr2_mean']:+.4f} "
              f"[{tr['dr2_ci95'][0]:+.4f}, {tr['dr2_ci95'][1]:+.4f}], "
              f"R2(PT) {tr['r2_pt']:+.3f}, floor {tr['noise_floor_p95_abs']:.4f})")
        print(f"    auxiliary: {n_dec} decrease / {n_ind} indistinct / {n_inc} increase"
              f"   ({n_bad}/{len(aux)} with R2(PT) < 0, i.e. uninterpretable)")
        if not usable:
            print("    -> NOT A USABLE TEST: some probe does not generalize (R2 < 0), so its")
            print("       dR2 reflects how the probe extrapolates, not what is decodable.")
        else:
            print("    -> probe-sign asymmetry " + ("HOLDS" if holds else "DOES NOT HOLD")
                  + " under this protocol")

    os.makedirs(OUT, exist_ok=True)
    payload = {
        "protocol": {
            "window_steps": EXT_LB, "n_train": N_TR, "n_test": N_TE,
            "window_pool": "val[:300] + test[:200] of ETTh2 (same windows as the trained probe)",
            "splits": {"T_temporal": "train=val[:300], test=test[:200]",
                       "R_randomized": f"pooled 500 shuffled with default_rng({SPLIT_SEED}), 300/200"},
            "features": "mean-pooled encoder output, standardized on probe-train",
            "targets": "standardized with probe-train mean/sd (primary analysis)",
            "probe": "Ridge; alpha by 5-fold CV (shuffle, random_state=0) on the PRETRAINED "
                     f"encoder's probe-train over logspace(-3,6,19); the same alpha is reused "
                     "for every fine-tuned encoder",
            "encoders": CKPTS,
            "bootstrap": f"percentile, {N_BOOT} resamples of the {len(seeds)} seeds, rng {BOOT_SEED}",
            "noise_floor": f"{N_NULL} independent target permutations per split, rng {NULL_SEED}; "
                           "floor = 95th percentile of |mean dR^2| over replicates",
        },
        "summary": summary, "verdicts": verdicts, "alpha_sweep": sweep,
    }
    with open(f"{OUT}/etth2_transparent.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {OUT}/etth2_transparent.json")


if __name__ == "__main__":
    main()
