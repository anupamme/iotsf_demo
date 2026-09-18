#!/usr/bin/env python3
"""Six representation-change metrics on the encoders that retained checkpoints.

WHY THIS EXISTS. The paper's negative reading of drift rests on one similarity measure, linear CKA,
so a reader is entitled to ask whether "drift does not order the outcome" is a fact about
representations or a fact about CKA. This computes five further measures on the same encoders and the
same windows and asks two questions of them: do they agree with each other, and does any of them
order held-out forgetting where CKA does not.

WHAT IT CANNOT DO, STATED UP FRONT. Only n=10k runs retained encoder checkpoints. The 22-cell Moirai
matrix in the body is n=500/1000 and kept none, so this battery CANNOT re-run the cross-cell
correlation under a different metric -- that would need the matrix retrained, on different hardware,
which would perturb every number in the paper. What it can do is test metric-specificity on 59
encoders spanning three datasets and two freezing regimes. The appendix says so in those words.

THE SELF-CHECK THAT MATTERS. Each run's JSON already stores `final_cka`, computed during training on
the first 300 validation eval windows with the encoder output mean-pooled. This script rebuilds those
windows and that pooling and recomputes linear CKA, so every other metric is known to be measured on
the representations the paper's CKA column actually describes. A battery that quietly used different
windows would produce six internally consistent numbers that answer a different question.

  MISSING-KEY ASSERTION. load_state_dict(strict=False) silently loads NOTHING when the key
  namespaces disagree, which returns CKA ~= 1.0 and reads as "no drift". That defect has occurred in
  this codebase before, so the load is asserted rather than trusted.

USAGE
  .venv-probe/bin/python scripts/drift_metric_battery.py --compute [--dev cpu|mps] [--groups g1,g2]
  .venv-probe/bin/python scripts/drift_metric_battery.py --report

  DRIFT_METRIC_OUT  output JSON path (default results/v52_drift_metrics/battery.json)
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("DRIFT_METRIC_OUT", os.path.join(ROOT, "results/v52_drift_metrics/battery.json"))
TABLE = os.path.join(ROOT, "paper_8/tables/drift_metrics.tex")

LOOKBACK, HORIZON, EVAL_LIMIT, REP_SAMPLES = 96, 96, 300, 500

# The dataset is NOT recorded in the run JSONs -- it is implied by the output directory, and the
# launcher scripts are the only record of which. Declared here explicitly, with the launcher named,
# so a wrong pairing is a visible edit rather than an inferred default.
GROUPS = {
    # tag: (checkpoint glob, dataset csv, condition, launcher that produced it)
    "etth2_v19_cuda": ("results/v19_cuda_etth2_n10k/seed*/best_encoder.pt",
                       "ETTh2.csv", "B", "run_v19 (CUDA)"),
    "etth2_v18_mps": ("results/v18_mps_deterministic_n10k/seed*/best_encoder.pt",
                      "ETTh2.csv", "B", "run_v18 (MPS, deterministic)"),
    "etth2_v17r2": ("results/v17r2_etth2_n10k_es_mlp/seed*/best_encoder.pt",
                    "ETTh2.csv", "B", "scripts/run_v17r2_n10k_es_mlp.sh"),
    "ettm2_v19_cuda": ("results/v19_cuda_ettm2_n10k/seed*/best_encoder.pt",
                       "ETTm2.csv", "B", "run_v19 (CUDA)"),
    "etth1_v21": ("results/v21_etth1_n10k/seed*/best_encoder.pt",
                  "ETTh1.csv", "B", "scripts/run_overnight.sh"),
    "etth2_v49_N3": ("results/v49_layerunfreeze_10seed/N3_seed*/best_encoder.pt",
                     "ETTh2.csv", "D-top3", "scripts/run_layerunfreeze_10seed_mps.sh"),
    "etth2_v49_N6": ("results/v49_layerunfreeze_10seed/N6_seed*/best_encoder.pt",
                     "ETTh2.csv", "D-all6", "scripts/run_layerunfreeze_10seed_mps.sh"),
}

METRICS = ("linear_cka", "linear_cka_flat", "rbf_cka", "svcca", "procrustes", "mean_cosine",
           "erank_ratio")
PRETTY = {
    "linear_cka": "Linear CKA",
    "linear_cka_flat": "Linear CKA (token)",
    "rbf_cka": "RBF CKA",
    "svcca": "SVCCA",
    "procrustes": "Procrustes sim.",
    "mean_cosine": "Mean cosine",
    "erank_ratio": "Eff.\\ rank ratio",
}


# ------------------------------------------------------------------------------------------------
# Metrics. Every one takes two (N, D) matrices with ROW CORRESPONDENCE: row i of each is the same
# window through the pre-trained and the fine-tuned encoder. Losing that correspondence silently
# turns cosine and Procrustes into noise, so the callers never reorder.
# ------------------------------------------------------------------------------------------------
def _center(X):
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y):
    """Kornblith et al. linear CKA. Reproduces the paper's stored `final_cka`."""
    X, Y = _center(X), _center(Y)
    XtX, YtY = X.T @ X, Y.T @ Y
    num = np.trace(XtX @ YtY)
    den = np.sqrt(np.trace(XtX @ XtX) * np.trace(YtY @ YtY))
    return float(num / den) if den > 1e-10 else 0.0


def _rbf_gram(X, frac):
    """RBF Gram matrix with bandwidth set from the median pairwise distance of X itself.

    The bandwidth is a free parameter of RBF CKA and the value chosen decides how much of the answer
    is the kernel's. Setting it per matrix from the median distance is the standard heuristic and is
    reported (`rbf_frac`) so the choice is visible rather than implicit.
    """
    sq = np.sum(X ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * (X @ X.T), 0.0)
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    sigma2 = frac * med if med > 0 else 1.0
    return np.exp(-d2 / (2.0 * sigma2))


def rbf_cka(X, Y, frac=0.5):
    """CKA with an RBF kernel: sensitive to nonlinear structure linear CKA cannot see."""
    n = len(X)
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ _rbf_gram(_center(X), frac) @ H
    L = H @ _rbf_gram(_center(Y), frac) @ H
    num = float(np.sum(K * L))
    den = float(np.sqrt(np.sum(K * K) * np.sum(L * L)))
    return num / den if den > 1e-10 else 0.0


def svcca(X, Y, var_keep=0.99, max_k=20):
    """Mean canonical correlation between the top principal subspaces (Raghu et al.).

    Unlike CKA this is invariant to invertible linear maps, so it asks whether the SAME INFORMATION
    survives in a possibly rotated and rescaled basis -- the objection that a representation can
    drift far and still carry the task in a transformed basis. Components are kept to `var_keep` of
    the variance, capped at `max_k`, and the kept counts are recorded.
    """
    def reduce(M):
        M = _center(M)
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        if s.sum() <= 0:
            return U[:, :1], 1
        keep = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), var_keep) + 1)
        keep = max(1, min(keep, max_k, M.shape[1]))
        return U[:, :keep], keep

    Qx, kx = reduce(X)
    Qy, ky = reduce(Y)
    sv = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return float(np.mean(np.clip(sv, 0.0, 1.0))), kx, ky


def procrustes_sim(X, Y):
    """Orthogonal-Procrustes similarity in [0, 1]: 1 iff Y is X up to rotation and global scale.

    With both matrices centred and scaled to unit Frobenius norm,
    min_Q ||X - Y Q||_F^2 = 2 - 2 ||Y^T X||_nuc, so the nuclear norm IS the similarity.
    """
    X, Y = _center(X), _center(Y)
    nx, ny = np.linalg.norm(X), np.linalg.norm(Y)
    if nx < 1e-12 or ny < 1e-12:
        return 0.0
    return float(np.clip(np.linalg.svd((Y / ny).T @ (X / nx), compute_uv=False).sum(), 0.0, 1.0))


def mean_cosine(X, Y):
    """Mean cosine between corresponding rows, WITHOUT centring or alignment.

    Deliberately the least invariant measure in the battery: a pure rotation of the representation
    destroys it while leaving CKA, SVCCA and Procrustes at 1. It is here as the opposite extreme, so
    "the metrics agree" is a claim spanning an invariance range rather than a family of near-clones.
    """
    a = np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1)
    ok = a > 1e-12
    return float(np.mean(np.sum(X[ok] * Y[ok], axis=1) / a[ok])) if ok.any() else 0.0


def effective_rank(X):
    """exp(entropy of the normalised singular-value spectrum): how many directions are in use."""
    s = np.linalg.svd(_center(X), compute_uv=False)
    if s.sum() <= 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def all_metrics(pooled_pt, pooled_ft, flat_pt, flat_ft):
    sv, kx, ky = svcca(pooled_pt, pooled_ft)
    erank_pt, erank_ft = effective_rank(pooled_pt), effective_rank(pooled_ft)
    return dict(
        linear_cka=linear_cka(pooled_pt, pooled_ft),
        linear_cka_flat=linear_cka(flat_pt, flat_ft),
        rbf_cka=rbf_cka(pooled_pt, pooled_ft),
        svcca=sv, svcca_k=(kx, ky),
        procrustes=procrustes_sim(pooled_pt, pooled_ft),
        mean_cosine=mean_cosine(pooled_pt, pooled_ft),
        erank_pt=erank_pt, erank_ft=erank_ft,
        erank_ratio=(erank_ft / erank_pt) if erank_pt > 0 else 0.0,
    )


# ------------------------------------------------------------------------------------------------
# Compute
# ------------------------------------------------------------------------------------------------
def eval_windows(csv_name):
    """The first 300 validation eval windows, exactly as finetune_forecasting.py builds them.

    ext_lb = lookback + horizon (NOT 2*lookback) and the values are raw, not normalised: the model
    z-scores internally. Both details are load-bearing -- getting either wrong yields a valid-looking
    CKA on windows the paper's CKA was never measured on.
    """
    from src.data.forecasting_loader import get_forecasting_loader
    loader = get_forecasting_loader(os.path.join(ROOT, "data/forecasting", csv_name),
                                    lookback_window=LOOKBACK, forecast_horizon=HORIZON, features="M")
    _, val_df, _ = loader.get_splits()
    vals = val_df[loader.FEATURE_COLUMNS].values
    ext = LOOKBACK + HORIZON
    total = ext + HORIZON
    X = np.array([vals[i:i + ext] for i in range(len(vals) - total + 1)])
    return X[:EVAL_LIMIT], len(loader.FEATURE_COLUMNS)


def run_json_for(ckpt):
    """The condition JSON beside a checkpoint, for `final_cka` and `forgetting_pct`."""
    hits = sorted(glob.glob(os.path.join(os.path.dirname(ckpt), "condition_*_h96_s*.json")))
    if len(hits) != 1:
        raise AssertionError(f"{ckpt}: expected 1 condition JSON beside it, found {len(hits)}")
    return json.load(open(hits[0])), os.path.relpath(hits[0], ROOT)


def compute(tags, dev):
    import torch
    from src.models.moirai_detector import MoiraiAnomalyDetector, _apply_uni2ts_gradient_patch
    from scripts.finetune_forecasting import extract_representations

    torch.manual_seed(0)
    np.random.seed(0)
    _apply_uni2ts_gradient_patch()
    if dev == "mps":
        from scripts.finetune_forecasting import _patch_packed_scaler_for_mps
        _patch_packed_scaler_for_mps()

    records, cache = [], {}
    for tag in tags:
        pattern, csv_name, cond, launcher = GROUPS[tag]
        ckpts = sorted(glob.glob(os.path.join(ROOT, pattern)))
        assert ckpts, f"{tag}: no checkpoints matched {pattern}"

        if csv_name not in cache:
            X, n_feat = eval_windows(csv_name)
            det = MoiraiAnomalyDetector(model_size="small", context_length=LOOKBACK,
                                        prediction_length=HORIZON, target_dim=n_feat,
                                        num_samples=20, device=dev)
            det.initialize()
            model = det.model
            data = torch.from_numpy(X).float()
            pt_pooled = extract_representations(model, data, None, device=dev,
                                                max_samples=REP_SAMPLES)
            pt_seq = extract_representations(model, data, None, device=dev,
                                            max_samples=REP_SAMPLES, keep_sequence=True)
            # extract_representations swallows forward-pass exceptions, so an all-failed extraction
            # returns an empty array and every metric below would be computed on nothing.
            assert len(pt_pooled) and len(pt_seq), (
                f"{csv_name}: empty pre-trained representations "
                f"(pooled {pt_pooled.shape}, seq {pt_seq.shape}) -- every forward pass failed")
            cache[csv_name] = (model, data, pt_pooled, pt_seq.reshape(-1, pt_seq.shape[-1]),
                               {k: v.clone() for k, v in model.state_dict().items()})
            print(f"[{csv_name}] windows={len(X)} feat={n_feat} pooled={pt_pooled.shape} "
                  f"flat={pt_seq.shape}")
        model, data, pt_pooled, pt_flat, pretrained_sd = cache[csv_name]

        for ck in ckpts:
            seed = int(re.search(r"seed(\d+)", ck).group(1))
            meta, json_rel = run_json_for(ck)
            sd = torch.load(ck, map_location=dev, weights_only=False)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            assert not missing and not unexpected, (
                f"{ck}: {len(missing)} missing / {len(unexpected)} unexpected keys; "
                f"strict=False would have loaded nothing and reported no drift")
            ft_pooled = extract_representations(model, data, None, device=dev,
                                               max_samples=REP_SAMPLES)
            ft_seq = extract_representations(model, data, None, device=dev,
                                            max_samples=REP_SAMPLES, keep_sequence=True)
            assert ft_pooled.shape == pt_pooled.shape, (
                f"{ck}: fine-tuned reps {ft_pooled.shape} != pre-trained {pt_pooled.shape}; "
                f"row correspondence is what every metric here assumes")
            m = all_metrics(pt_pooled, ft_pooled, pt_flat, ft_seq.reshape(-1, ft_seq.shape[-1]))
            model.load_state_dict(pretrained_sd)          # restore, so the next load starts clean
            rec = dict(tag=tag, dataset=csv_name.replace(".csv", ""), condition=cond, seed=seed,
                       ckpt=os.path.relpath(ck, ROOT), run_json=json_rel, launcher=launcher,
                       stored_cka=meta.get("final_cka"), stored_drift=meta.get("final_weight_drift"),
                       forgetting_pct=meta.get("forgetting_pct"), **m)
            records.append(rec)
            print(f"  {tag} s{seed}: cka {m['linear_cka']:.3f} (stored {rec['stored_cka']:.3f}, "
                  f"d={abs(m['linear_cka'] - rec['stored_cka']):.3f})  rbf {m['rbf_cka']:.3f}  "
                  f"svcca {m['svcca']:.3f}  proc {m['procrustes']:.3f}  "
                  f"cos {m['mean_cosine']:.3f}  erank {m['erank_ratio']:.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = dict(records=records, lookback=LOOKBACK, horizon=HORIZON, eval_limit=EVAL_LIMIT,
                   rep_samples=REP_SAMPLES, device=dev, metrics=list(METRICS),
                   groups={t: GROUPS[t] for t in tags})
    json.dump(payload, open(OUT, "w"), indent=1)
    print(f"\nwrote {len(records)} records -> {os.path.relpath(OUT, ROOT)}")
    return payload


# ------------------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------------------
def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return float(r.correlation), float(r.pvalue)


def report(payload):
    recs = payload["records"]
    print(f"\n=== {len(recs)} encoders, {len({r['dataset'] for r in recs})} datasets, "
          f"{len({r['condition'] for r in recs})} conditions ===")

    # Self-check first: if the recomputed CKA does not match the stored column, nothing below is
    # about the paper's encoders and the run is void.
    d = [abs(r["linear_cka"] - r["stored_cka"]) for r in recs if r["stored_cka"] is not None]
    print(f"self-check |recomputed - stored| linear CKA: max {max(d):.4f}  mean {np.mean(d):.4f}")

    # `forgetting_pct` in these run records is the SELECTION-SPLIT figure: these runs stored no
    # zero-shot test MSE, so a held-out forgetting cannot be reconstructed from them. Since
    # Sec. heldout shows same-split scoring can invert the sign of the intervention contrast, this
    # arm is a metric comparison and not a held-out claim, and the appendix says so.
    print("\nper metric: range over encoders, and rank correlation with SELECTION-SPLIT forgetting")
    have_f = [r for r in recs if r.get("forgetting_pct") is not None]
    for m in METRICS:
        v = np.array([r[m] for r in recs], float)
        rho, p = spearman([r[m] for r in have_f], [r["forgetting_pct"] for r in have_f])
        gr = group_rhos(recs, m)
        print(f"  {PRETTY[m]:<20s} [{v.min():+.3f}, {v.max():+.3f}]   "
              f"rho(forgetting) pooled {rho:+.3f} (p={p:.3f}, n={len(have_f)})  "
              f"per group [{min(gr.values()):+.2f}, {max(gr.values()):+.2f}] over {len(gr)}")

    print("\ninter-metric Spearman (do the measures order the encoders alike?)")
    hdr = "".join(f"{m[:9]:>11s}" for m in METRICS)
    print(f"{'':<20s}{hdr}")
    for a in METRICS:
        row = "".join(f"{spearman([r[a] for r in recs], [r[b] for r in recs])[0]:>11.3f}"
                      for b in METRICS)
        print(f"  {PRETTY[a]:<18s}{row}")

    print("\nwithin dataset+condition group (seed noise only; cross-group is a dataset effect)")
    for tag in sorted({r["tag"] for r in recs}):
        g = [r for r in recs if r["tag"] == tag]
        gf = [r for r in g if r.get("forgetting_pct") is not None]
        line = f"  {tag:<16s} n={len(g):<3d}"
        for m in ("linear_cka", "rbf_cka", "svcca", "procrustes", "mean_cosine"):
            line += f" {m[:4]}={np.mean([r[m] for r in g]):+.3f}"
        if len(gf) >= 4:
            rho, _ = spearman([r["linear_cka"] for r in gf], [r["forgetting_pct"] for r in gf])
            line += f"  rho_cka/forg={rho:+.3f}"
        print(line)


MIN_GROUP = 5   # groups smaller than this carry no rank information and are named, not silently used


def group_rhos(recs, metric):
    """Per-group Spearman of `metric` against selection-split forgetting.

    Pooling the 59 encoders would correlate dataset identity as much as drift -- the same
    group-artefact objection Appendix~crosscell makes about the pooled CKA row. The per-group values
    are the honest reading, so both are reported and the pooled one is labelled descriptive.
    """
    out = {}
    for tag in sorted({r["tag"] for r in recs}):
        g = [r for r in recs if r["tag"] == tag and r.get("forgetting_pct") is not None]
        if len(g) >= MIN_GROUP:
            out[tag] = spearman([r[metric] for r in g], [r["forgetting_pct"] for r in g])[0]
    return out


def emit_latex(payload):
    recs = payload["records"]
    have_f = [r for r in recs if r.get("forgetting_pct") is not None]
    rows = []
    for m in METRICS:
        v = np.array([r[m] for r in recs], float)
        rho, _ = spearman([r[m] for r in have_f], [r["forgetting_pct"] for r in have_f])
        rho_cka, _ = spearman([r[m] for r in recs], [r["linear_cka"] for r in recs])
        gr = group_rhos(recs, m)
        rows.append(f"{PRETTY[m]} & ${v.min():+.3f}$ & ${v.max():+.3f}$ & ${v.mean():+.3f}$ & "
                    f"${rho_cka:+.3f}$ & ${rho:+.3f}$ & "
                    f"$[{min(gr.values()):+.2f},{max(gr.values()):+.2f}]$ \\\\")
    body = "\n".join(rows)
    tex = (
        "% GENERATED by scripts/drift_metric_battery.py --report -- do not hand-edit.\n"
        "\\centering\n\\small\n"
        "\\begin{tabular}{lrrrrrc}\n\\toprule\n"
        " & \\multicolumn{3}{c}{over 59 encoders} & & "
        "\\multicolumn{2}{c}{$\\rho$ with forgetting} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}\n"
        "Measure & min & max & mean & $\\rho$ with CKA & pooled & per group \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    open(TABLE, "w").write(tex)
    print(f"\nwrote {os.path.relpath(TABLE, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compute", action="store_true", help="extract representations and score")
    ap.add_argument("--report", action="store_true", help="report/emit from the stored JSON")
    ap.add_argument("--dev", default="cpu", help="torch device for --compute (cpu or mps)")
    ap.add_argument("--groups", default=",".join(GROUPS), help="comma-separated group tags")
    args = ap.parse_args()
    if not (args.compute or args.report):
        ap.error("pass --compute, --report, or both")

    tags = [t.strip() for t in args.groups.split(",") if t.strip()]
    bad = [t for t in tags if t not in GROUPS]
    if bad:
        ap.error(f"unknown group(s) {bad}; declared: {list(GROUPS)}")

    payload = compute(tags, args.dev) if args.compute else json.load(open(OUT))
    if args.report:
        report(payload)
        emit_latex(payload)


if __name__ == "__main__":
    main()
