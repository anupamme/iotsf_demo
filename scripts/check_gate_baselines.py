#!/usr/bin/env python3
"""
Correctness checks for the gate's baseline ladder, run before any rung is used as a denominator.

WHY THIS EXISTS. Every rung in gate_baselines.py is a DENOMINATOR in R2_task = 1 - MSE_zs/MSE_base, so
a rung that is silently weak than it claims to be does not produce an error -- it produces a larger
denominator, a larger R2_task, and a checkpoint that looks better than it is. The failure mode is
directional and flattering, which is the worst kind to leave untested. Three of the eight rungs
(persistence, dlinear, mlp) were added on 2026-09-18 and are checked here against properties that hold
by construction rather than against previously-printed numbers, because a check written against the
paper's own output is circular -- that mistake has already been made once in this project
(layerwise_cka_from_ckpt.py).

What is checked, and what each check would catch:

  1. persistence on a flat series is exact                -> an off-by-one in which step is repeated
  2. persistence == seasonal_naive at season 1            -> the two rungs having silently fused
  3. _moving_avg matches a naive per-index reference      -> the cumulative-sum trick's padding
  4. _moving_avg preserves a constant                     -> edge replication dragging the trend
  5. dlinear recovers an exact [trend|seasonal] map       -> the feature block order or the solve
  6. the [M ; I-M] operator has rank exactly lookback     -> the claim that the decomposition is a
                                                             reparametrisation, which is why the
                                                             shipped default pools across channels
  7. individual-dlinear train MSE == ar train MSE         -> that same claim, measured
  8. shared-dlinear train MSE >= ar train MSE             -> the pooled solve not actually pooling
  9. dlinear has no cross-channel inputs                  -> cross-channel leakage in the fit
 10. mlp output shape and determinism under a fixed seed  -> a reshape transposing (h, D)
 11. mlp records a per-channel penalty and its cap        -> a silent cap reading as full coverage
 12. every NAMES entry dispatches and returns its shape   -> a rung added to NAMES but not to predict()

Run:  .venv12/bin/python scripts/check_gate_baselines.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_baselines as gb                                            # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def windows(rng, n, lb, h, d):
    """(ctx, tgt) on an already-normalised scale, the contract every rung is written to."""
    series = np.cumsum(rng.normal(size=(n + lb + h, d)), axis=0)
    series = (series - series.mean(0)) / (series.std(0) + 1e-8)
    ctx = np.stack([series[i:i + lb] for i in range(n)])
    tgt = np.stack([series[i + lb:i + lb + h] for i in range(n)])
    return ctx, tgt


def naive_moving_avg(x, k):
    """Reference implementation: replicate-pad, then average each length-k span with a Python loop."""
    left, right = (k - 1) // 2, k // 2
    out = np.empty_like(x)
    for i in range(x.shape[1]):
        idx = [min(max(j, 0), x.shape[1] - 1) for j in range(i - left, i + right + 1)]
        out[:, i, :] = x[:, idx, :].mean(axis=1)
    return out


def main():
    rng = np.random.default_rng(0)
    lb, h, d = 96, 24, 3
    ctx, tgt = windows(rng, 400, lb, h, d)
    ctx_tr, tgt_tr = ctx[:300], tgt[:300]
    ctx_ev, tgt_ev = ctx[300:], tgt[300:]

    print("persistence")
    flat = np.tile(np.arange(1.0, d + 1), (5, lb, 1))
    p, _ = gb.persistence(flat, h)
    check("flat series reproduced exactly", np.allclose(p, np.tile(np.arange(1.0, d + 1), (5, h, 1))))
    p, _ = gb.persistence(ctx_ev, h)
    check("repeats the LAST context step, not the first",
          np.allclose(p[:, 0, :], ctx_ev[:, -1, :]) and np.allclose(p[:, -1, :], ctx_ev[:, -1, :]))
    sn, _ = gb.seasonal_naive(ctx_ev, lb, h, 1)
    check("equals seasonal_naive at season 1 and differs at the real period",
          np.allclose(p, sn) and not np.allclose(p, gb.seasonal_naive(ctx_ev, lb, h, 24)[0]))

    print("moving average (DLinear series_decomp)")
    for k in (3, 4, 25):
        got, want = gb._moving_avg(ctx_ev, k), naive_moving_avg(ctx_ev, k)
        check(f"matches naive reference at kernel {k}", np.allclose(got, want),
              f"max abs diff {np.abs(got - want).max():.2e}")
    const = np.full((4, lb, d), 2.5)
    check("preserves a constant series (edge replication, not zero padding)",
          np.allclose(gb._moving_avg(const, 25), const))

    print("dlinear")
    F = gb._dlinear_features(ctx_tr, lb, gb.DLINEAR_KERNEL)
    check("feature block is [trend | seasonal] and sums back to the window",
          F.shape[1] == 2 * lb
          and np.allclose(F[:, :lb, :] + F[:, lb:, :], ctx_tr[:, -lb:, :]))
    # A target that IS an exact linear function of the features must be recovered near-exactly.
    Wtrue = rng.normal(scale=0.1, size=(2 * lb, h))
    tgt_lin = np.stack([F[:, :, c] @ Wtrue for c in range(d)], axis=2)
    coefs, info = gb.fit_dlinear(ctx_tr, tgt_lin, lb, lam=1e-10)
    pred = gb.apply_dlinear(coefs, ctx_tr, lb, h)
    check("recovers an exact [trend|seasonal] linear map",
          float(np.mean((pred - tgt_lin) ** 2)) < 1e-12,
          f"train MSE {np.mean((pred - tgt_lin) ** 2):.2e}")
    check("reports its kernel and channel-independence",
          info["kernel"] == gb.DLINEAR_KERNEL and info["channel_independent"] is True, str(info))
    # THE DECOMPOSITION IS A REPARAMETRISATION. [M x | (I-M) x] is x through the stacked operator
    # [M ; I-M], whose rank is exactly `lookback` -- so individual-DLinear spans the SAME functions as
    # ar, and at a negligible penalty the two must fit the training windows equally. This is the check
    # that keeps the docstring's claim honest; if the rank were 2L the claim would be wrong.
    Mm = gb._moving_avg(np.eye(lb)[None, :, :], gb.DLINEAR_KERNEL)[0]
    stacked = np.vstack([Mm, np.eye(lb) - Mm])
    check("[trend | seasonal] operator has rank exactly lookback (not 2*lookback)",
          np.linalg.matrix_rank(stacked) == lb, f"rank {np.linalg.matrix_rank(stacked)} vs {lb}")
    dl_ind = np.mean((gb.apply_dlinear(
        gb.fit_dlinear(ctx_tr, tgt_tr, lb, lam=1e-8, individual=True)[0], ctx_tr, lb, h) - tgt_tr) ** 2)
    ar_tr = np.mean((gb.apply_ar(gb.fit_ar(ctx_tr, tgt_tr, lb, lam=1e-8),
                                 ctx_tr, lb, h) - tgt_tr) ** 2)
    check("individual variant matches ar on train (same function class)",
          abs(dl_ind - ar_tr) < 1e-6 * max(1.0, ar_tr),
          f"dlinear_individual {dl_ind:.8f} vs ar {ar_tr:.8f}")
    # The shipped default pools one map across channels, so it is NESTED INSIDE per-channel ar and
    # must fit the training windows no better. A shared map that beat ar would mean the pooled solve
    # is not actually shared.
    dl_shared = np.mean((gb.apply_dlinear(gb.fit_dlinear(ctx_tr, tgt_tr, lb)[0], ctx_tr, lb, h)
                         - tgt_tr) ** 2)
    check("default (channel-shared) fits train no better than per-channel ar",
          dl_shared >= ar_tr - 1e-9, f"dlinear_shared {dl_shared:.6f} vs ar {ar_tr:.6f}")
    coefs, dinfo = gb.fit_dlinear(ctx_tr, tgt_tr, lb)
    check("default fits ONE shared map, and says so in its info",
          len(coefs) == 1 and dinfo["channel_shared_weights"] and not dinfo["individual"], str(dinfo))
    check("individual variant fits one map per channel",
          len(gb.fit_dlinear(ctx_tr, tgt_tr, lb, individual=True)[0]) == d)
    # No cross-channel terms either way: perturbing channel 1's targets must not move channel 0's
    # predictions under the INDIVIDUAL variant. (The shared variant pools, so it legitimately does.)
    tgt_perturbed = tgt_tr.copy()
    tgt_perturbed[:, :, 1] += 5.0
    base = gb.apply_dlinear(gb.fit_dlinear(ctx_tr, tgt_tr, lb, individual=True)[0], ctx_ev, lb, h)
    pert = gb.apply_dlinear(gb.fit_dlinear(ctx_tr, tgt_perturbed, lb, individual=True)[0],
                            ctx_ev, lb, h)
    check("individual variant has no cross-channel leakage",
          np.allclose(base[:, :, 0], pert[:, :, 0]) and not np.allclose(base[:, :, 1],
                                                                       pert[:, :, 1]))

    print("mlp")
    # Small and 1-D so the fits are quick; the properties checked are structural, not accuracy.
    lb_s, h_s, d_s = 24, 6, 2
    c_tr, t_tr = windows(np.random.default_rng(1), 300, lb_s, h_s, d_s)
    c_ev, _ = windows(np.random.default_rng(2), 40, lb_s, h_s, d_s)
    models, minfo = gb.fit_mlp(c_tr, t_tr, lb_s, h_s, alphas=(1e-4, 1e0), seed=0)
    pred = gb.apply_mlp(models, c_ev, lb_s, h_s)
    check("output shape is (N, horizon, D)", pred.shape == (len(c_ev), h_s, d_s), str(pred.shape))
    m2, _ = gb.fit_mlp(c_tr, t_tr, lb_s, h_s, alphas=(1e-4, 1e0), seed=0)
    check("is deterministic at a fixed seed",
          np.allclose(pred, gb.apply_mlp(m2, c_ev, lb_s, h_s)))
    check("records one selected penalty per channel, the cap and the budget",
          len(minfo["alpha_per_channel"]) == d_s and "capped" in minfo
          and "channels_hitting_iter_budget" in minfo, str(minfo))
    check("selects per channel from the grid it was given",
          all(a in (1e-4, 1e0) for a in minfo["alpha_per_channel"]),
          str(minfo["alpha_per_channel"]))
    cap_models, cap_info = gb.fit_mlp(c_tr, t_tr, lb_s, h_s, alphas=(1e-4,), max_windows=50, seed=0)
    check("reports a cap when one binds", cap_info["capped"] and cap_info["n_train_windows"] == 50,
          f"n_train_windows={cap_info['n_train_windows']} avail={cap_info['avail_windows']}")

    print("dispatcher covers every declared rung")
    for name in gb.NAMES:
        try:
            pred, info = gb.predict(name, ctx_ev, lb, h, train=(ctx_tr, tgt_tr), dataset="etth1") \
                if name != "mlp" else \
                gb.predict(name, c_ev, lb_s, h_s, train=(c_tr, t_tr), dataset="etth1")
            want = (len(ctx_ev), h, d) if name != "mlp" else (len(c_ev), h_s, d_s)
            check(f"{name}: dispatches and returns {want}", pred.shape == want and
                  isinstance(info, dict), str(pred.shape))
        except Exception as e:                                          # noqa: BLE001
            check(f"{name}: dispatches", False, f"{type(e).__name__}: {e}")

    print(f"\n{len(FAILURES)} failure(s)" + (": " + ", ".join(FAILURES) if FAILURES else ""))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
