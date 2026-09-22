#!/usr/bin/env python3
"""Deterministic three-split window construction for M4-Monthly, shared by the gate and the runs.

WHY THIS FILE EXISTS. The Chronos/M4-Monthly cell was the one cell in this paper whose numbers had no
run record: an appendix reported ten seeds, a CKA trajectory and an 84.5% gate that no file under
results/ could produce, because the series were never retained and the selection was never written
down. Reconstructing it needs three things pinned -- WHICH series, WHICH windows, and WHICH split each
window belongs to -- and pinned in ONE place, because the gate (scripts/gate_chronos_m4.py) and the
fine-tuning runs (scripts/chronos_m4_corrected.py) must see the same windows or the gate's denominator
stops being comparable to the runs' numerator. Everything below is deterministic: no RNG except the
train-window subsample, which takes the run's own seed.

THE SPLIT, AND WHY IT IS NOT THE ONE THE OLD SCRIPT USED. scripts/finetune_chronos_m4.py built one
pool of sliding windows over each series' whole history, subsampled it at random, and then cut the
first 80% off as "train" and the rest as "val". After a random subsample a positional cut is not a
chronological split: neighbouring windows overlap in 95 of their 96 context steps, so near-duplicates
land on both sides, and early stopping then selects on windows the model effectively trained on. It
also scored its gate on the TEST windows, where this paper's protocol requires the selection split.
Both are fixed here, by splitting each series in time before any window is built:

    history[0 : L-114]     -> TRAIN windows      (sliding, lookback 96 -> horizon 18)
    history[L-114 : L]     -> SELECTION window   (context history[L-114:L-18], target history[L-18:L])
    official M4 test       -> HELD-OUT window    (context history[L-96:L],     target test[0:18])

so a target used for training stops 114 steps before the history ends, a target used for early
stopping and for the gate is the final 18 observations of the history, and a target used for any
reported outcome is an official M4-Monthly test value the model never saw in any form.

ONE OVERLAP IS UNAVOIDABLE AND IS DISCLOSED RATHER THAN HIDDEN. The held-out context is the last 96
observations of the history, which CONTAINS the 18 values used as the selection target. M4's test set
is defined as the 18 steps immediately following the history, so the only way to avoid that overlap is
to discard M4's official test set and invent our own -- a worse trade. The overlap is in the INPUTS:
no held-out target is ever a training or selection target, which is the property early stopping can
otherwise launder. It is the same rolling-origin arrangement the M4 competition itself used.

WHY min_len = 228 = 2*(96+18). A series must supply all three splits, and the smallest history that
does is one full (lookback + horizon) block for training plus one for selection. 21,936 of M4's 48,000
monthly series qualify, so the constraint costs nothing but has to be stated: the cell is about M4's
LONGER monthly series, not about M4-Monthly as a whole.
"""
import numpy as np

LOOKBACK = 96
HORIZON = 18
NUM_SERIES = 200
MIN_LEN = 2 * (LOOKBACK + HORIZON)          # 228; see the module docstring
TRAIN_CSV = "data/forecasting/M4_Monthly_train.csv"
TEST_CSV = "data/forecasting/M4_Monthly_test.csv"

# Identical to chronos_mse_finetune.MIN_CONTEXT_STD and for the same reason: the protocol z-scores
# every window by its own context std, and a flat context divides by ~0. On this cell it drops 27 of
# 57,354 candidate TRAIN contexts and no selection or held-out context at all -- reported by
# build_splits() in the info dict rather than left implicit, because a silent filter reads as none.
MIN_CONTEXT_STD = 1e-8


def load_series(train_csv=TRAIN_CSV, test_csv=TEST_CSV, n_series=NUM_SERIES, min_len=MIN_LEN):
    """The first `n_series` M4-Monthly series, in FILE ORDER, with a history of at least `min_len`.

    File order, not a random draw: a random selection would need its own seed recorded, and a reader
    checking the manifest hash could still not tell which 200 of 48,000 series we used without
    re-implementing our RNG. `M1, M2, M3, M6, M9, ...` is checkable by eye against the CSV.
    """
    import pandas as pd
    tr = pd.read_csv(train_csv, index_col=0)
    te = pd.read_csv(test_csv, index_col=0)
    out = []
    for idx in tr.index:
        hist = tr.loc[idx].dropna().values.astype(np.float64)
        if len(hist) < min_len:
            continue
        test = te.loc[idx].dropna().values.astype(np.float64)
        if len(test) < HORIZON:
            continue
        out.append((str(idx), hist, test[:HORIZON]))
        if len(out) >= n_series:
            break
    if len(out) < n_series:
        raise RuntimeError(f"only {len(out)} of {n_series} series meet min_len={min_len}")
    return out


def _windows(seg, lookback, horizon):
    n = len(seg) - lookback - horizon + 1
    if n <= 0:
        return np.empty((0, lookback)), np.empty((0, horizon))
    ctx = np.array([seg[i:i + lookback] for i in range(n)])
    tgt = np.array([seg[i + lookback:i + lookback + horizon] for i in range(n)])
    return ctx, tgt


def build_splits(series=None, lookback=LOOKBACK, horizon=HORIZON, max_train=None, seed=42):
    """(train, selection, heldout, info) window pairs under the split in the module docstring.

    `max_train` subsamples the TRAIN pool only, with RandomState(seed) -- the one place a seed enters
    the data, mirroring build_windows() in chronos_mse_finetune.py so the two arms of the Chronos
    backbone draw their training windows the same way.
    """
    if series is None:
        series = load_series()
    block = lookback + horizon
    ctx_tr, tgt_tr, ctx_sel, tgt_sel, ctx_ho, tgt_ho = [], [], [], [], [], []
    for _, hist, test in series:
        c, t = _windows(hist[:-block], lookback, horizon)
        if len(c):
            ctx_tr.append(c)
            tgt_tr.append(t)
        ctx_sel.append(hist[-block:-horizon])
        tgt_sel.append(hist[-horizon:])
        ctx_ho.append(hist[-lookback:])
        tgt_ho.append(test)
    ctx_tr = np.concatenate(ctx_tr)
    tgt_tr = np.concatenate(tgt_tr)
    avail = len(ctx_tr)

    keep = ctx_tr.std(axis=1) > MIN_CONTEXT_STD
    n_dropped = int((~keep).sum())
    ctx_tr, tgt_tr = ctx_tr[keep], tgt_tr[keep]

    capped = max_train is not None and len(ctx_tr) > max_train
    if capped:
        idx = np.random.RandomState(seed).choice(len(ctx_tr), max_train, replace=False)
        ctx_tr, tgt_tr = ctx_tr[idx], tgt_tr[idx]

    info = dict(n_series=len(series), series_ids=[s[0] for s in series],
                lookback=lookback, horizon=horizon, min_len=MIN_LEN,
                train_windows_available=avail, train_windows_used=int(len(ctx_tr)),
                degenerate_train_contexts_dropped=n_dropped,
                train_subsampled=bool(capped), train_seed=seed,
                n_selection_windows=len(ctx_sel), n_heldout_windows=len(ctx_ho),
                selection_target="final 18 observations of each training history",
                heldout_target="official M4-Monthly test values")
    return ((ctx_tr, tgt_tr),
            (np.array(ctx_sel), np.array(tgt_sel)),
            (np.array(ctx_ho), np.array(tgt_ho)),
            info)


if __name__ == "__main__":
    import json
    tr, sel, ho, info = build_splits(max_train=500)
    print(json.dumps({k: v for k, v in info.items() if k != "series_ids"}, indent=2))
    print("series_ids[:8] =", info["series_ids"][:8], "... last =", info["series_ids"][-1])
    for name, (c, t) in (("train", tr), ("selection", sel), ("heldout", ho)):
        print(f"  {name:10s} ctx {c.shape} tgt {t.shape}")
