#!/usr/bin/env python3
"""
Generate task B for the positive control: a series whose correct forecast OPPOSES the persistence
prior every time-series foundation model carries.

WHY A SYNTHETIC TASK B AT ALL. The positive control has to construct a regime where fine-tuning
demonstrably destroys a pre-trained capability. Fine-tuning Moirai on another ETT series does not do
that -- that is the whole finding of this paper's main matrix, where B often HELPS. To destroy a
capability on purpose you need a target task whose optimal solution is incompatible with the source
task's, and no benchmark pair in the matrix supplies that. So we build one, and we build it to be
adversarial by construction rather than by luck.

THE CONSTRUCTION, AND WHY THIS ONE. The series is a smooth seasonal shape whose PHASE INVERTS every
period:

    x[j, t] = a[j] * sign(t) * s[j](t mod P) + sigma[j] * noise,     sign(t) = (-1) ** (t // P)

Two properties, and the control needs both:

  LEARNABLE. Given at least one full period of context the map is deterministic: read the phase, read
  the current sign, negate it. A linear model on the last P steps can represent it exactly. If the
  target task were not learnable, a collapse on task A would be weight thrashing rather than
  forgetting, and the pre-registration makes that a named validity condition.

  ANTI-PERSISTENT. Repeating the last period -- seasonal-naive, which is the prior every TSFM is
  pre-trained into and the strongest admissible rung on task A's own cell -- returns exactly the
  NEGATION of the truth, the worst possible answer at that amplitude. Persistence fails the same way.
  So learning task B requires overwriting the prior, not merely extending it.

The two are checked, not asserted: main() scores seasonal-naive, its negation, and the noise-floor
oracle on the generated array, and exits nonzero unless seasonal-naive is worse than its own negation
by a wide margin. A generator that silently produced a persistence-friendly series would give the
positive control a target task that cannot destroy anything, and the run would look like outcome 3
("destruction not achieved") for a reason that had nothing to do with the model.

SHAPE. 7 columns and a `date` index under ETTh1's column names, because
src/data/forecasting_loader.py validates against FEATURE_COLUMNS and we want task B to travel through
the IDENTICAL loader, the identical 12/4/4 chronological split and the identical window construction
as every other cell in the paper. Not one line of new data-path code is involved in the comparison.

PROVENANCE. The CSV is gitignored (*.csv) and is NOT a benchmark file: it is not in
results/data_manifest.json and does not belong there, because a reader does not download it -- they
regenerate it. What is committed is this script, the seed, and results/synthetic_task_b.json carrying
the SHA-256 the pre-registration pins. Re-running this script must reproduce that hash byte for byte.

Usage:  .venv12/bin/python scripts/make_conflicting_series.py
        .venv12/bin/python scripts/make_conflicting_series.py --check   # verify, write nothing
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data/synthetic/antipersistent_v1.csv"
OUT_JSON = ROOT / "results/synthetic_task_b.json"

SEED = 20260921        # fixed once; changing it changes the hash the registration pins
T = 17420             # ETTh2's length, so splits and sample budgets are comparable
# The sign flips every PERIOD steps, so x(t + PERIOD) = -x(t) EXACTLY (up to noise). PERIOD is set to
# the evaluation horizon, not to a calendar day, and the difference is load-bearing: at PERIOD=96 with
# h=192 the target spans two half-periods of opposite sign, seasonal-naive is wrong on the first half
# and right on the second, and the conflict averages away to nothing. The first version of this file
# did exactly that and the assert below caught it at ratio 1.00. With PERIOD = h the prior is
# uniformly, maximally wrong across the whole forecast window.
PERIOD = 192
HORIZON = 192         # task B's horizon, and hence the flip period
LOOKBACK = 96         # so the evaluation context is 96+192 = 288 steps = 1.5 periods
N_HARMONICS = 3
NOISE_FRAC = 0.10     # sigma as a fraction of each column's amplitude: signal dominates
START = "2016-07-01 00:00:00"
COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def build(seed=SEED, t=T, period=PERIOD):
    """The series, plus the pieces needed to score it. Pure: same seed -> same array."""
    rng = np.random.default_rng(seed)
    idx = np.arange(t)
    phase = (idx % period) / period * 2 * np.pi
    sign = np.where((idx // period) % 2 == 0, 1.0, -1.0)

    shapes, amps = [], []
    for _ in COLUMNS:
        # A smooth periodic shape: a few harmonics with seeded coefficients. Smooth matters -- a
        # jagged target would be unlearnable at this noise level for reasons unrelated to the prior.
        s = np.zeros(t)
        for k in range(1, N_HARMONICS + 1):
            s += rng.normal(0, 1.0 / k) * np.sin(k * phase) + rng.normal(0, 1.0 / k) * np.cos(k * phase)
        s /= np.std(s)                      # unit-variance shape, so amplitude is the only scale
        a = float(rng.uniform(2.0, 12.0))   # ETT-like heterogeneous column scales
        shapes.append(s)
        amps.append(a)

    clean = np.stack([a * sign * s for a, s in zip(amps, shapes)], axis=1)
    noise = rng.normal(0, 1.0, size=clean.shape) * (np.array(amps) * NOISE_FRAC)
    return clean + noise, clean, np.array(amps)


def score(x, period=PERIOD, horizon=HORIZON):
    """Is it anti-persistent, and is the structure there to learn?

    Scored on standardised columns so the numbers are comparable across scales, on the same window
    geometry the paper evaluates: forecast `horizon` steps from the step after the input. Windows start
    at arbitrary offsets (stride 37, coprime with the period) precisely so the result cannot depend on
    the windows happening to align with the sign flip.
    """
    z = (x - x.mean(0)) / x.std(0)
    starts = np.arange(period, len(z) - horizon, 37)
    truth = np.stack([z[i:i + horizon] for i in starts])
    # seasonal-naive at lag `period`: repeat the last full period, tiled if the horizon is longer.
    snaive = np.stack([np.tile(z[i - period:i], (horizon // period + 1, 1))[:horizon] for i in starts])
    # persistence: hold the last observed value across the whole horizon.
    persist = np.stack([np.repeat(z[i - 1][None, :], horizon, axis=0) for i in starts])
    mse = lambda p: float(np.mean((p - truth) ** 2))
    return dict(
        mse_seasonal_naive=mse(snaive),
        mse_negated_seasonal_naive=mse(-snaive),
        mse_persistence=mse(persist),
        mse_climatology=mse(np.zeros_like(truth)),    # the standardised mean, i.e. the floor
        n_windows=int(len(starts)), horizon=horizon, seasonal_naive_lag=period,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="regenerate in memory and verify against results/synthetic_task_b.json")
    a = ap.parse_args()

    x, clean, amps = build()
    s = score(x)

    # THE TWO PROPERTIES, CHECKED. Negating seasonal-naive must beat seasonal-naive by a wide margin
    # (anti-persistence), and must also beat climatology (there is real structure to learn, so the
    # win is not just "the mean is better than a bad guess").
    ratio = s["mse_seasonal_naive"] / s["mse_negated_seasonal_naive"]
    assert ratio > 5.0, (
        f"the series is not strongly anti-persistent: seasonal-naive MSE is only {ratio:.2f}x its "
        f"own negation. Task B would not conflict with the pre-trained prior, so a positive control "
        f"built on it could not destroy anything.")
    assert s["mse_negated_seasonal_naive"] < 0.5 * s["mse_climatology"], (
        f"negated seasonal-naive ({s['mse_negated_seasonal_naive']:.4f}) does not beat climatology "
        f"({s['mse_climatology']:.4f}); there is no learnable structure above the noise floor.")

    dates = pd.date_range(START, periods=T, freq="h")
    df = pd.DataFrame(x, columns=COLUMNS)
    df.insert(0, "date", dates)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if a.check:
        tmp = OUT_CSV.with_suffix(".check.csv")
        df.to_csv(tmp, index=False)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        tmp.unlink()
    else:
        df.to_csv(OUT_CSV, index=False)
        digest = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()

    record = dict(
        path=str(OUT_CSV.relative_to(ROOT)), sha256=digest,
        generator="scripts/make_conflicting_series.py", seed=SEED,
        n_timesteps=T, period=PERIOD, horizon=HORIZON, lookback=LOOKBACK,
        n_harmonics=N_HARMONICS, noise_frac=NOISE_FRAC,
        columns=COLUMNS, start=START, freq="h",
        amplitudes=[round(float(v), 6) for v in amps],
        construction=("x[j,t] = a[j] * (-1)**(t//P) * s[j](t mod P) + sigma[j]*noise; s[j] a "
                      "3-harmonic unit-variance periodic shape, P=192 = h"),
        anti_persistence=s,
        seasonal_naive_over_its_negation=round(ratio, 4),
    )

    if a.check:
        if not OUT_JSON.exists():
            raise SystemExit(f"{OUT_JSON.relative_to(ROOT)} missing; run without --check first")
        stored = json.load(open(OUT_JSON))
        if stored["sha256"] != digest:
            raise SystemExit(f"HASH MISMATCH\n  stored {stored['sha256']}\n  rebuilt {digest}\n"
                             f"  the generator no longer reproduces the registered task B")
        print(f"ok  regenerated byte-identically: {digest[:16]}...")
    else:
        OUT_JSON.write_text(json.dumps(record, indent=2))
        print(f"wrote {OUT_CSV.relative_to(ROOT)}  sha256={digest[:16]}...  "
              f"{T} rows x {len(COLUMNS)} cols")
        print(f"wrote {OUT_JSON.relative_to(ROOT)}")

    print(f"\nanti-persistence at h={s['horizon']} over {s['n_windows']} standardised windows:")
    print(f"  seasonal-naive          {s['mse_seasonal_naive']:8.4f}   <- the pre-trained prior")
    print(f"  NEGATED seasonal-naive  {s['mse_negated_seasonal_naive']:8.4f}   <- the correct move")
    print(f"  persistence             {s['mse_persistence']:8.4f}")
    print(f"  climatology (the floor) {s['mse_climatology']:8.4f}")
    print(f"  seasonal-naive is {ratio:.1f}x worse than its own negation, so fitting task B requires "
          f"overwriting\n  the prior rather than extending it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
