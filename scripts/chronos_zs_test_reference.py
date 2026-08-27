#!/usr/bin/env python3
"""
One shared zero-shot reference on the held-out Chronos test windows.

WHY
---
B-D = (MSE_B - MSE_D) / MSE_ZS, so the zero-shot term is a *shared* denominator: both arms must
divide by the same number. But chronos_zs_mse decodes generatively with num_samples=20, drawing on
the global torch RNG, and the B and D scripts reach that call with different RNG state. Measured on
ETTh1 seed 42 the two arms disagreed by 1.4% (1.3349 vs 1.3537) on identical windows -- small, but
it makes B-D depend on which arm's estimate you happen to pick.

This script measures the reference once per dataset, with the RNG seeded explicitly and more
samples than the in-run measurement, and writes it where the analysis can pick it up. No training,
no fine-tuned weights: the pretrained checkpoint on the same 200 test windows (test_seed=0).

Run:  python3 scripts/chronos_zs_test_reference.py --device mps
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chronos_mse_finetune import DATASETS, MODEL_ID, build_windows, chronos_zs_mse, load_series


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--test-seed", type=int, default=0)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=100,
                   help="decode samples; higher than the in-run 20 to tighten the reference")
    p.add_argument("--out", default="results/v39_chronos_heldout/zs_test_reference.json")
    args = p.parse_args()

    from chronos import ChronosPipeline

    out = {"test_seed": args.test_seed, "rng_seed": args.rng_seed,
           "num_samples": args.num_samples, "datasets": {}}
    for name in ("ETTh1", "ETTh2"):
        cfg = DATASETS[name]
        lookback, horizon = cfg["lookback"], cfg["horizon"]
        _, _, test_s = load_series(name)
        ctx_te, tgt_te = build_windows(test_s, lookback, horizon, max_windows=200,
                                       seed=args.test_seed)

        pipe = ChronosPipeline.from_pretrained(MODEL_ID, dtype=torch.float32)
        pipe.tokenizer.config.prediction_length = horizon
        if args.device != "cpu":
            pipe.model.model = pipe.model.model.to(args.device)

        # Seed immediately before the sampled decode so the reference is reproducible.
        np.random.seed(args.rng_seed)
        torch.manual_seed(args.rng_seed)
        zs = chronos_zs_mse(pipe, ctx_te, tgt_te, args.device)

        out["datasets"][name] = {"zs_mse_test": zs, "n_windows": int(len(ctx_te))}
        print(f"  {name}: zero-shot test MSE = {zs:.6f}  over {len(ctx_te)} windows")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
