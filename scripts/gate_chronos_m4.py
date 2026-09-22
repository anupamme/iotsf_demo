#!/usr/bin/env python3
"""The corrected value gate for Chronos-T5-Small on M4-Monthly, on the split the protocol requires.

WHAT THIS REPLACES. The appendix quoted an 84.5% advantage for this cell. That number came from
scripts/finetune_chronos_m4.py, which divided by a per-series LinearRegression refitted for every
window and scored the ratio on the TEST split. Neither is the gate this paper defines: the gate's
denominator is ONE ridge-OLS map, lam=1e-4, fitted once on the training windows and applied unchanged,
and it is scored on the SELECTION split so that no gate decision is made on the windows an outcome is
later reported on. Swapping the unfitted trend denominator for the fitted one moved 17 of Moirai's 21
cells across the threshold, 16 of them to gate-fail, so a cell whose gate was never recomputed is not
a cell with a weak gate -- it is a cell with no gate.

WHAT IS COMPUTED, AND WHY ALL OF IT. The primary number is the `fitted` rung on the selection split.
But a single denominator is exactly the fragility this project already got caught by, so the whole
eight-rung ladder runs here as it does for every other cell, including the `constant` training-mean
floor that decides whether a denominator is ADMISSIBLE at all. The superseded `trend` rung is computed
too, on the same windows -- not because it is admissible, but because the paper has to be able to say
what the 84.5% actually measured, and that requires measuring it again the same way.

THE HELD-OUT GATE IS COMPUTED AND IS NOT THE GATE. Both splits are reported because the paper reports
both everywhere else and because the difference between them is one of its findings. The selection
number is the one any decision rests on.

SAMPLING IS STOCHASTIC. Chronos forecasts by sampling; `zs_mse` here is the median of 20 samples under
a fixed torch seed, recorded as `zs_seed`. The three fine-tuning runs each measure the same quantity
under their own seed, so the emitter reports the seed-mean gate from the runs alongside this
standalone number -- if they disagree by anything that matters, that disagreement is a measurement
fact about this cell and belongs in the paper.

Run:  .venv-probe/bin/python scripts/gate_chronos_m4.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import gate_baselines                                           # noqa: E402
from gate_all_cells import _window_linear_mse                    # noqa: E402
from m4_monthly_data import HORIZON, LOOKBACK, build_splits, load_series  # noqa: E402

OUT = ROOT / "results/chronos_m4/gate.json"
DATASET = "m4_monthly"          # gate_baselines.SEASON_OF key; season = 12
MODEL_ID = "amazon/chronos-t5-small"
ZS_SEED = 42
THRESHOLD = 0.20


def chronos_zs_mse(pipe, contexts, targets, batch_size=32):
    """Median-of-20-samples MSE on the per-window z-scored scale.

    Byte-for-byte the aggregation in chronos_mse_finetune.chronos_zs_mse: the numerator of the gate
    has to be the same quantity the runs report, and the way to guarantee that is to compute it the
    same way rather than to describe it the same way.
    """
    horizon = targets.shape[1]
    preds = []
    pipe.model.eval()
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = [torch.tensor(contexts[j], dtype=torch.float32)
                     for j in range(i, min(i + batch_size, len(contexts)))]
            samples = pipe.predict(batch, prediction_length=horizon, num_samples=20)
            preds.append(samples.median(dim=1).values.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    mses = []
    for i in range(len(contexts)):
        mu, sd = contexts[i].mean(), contexts[i].std() + 1e-8
        mses.append(float(np.mean(((preds[i] - mu) / sd - (targets[i] - mu) / sd) ** 2)))
    return float(np.mean(mses))


def main():
    reg = ROOT / "results/chronos_m4/preregistration.json"
    if not reg.exists():
        sys.exit("no registration at results/chronos_m4/preregistration.json -- register first")

    series = load_series()
    # max_train=None: the gate's denominator is fitted on ALL training windows, as gate_all_cells
    # does for every other cell. The 500-window cap belongs to the fine-tuning runs, not here.
    train, sel, ho, info = build_splits(series, max_train=None)
    print(f"{info['n_series']} series, {info['train_windows_used']} train windows, "
          f"{info['n_selection_windows']} selection, {info['n_heldout_windows']} held-out")

    torch.manual_seed(ZS_SEED)
    np.random.seed(ZS_SEED)
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(MODEL_ID, dtype=torch.float32)

    out = dict(cell="chronos_m4monthly", model_id=MODEL_ID, arm="chronos_m4",
               lookback=LOOKBACK, horizon=HORIZON, threshold=THRESHOLD, zs_seed=ZS_SEED,
               data=({k: v for k, v in info.items() if k != "series_ids"}),
               splits={})

    for split_name, (ctx, tgt) in (("selection", sel), ("heldout", ho)):
        zs = chronos_zs_mse(pipe, ctx, tgt)
        rungs = {}
        # "trend" first so the superseded number is on record beside the ones that replace it.
        for name in ("trend", "fitted") + gate_baselines.NAMES:
            info_out = {}
            need_train = name not in gate_baselines.NO_TRAINING
            mse = _window_linear_mse(ctx, tgt, LOOKBACK, HORIZON,
                                     train=(train if need_train else None),
                                     baseline=name, dataset=DATASET, info_out=info_out)
            rungs[name] = dict(baseline_mse=mse, r2_task=1 - zs / mse,
                               passes=bool(1 - zs / mse >= THRESHOLD), info=info_out)
            print(f"  {split_name:10s} {name:15s} MSE {mse:9.4f}  R2_task {1 - zs / mse:+.4f}"
                  f"{'  PASS' if 1 - zs / mse >= THRESHOLD else ''}")
        # Admissibility: a denominator worse than the training-mean predictor makes R2_task
        # meaningless in the direction that flatters the checkpoint. Recorded per rung so the
        # ladder can be read as "how far up does the advantage survive" rather than trusted.
        floor = rungs["constant"]["baseline_mse"]
        for name, d in rungs.items():
            d["admissible"] = bool(d["baseline_mse"] <= floor)
        admissible = [n for n, d in rungs.items() if d["admissible"] and n != "constant"]
        best = max(admissible, key=lambda n: rungs[n]["r2_task"]) if admissible else None
        out["splits"][split_name] = dict(
            zs_mse=zs, rungs=rungs, n_windows=int(len(ctx)),
            primary_r2_task=rungs["fitted"]["r2_task"],
            primary_passes=rungs["fitted"]["passes"],
            constant_floor_mse=floor,
            admissible_rungs=sorted(admissible),
            inadmissible_rungs=sorted(n for n, d in rungs.items() if not d["admissible"]),
            best_admissible_rung=best,
            worst_case_r2_task=(min(rungs[n]["r2_task"] for n in admissible) if admissible else None),
            survives_every_admissible_rung=bool(
                admissible and all(rungs[n]["r2_task"] >= THRESHOLD for n in admissible)),
        )
        print(f"  {split_name:10s} ZS {zs:.4f}  PRIMARY (fitted) R2_task "
              f"{rungs['fitted']['r2_task']:+.4f}  "
              f"{'PASS' if rungs['fitted']['passes'] else 'FAIL'} at {THRESHOLD}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
