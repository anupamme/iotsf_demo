#!/usr/bin/env python3
"""
Freeze the prospective predictions BEFORE any condition B or D run exists.

WHY THIS EXISTS
---------------
The six degradation cells were identified using the test split: the inclusion criterion is
recomputed there, so cell selection is retrospective. Showing (as we already do) that a
validation-side gate selects the same six answers a narrower question -- whether the INCLUSION
decision depends on the split -- and not the one a reviewer actually asks, which is whether the
criterion PREDICTS degradation on cells it has never seen.

This script writes the predictions for eight genuinely new cells and is committed to git before the
first B/D run is launched. The guarantee is structural rather than merely procedural: the outcome
variable (forg_B, forg_D, and hence degradation status) is computed from condition B and D runs
that DO NOT EXIST YET, and every quantity used to form a prediction comes from the training and
validation splits only.

TWO RULES ARE REGISTERED, and the comparison is the point.

  gate rule     gate_val >= 0.20 -> at risk of degradation.
                This is the paper's own criterion, pre-specified in Section 2.

  dataset rule  dataset in {ETTh1, Weather} -> degradation.
                This is POST HOC. It is the pattern we noticed in the original 13 Moirai cells,
                where degradation is separated perfectly by dataset (ETTh1/Weather degrade,
                ETTh2/ETTm2 do not) while gate values overlap heavily. It is registered here as a
                COMPETITOR, not as a contribution: if a rule we noticed after the fact beats the
                criterion we published, that is evidence about the criterion, and the paper says so.

Neither rule has seen these cells' test split. Run once, then commit.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/v47_prospective/preregistration.json"
GATE_THRESHOLD = 0.20
DEGRADATION_DATASETS = {"ETTh1", "Weather"}

# (size, dataset, horizon), in the priority order fixed before any run.
CELLS = [
    ("base", "Weather", 96), ("base", "Weather", 192),
    ("base", "ETTm2", 96), ("base", "ETTm2", 192),
    ("small", "Electricity7", 96), ("small", "Electricity7", 192),
    ("large", "ETTh1", 96), ("large", "Weather", 96),
]


def main():
    if OUT.exists():
        sys.exit(f"{OUT} already exists -- refusing to overwrite a registration")

    # Guard: the registration is only meaningful if no outcome run exists yet.
    existing = sorted(p.name for p in (ROOT / "results/v47_prospective").glob(
        "*/condition_[BD]/*.json"))
    if existing:
        sys.exit(f"refusing to register: condition B/D runs already exist: {existing[:3]}")

    gate_path = ROOT / "results/gate_val_side.json"
    if not gate_path.exists():
        sys.exit("run: .venv-probe/bin/python scripts/gate_all_cells.py --arm moirai --split val")
    gates = {k: v["r2_task"] for k, v in json.load(open(gate_path)).items()}

    entries = []
    for size, ds, h in CELLS:
        key = f"{size}_{ds}_h{h}"
        a = ROOT / f"results/v47_prospective/{key}/condition_A/condition_A_h{h}_s42.json"
        if not a.exists():
            sys.exit(f"missing zero-shot for {key}: run scripts/run_prospective_zs.sh first")
        zs = json.load(open(a))
        if key not in gates:
            sys.exit(f"no validation-side gate for {key}")
        g = gates[key]
        # VALIDATION-side quantities only. condition_A also stores zeroshot_test_mse; it is not read
        # here and plays no part in any prediction.
        entries.append(dict(
            cell=key, size=size, dataset=ds, horizon=h,
            zeroshot_val_mse=zs["zeroshot_mse"],
            gate_val=g,
            predict_gate_rule=bool(g >= GATE_THRESHOLD),
            predict_dataset_rule=ds in DEGRADATION_DATASETS,
        ))

    payload = dict(
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip(),
        gate_threshold=GATE_THRESHOLD,
        degradation_datasets=sorted(DEGRADATION_DATASETS),
        rules=dict(
            gate_rule="gate_val >= 0.20 -> at risk of degradation (pre-specified, Section 2)",
            dataset_rule=("dataset in {ETTh1, Weather} -> degradation (POST HOC pattern from the "
                          "original 13 Moirai cells, registered as a competitor)"),
        ),
        outcome_definition=("gate-passing AND forg_B > 0 in every seed AND forg_D < 0 in every "
                            "seed, exactly as scripts/cell_matrix.py degradation_cells() applies "
                            "it to the published cells"),
        note=("Predictions are a function of the training/validation splits and the dataset name "
              "only. The outcome requires condition B and D runs, which do not exist at the time "
              "this file is committed."),
        cells=entries,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(entries)} cells (gate values pending)")


if __name__ == "__main__":
    main()
