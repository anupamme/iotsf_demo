#!/usr/bin/env python3
"""The three Moirai capacities' parameter counts, and the ratio S4 quotes for them.

WHY THIS EXISTS
---------------
S1 describes the screen as "Moirai Small 13.8M, Base 91M, Large 311M" and S4 says "Moirai-Base is
6.3x larger than Moirai-Small". Those were the only backbone numbers in the body with no record
anywhere under results/: the counts are printed by src/models/moirai_detector.py at load time
(":348, Loaded MoiraiModule with {n:,} parameters") and every run log has them, but *.log is
gitignored, so nothing a clean clone can read carried them. Hand-typed numbers in this paper have a
history: the ratio was wrong. 91,357,728 / 13,827,528 = 6.61, not 6.3 -- 6.3 is the ratio of the
capacities as the Moirai paper rounds them (91M / 14.4M), not of the checkpoints we actually ran.

WHAT IT DOES. Constructs each MoiraiModule exactly the way the runs did -- same MixtureOutput
components, same config.json read from the HuggingFace cache -- and counts
sum(p.numel() for p in module.parameters()), which is the same expression the loader logs. Weights
are NEVER loaded: the count is a function of the config alone (this is why the loader can print it in
--random-init mode), so this script needs no safetensors and no GPU.

TIER. Needs torch, uni2ts and the HF cache, so it is NOT clean-clone re-derivable and is not in
rederive_all.sh's default sweep -- the same position drift_metric_battery.py's --compute pass is in.
Its OUTPUT (results/model_sizes.json) is committed, and that is what check_paper_numbers.py reads, so
the paper's numbers stay TIER A even though the count behind them is not.

The guard: the three counts are asserted against the values the run logs have printed since May 2026.
Those are hand-typed here on purpose and they are not a paper number -- they are the historical record
this script must reproduce. If Salesforce re-publishes a config under the same name, this fails rather
than silently re-describing the models the paper's runs used.

Run:  .venv-probe/bin/python scripts/emit_model_sizes.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/model_sizes.json"

# What every run log has printed since the first Moirai run (2026-05-02). See the docstring: a guard
# on reproduction, not a source for any number in the paper.
LOGGED = {"small": 13_827_528, "base": 91_357_728, "large": 310_970_624}


def count(size):
    """Parameter count of the MoiraiModule the runs built, from config.json alone."""
    import torch  # noqa: F401  (uni2ts imports fail without it on the path first)
    from huggingface_hub import hf_hub_download
    from uni2ts.distribution import (LogNormalOutput, MixtureOutput, NegativeBinomialOutput,
                                     NormalFixedScaleOutput, StudentTOutput)
    from uni2ts.model.moirai import MoiraiModule

    from src.models.moirai_detector import MODEL_SIZE_MAP

    model_id = MODEL_SIZE_MAP[size]
    with open(hf_hub_download(repo_id=model_id, filename="config.json")) as f:
        config = json.load(f)
    module = MoiraiModule(
        distr_output=MixtureOutput(components=[StudentTOutput(), NormalFixedScaleOutput(scale=0.001),
                                              NegativeBinomialOutput(), LogNormalOutput()]),
        d_model=config["d_model"], num_layers=config["num_layers"],
        patch_sizes=tuple(config["patch_sizes"]), max_seq_len=config["max_seq_len"],
        attn_dropout_p=config["attn_dropout_p"], dropout_p=config["dropout_p"],
        scaling=config.get("scaling", True))
    return model_id, sum(p.numel() for p in module.parameters()), config


def main():
    out = {"generated_by": "scripts/emit_model_sizes.py",
           "counted_as": "sum(p.numel() for p in MoiraiModule.parameters()), weights not loaded",
           "models": {}}
    for size in ("small", "base", "large"):
        model_id, n, config = count(size)
        assert n == LOGGED[size], (
            f"{model_id} now builds {n:,} parameters; every run log since 2026-05-02 records "
            f"{LOGGED[size]:,}. The checkpoint config changed, so the paper's backbone description "
            f"no longer describes the models it ran on.")
        out["models"][size] = dict(model_id=model_id, n_params=n,
                                   d_model=config["d_model"], num_layers=config["num_layers"])
        print(f"  {size:6s} {model_id:34s} {n:>12,}")
    b, s = out["models"]["base"]["n_params"], out["models"]["small"]["n_params"]
    out["base_over_small"] = b / s
    print(f"\n  Base / Small = {b / s:.4f}x  (S4 quotes this to 1 dp)")
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"  wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
