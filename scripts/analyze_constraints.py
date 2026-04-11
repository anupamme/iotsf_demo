#!/usr/bin/env python3
"""
Protocol constraint compliance analysis for NeurIPS 2026 Table 3.

Generates N synthetic attack samples per (protocol × attack_type) and
measures the pass rate at each strictness level (strict / moderate / permissive).

Output: results/constraint_analysis.json
        results/constraint_analysis.txt  (human-readable table)

Usage:
    python scripts/analyze_constraints.py
    python scripts/analyze_constraints.py --n-samples 50 --output-dir results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.constraints.manager import IoTConstraintManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATTACK_TYPES = ["slow_exfiltration", "lotl_mimicry", "beacon", "protocol_anomaly"]
PROTOCOLS = ["modbus", "mqtt", "coap"]
STRICTNESS_LEVELS = ["strict", "moderate", "permissive"]


def load_or_generate_samples(
    synthetic_dir: str, attack_type: str, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Load pre-generated synthetic samples if available, otherwise fall back to
    statistical mock generation.
    """
    synth = Path(synthetic_dir)
    # Try highest stealth level first for the most challenging samples
    for stealth in [95, 90, 85]:
        fp = synth / f"{attack_type}_stealth_{stealth}.npy"
        if fp.exists():
            arr = np.load(fp)
            if len(arr) >= n_samples:
                idx = rng.choice(len(arr), size=n_samples, replace=False)
            else:
                idx = rng.choice(len(arr), size=n_samples, replace=True)
            return arr[idx]

    # Fallback: generate Gaussian noise samples of the right shape
    logger.warning(f"No pre-generated file for {attack_type}; using mock samples")
    return rng.standard_normal((n_samples, 128, 12)).astype(np.float32)


def check_samples(
    manager: IoTConstraintManager,
    samples: np.ndarray,
    protocol: str,
    strictness: str,
) -> dict:
    """
    Run all samples through the constraint manager and tally pass/fail.

    Returns a dict with keys: n_pass, n_fail, pass_rate, violation_counts
    """
    n_pass = 0
    violation_counts: dict = {}

    for sample in samples:
        try:
            report = manager.validate(sample, protocol=protocol, strictness=strictness)
            if report.is_valid:
                n_pass += 1
            else:
                for v in getattr(report, "violations", []):
                    key = str(v)[:60]
                    violation_counts[key] = violation_counts.get(key, 0) + 1
        except Exception as e:
            # Count as fail
            key = f"error:{type(e).__name__}"
            violation_counts[key] = violation_counts.get(key, 0) + 1

    n_total = len(samples)
    return {
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_total - n_pass,
        "pass_rate": n_pass / n_total if n_total > 0 else 0.0,
        "top_violations": dict(sorted(violation_counts.items(),
                                       key=lambda x: -x[1])[:5]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Protocol constraint compliance analysis (NeurIPS Table 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--synthetic-dir", default="data/synthetic")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Samples per (attack_type × protocol) cell (default 100)")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Initialise constraint manager
    try:
        manager = IoTConstraintManager()
    except Exception as e:
        logger.error(f"Could not initialise IoTConstraintManager: {e}")
        sys.exit(1)

    results = {}   # {attack_type: {protocol: {strictness: stats}}}

    total_cells = len(ATTACK_TYPES) * len(PROTOCOLS) * len(STRICTNESS_LEVELS)
    done = 0

    for attack_type in ATTACK_TYPES:
        results[attack_type] = {}
        logger.info(f"\nAttack type: {attack_type}")

        samples = load_or_generate_samples(
            args.synthetic_dir, attack_type, args.n_samples, rng
        )

        for protocol in PROTOCOLS:
            results[attack_type][protocol] = {}
            for strictness in STRICTNESS_LEVELS:
                stats = check_samples(manager, samples, protocol, strictness)
                results[attack_type][protocol][strictness] = stats
                done += 1
                logger.info(
                    f"  {protocol}/{strictness}: "
                    f"pass_rate={stats['pass_rate']:.2%} "
                    f"({stats['n_pass']}/{stats['n_total']})"
                )

    # --- Save JSON ---
    out_json = output_path / "constraint_analysis.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Saved JSON to {out_json}")

    # --- Build readable table ---
    lines = ["=" * 90]
    lines.append("PROTOCOL CONSTRAINT COMPLIANCE PASS RATES  (Table 3)")
    lines.append("=" * 90)
    lines.append(
        f"{'Attack Type':<22} {'Protocol':<10} {'Strict':>8} {'Moderate':>10} {'Permissive':>12}"
    )
    lines.append("-" * 90)

    for attack_type in ATTACK_TYPES:
        for protocol in PROTOCOLS:
            row = results[attack_type][protocol]
            lines.append(
                f"{attack_type:<22} {protocol:<10} "
                f"{row['strict']['pass_rate']:>8.1%} "
                f"{row['moderate']['pass_rate']:>10.1%} "
                f"{row['permissive']['pass_rate']:>12.1%}"
            )
        lines.append("")

    lines.append("=" * 90)
    table_text = "\n".join(lines)
    print("\n" + table_text)

    out_txt = output_path / "constraint_analysis.txt"
    out_txt.write_text(table_text)
    logger.success(f"Saved table to {out_txt}")


if __name__ == "__main__":
    main()
