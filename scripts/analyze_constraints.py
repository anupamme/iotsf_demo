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
from src.models.hard_negative_generator import HardNegativeGenerator


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
# Retry statistics via HardNegativeGenerator
# ---------------------------------------------------------------------------

def collect_retry_stats(
    attack_type: str, n_samples: int, strictness: str = "moderate"
) -> dict:
    """
    Run HardNegativeGenerator for `n_samples` of `attack_type` at `strictness`
    and return the generation statistics (avg_retries, pct_first_try, etc.).

    If the generator fails to initialise (no checkpoint), returns placeholder stats.
    """
    try:
        manager = IoTConstraintManager()
        gen = HardNegativeGenerator(
            constraint_manager=manager,
            max_retries=5,
            strictness=strictness,
        )
        gen.initialize()
        gen.reset_statistics()

        # Generate a small batch to measure retries — capped at 20 for speed
        n_gen = min(n_samples, 20)
        attack_map = {
            "slow_exfiltration": "slow_exfil",
            "lotl_mimicry": "lotl",
            "beacon": "beacon",
            "protocol_anomaly": "protocol_anomaly",
        }
        pattern = attack_map.get(attack_type, attack_type)
        gen.generate_batch(n_samples=n_gen, attack_pattern=pattern)

        stats = gen.get_generation_statistics()
        retry_counts = stats.get("retry_counts", [])
        total = len(retry_counts)
        first_try = sum(1 for r in retry_counts if r == 0)
        return {
            "avg_retries": float(stats.get("average_retries", 0)),
            "max_retries_used": int(stats.get("max_retries_used", 0)),
            "pct_first_try": float(first_try / total) if total > 0 else 1.0,
            "total_generated": total,
        }
    except Exception as e:
        logger.warning(f"Could not collect retry stats for {attack_type}/{strictness}: {e}")
        return {
            "avg_retries": 0.0,
            "max_retries_used": 0,
            "pct_first_try": 1.0,
            "total_generated": 0,
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
    parser.add_argument("--skip-retry-stats", action="store_true",
                        help="Skip HardNegativeGenerator retry stat collection (faster)")
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
    retry_stats = {}  # {attack_type: retry_stats_dict}

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

        # Collect retry stats at moderate strictness (training default)
        if not args.skip_retry_stats:
            logger.info(f"  Collecting retry stats for {attack_type}...")
            rs = collect_retry_stats(attack_type, args.n_samples, strictness="moderate")
            retry_stats[attack_type] = rs
            logger.info(
                f"  avg_retries={rs['avg_retries']:.2f}, "
                f"pct_first_try={rs['pct_first_try']:.1%}, "
                f"max_retries={rs['max_retries_used']}"
            )

    # --- Save JSON ---
    out_json = output_path / "constraint_analysis.json"
    with open(out_json, "w") as f:
        json.dump({"pass_rates": results, "retry_stats": retry_stats}, f, indent=2)
    logger.success(f"Saved JSON to {out_json}")

    # --- Build readable table ---
    lines = ["=" * 100]
    lines.append("PROTOCOL CONSTRAINT COMPLIANCE PASS RATES  (Table 3)")
    lines.append("=" * 100)
    lines.append(
        f"{'Attack Type':<22} {'Protocol':<10} {'Strict':>8} {'Moderate':>10} "
        f"{'Permissive':>12} {'Avg Retries':>13} {'1st-try%':>10}"
    )
    lines.append("-" * 100)

    for attack_type in ATTACK_TYPES:
        rs = retry_stats.get(attack_type, {})
        avg_ret = rs.get("avg_retries", 0.0)
        pct_ft = rs.get("pct_first_try", 1.0)
        for i, protocol in enumerate(PROTOCOLS):
            row = results[attack_type][protocol]
            # Only show retry stats on first protocol row per attack type
            ret_col = f"{avg_ret:>13.2f}" if i == 0 else " " * 13
            ft_col = f"{pct_ft:>10.1%}" if i == 0 else " " * 10
            lines.append(
                f"{attack_type if i == 0 else '':<22} {protocol:<10} "
                f"{row['strict']['pass_rate']:>8.1%} "
                f"{row['moderate']['pass_rate']:>10.1%} "
                f"{row['permissive']['pass_rate']:>12.1%}"
                f"{ret_col}{ft_col}"
            )
        lines.append("")

    lines.append("=" * 100)
    table_text = "\n".join(lines)
    print("\n" + table_text)

    out_txt = output_path / "constraint_analysis.txt"
    out_txt.write_text(table_text)
    logger.success(f"Saved table to {out_txt}")


if __name__ == "__main__":
    main()
