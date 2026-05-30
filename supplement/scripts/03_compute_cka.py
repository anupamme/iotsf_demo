#!/usr/bin/env python3
"""Compute CKA between pre-trained and fine-tuned encoder representations.

Can operate on saved result JSONs (which already contain CKA) or recompute
from saved encoder checkpoints.

Usage:
    python scripts/03_compute_cka.py --results-dir runs/
    python scripts/03_compute_cka.py --config configs/etth2_small_n500.yaml \
        --pretrained-reps reps/pretrained.npy --finetuned-reps reps/finetuned.npy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cka import linear_CKA
from src.utils import load_results


def summarize_from_results(results_dir: str):
    """Summarize CKA values from pre-computed result JSONs."""
    results = load_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"{'Condition':<12} {'Seed':>6} {'CKA':>8} {'Forgetting%':>12}")
    print("-" * 42)

    for r in sorted(results, key=lambda x: (x.get("condition", ""), x.get("seed", 0))):
        cka = r.get("final_cka")
        if cka is None:
            continue
        print(
            f"{r['condition']:<12} {r['seed']:>6} {cka:>8.4f} "
            f"{r.get('forgetting_pct', 0):>+11.1f}%"
        )


def compute_from_arrays(pre_path: str, post_path: str):
    """Compute CKA from saved numpy representation arrays."""
    pre = np.load(pre_path)
    post = np.load(post_path)
    cka = linear_CKA(pre, post)
    print(f"Pre-trained reps: {pre.shape}")
    print(f"Fine-tuned reps:  {post.shape}")
    print(f"Linear CKA: {cka:.6f}")
    return cka


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="runs/")
    parser.add_argument("--pretrained-reps", default=None)
    parser.add_argument("--finetuned-reps", default=None)
    args = parser.parse_args()

    if args.pretrained_reps and args.finetuned_reps:
        compute_from_arrays(args.pretrained_reps, args.finetuned_reps)
    else:
        summarize_from_results(args.results_dir)


if __name__ == "__main__":
    main()
