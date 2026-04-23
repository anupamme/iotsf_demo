"""
Per-attack-type and per-stealth-level metrics breakdown.

Extends IDSMetrics to produce the 4×3 matrix of F1 scores (attack type × stealth
level) used in Figure 3 of the NeurIPS 2026 paper.

Usage:
    from src.evaluation.per_attack_metrics import PerAttackMetrics

    matrix = PerAttackMetrics.compute_matrix(detector, synthetic_dir)
    # Returns dict: {attack_type: {stealth: metrics_dict}}
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .metrics import IDSMetrics


ATTACK_TYPES = ["slow_exfiltration", "lotl_mimicry", "beacon", "protocol_anomaly"]
STEALTH_LEVELS = [85, 90, 95]


class PerAttackMetrics:
    """
    Compute per-condition evaluation metrics for the paper's Figure 3 heatmap.
    """

    @staticmethod
    def load_condition(
        synthetic_dir: str,
        attack_type: str,
        stealth: int,
        benign: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single attack condition and pair it with benign samples.

        Returns (X, y) where y=0 for benign, y=1 for attack.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        fp = Path(synthetic_dir) / f"{attack_type}_stealth_{stealth}.npy"
        if not fp.exists():
            raise FileNotFoundError(f"Missing: {fp}")

        attacks = np.load(fp)                             # (n_attack, T, F)
        n_b = min(len(benign), len(attacks))
        idx = rng.choice(len(benign), size=n_b, replace=False)
        b = benign[idx]

        X = np.concatenate([b, attacks])
        y = np.array([0] * len(b) + [1] * len(attacks))
        return X, y

    @staticmethod
    def compute_matrix(
        predict_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        synthetic_dir: str,
        benign_path: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Compute metrics for every (attack_type, stealth) cell.

        Parameters
        ----------
        predict_fn : callable(X) -> (y_pred, y_scores)
            A function that takes an array of shape (N, T, F) and returns
            binary predictions and anomaly scores.
        synthetic_dir : str
            Directory containing the pre-generated .npy files.
        benign_path : str, optional
            Path to benign_samples.npy.  Defaults to
            <synthetic_dir>/benign_samples.npy.
        rng : np.random.Generator, optional

        Returns
        -------
        matrix : dict {attack_type -> {stealth -> metrics_dict}}
        """
        if rng is None:
            rng = np.random.default_rng(42)

        benign_p = Path(benign_path) if benign_path else Path(synthetic_dir) / "benign_samples.npy"
        if not benign_p.exists():
            raise FileNotFoundError(f"benign_samples.npy not found at {benign_p}")
        benign = np.load(benign_p)

        matrix: Dict[str, Dict[int, Dict[str, Any]]] = {}

        for at in ATTACK_TYPES:
            matrix[at] = {}
            for stealth in STEALTH_LEVELS:
                try:
                    X, y = PerAttackMetrics.load_condition(
                        synthetic_dir, at, stealth, benign, rng
                    )
                    y_pred, y_scores = predict_fn(X)
                    metrics = IDSMetrics.compute_all_metrics(y, y_pred, y_scores)
                    # Convert ndarray values for JSON safety
                    matrix[at][stealth] = {
                        k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in metrics.items()
                    }
                except FileNotFoundError as exc:
                    matrix[at][stealth] = {"error": str(exc)}

        return matrix

    @staticmethod
    def to_f1_heatmap(matrix: Dict) -> Tuple[list, list, np.ndarray]:
        """
        Extract a (n_attack_types, n_stealth) F1 matrix for plotting.

        Returns
        -------
        attack_labels : list[str]
        stealth_labels : list[str]
        heatmap        : np.ndarray of shape (n_attack_types, n_stealth)
        """
        heatmap = np.full((len(ATTACK_TYPES), len(STEALTH_LEVELS)), np.nan)

        for i, at in enumerate(ATTACK_TYPES):
            for j, stealth in enumerate(STEALTH_LEVELS):
                cell = matrix.get(at, {}).get(stealth, {})
                if "f1" in cell:
                    heatmap[i, j] = cell["f1"]

        attack_labels = [at.replace("_", " ").title() for at in ATTACK_TYPES]
        stealth_labels = [f"Stealth {s}%" for s in STEALTH_LEVELS]
        return attack_labels, stealth_labels, heatmap

    @staticmethod
    def print_summary(matrix: Dict) -> str:
        """Format a readable summary of the per-attack-type results."""
        lines = ["=" * 70, "Per-Attack-Type × Stealth Level  F1 Score Matrix", "=" * 70]
        header = f"{'Attack Type':<25}" + "".join(f"  Stealth {s}%" for s in STEALTH_LEVELS)
        lines.append(header)
        lines.append("-" * 70)
        for at in ATTACK_TYPES:
            row = f"{at:<25}"
            for stealth in STEALTH_LEVELS:
                cell = matrix.get(at, {}).get(stealth, {})
                f1 = cell.get("f1", float("nan"))
                row += f"  {f1:>10.3f}"
            lines.append(row)
        lines.append("=" * 70)
        return "\n".join(lines)
