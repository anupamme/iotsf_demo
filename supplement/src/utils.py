"""Utility functions: config loading, seed setting, device detection, I/O."""

import json
import os
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import yaml


def load_config(config_path: str) -> Namespace:
    """Load YAML config and return as namespace."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return Namespace(**cfg)


def set_seed(seed: int, deterministic: bool = False):
    """Set all random seeds for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_str: str = "auto"):
    """Resolve device string to torch.device."""
    import torch

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def save_results(results: dict, output_path: str):
    """Save results dict as JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(results_dir: str, pattern: str = "*.json") -> list:
    """Load all JSON result files from a directory."""
    results = []
    for f in sorted(Path(results_dir).glob(pattern)):
        with open(f) as fp:
            results.append(json.load(fp))
    return results
