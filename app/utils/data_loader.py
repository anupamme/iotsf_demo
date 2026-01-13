"""Demo data loading utilities."""

import numpy as np
from pathlib import Path
from typing import Dict
import random


def load_demo_samples(data_dir: Path, config: dict) -> Dict:
    """
    Load and prepare demo samples for the challenge.

    Loads 3 benign samples and 3 attack samples (one of each type),
    shuffles them consistently, and returns with labels.

    Args:
        data_dir: Path to synthetic data directory
        config: Configuration dictionary

    Returns:
        Dict with:
            - samples: List of 6 numpy arrays (128, 12)
            - labels: List of 6 booleans (True=attack, False=benign)
            - attack_types: List of 6 strings describing each sample
            - shuffled_indices: Original indices before shuffle

    Example:
        >>> config = {'demo': {'attack_types': ['slow_exfiltration_stealth_95']}}
        >>> data = load_demo_samples(Path('data/synthetic'), config)
        >>> len(data['samples'])
        6
    """
    # Load synthetic data files
    benign = np.load(data_dir / "benign_samples.npy")

    # Get attack types from config
    attack_types_config = config.get('demo', {}).get('attack_types', [
        'slow_exfiltration_stealth_95',
        'lotl_mimicry_stealth_90',
        'protocol_anomaly_stealth_85'
    ])

    # Load attack samples
    attack_data = []
    for attack_type in attack_types_config:
        attack_file = data_dir / f"{attack_type}.npy"
        if attack_file.exists():
            attack_data.append(np.load(attack_file))
        else:
            raise FileNotFoundError(f"Attack file not found: {attack_file}")

    # Select samples: 3 benign + 1 of each attack type
    samples = [
        benign[0],
        benign[1],
        benign[2],
    ]

    # Add one sample from each attack type
    for attack_samples in attack_data:
        samples.append(attack_samples[0])

    # Create labels and type descriptions
    n_benign = 3
    n_attacks = len(attack_data)

    labels = [False] * n_benign + [True] * n_attacks
    attack_types = ['Benign'] * n_benign + attack_types_config

    # Shuffle consistently using fixed seed for reproducibility
    indices = list(range(len(samples)))
    random.seed(42)  # Fixed seed ensures same shuffle every time
    random.shuffle(indices)

    # Apply shuffle
    samples = [samples[i] for i in indices]
    labels = [labels[i] for i in indices]
    attack_types = [attack_types[i] for i in indices]

    return {
        'samples': samples,
        'labels': labels,
        'attack_types': attack_types,
        'shuffled_indices': indices
    }
