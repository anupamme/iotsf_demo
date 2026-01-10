"""
CICIoT2023 Dataset Loader

Provides functionality to load and sample from CICIoT2023 dataset or synthetic data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
import pickle


class CICIoT2023Loader:
    """
    Loader for CICIoT2023 IoT network traffic dataset.

    Handles loading, filtering, and sampling of network flow records.
    Supports both real CICIoT2023 data and synthetic test data.
    """

    # Attack type categories from CICIoT2023
    ATTACK_CATEGORIES = {
        'DDoS': [
            'DDoS-ICMP_Flood', 'DDoS-UDP_Flood', 'DDoS-TCP_Flood',
            'DDoS-PSHACK_Flood', 'DDoS-SYN_Flood', 'DDoS-RSTFINFlood',
            'DDoS-SynonymousIP_Flood', 'DDoS-ICMP_Fragmentation',
            'DDoS-ACK_Fragmentation', 'DDoS-UDP_Fragmentation',
            'DDoS-HTTP_Flood', 'DDoS-SlowLoris'
        ],
        'DoS': ['DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-SYN_Flood', 'DoS-HTTP_Flood'],
        'Recon': [
            'Recon-PingSweep', 'Recon-OSScan', 'Recon-PortScan',
            'Recon-HostDiscovery'
        ],
        'Web': [
            'SqlInjection', 'CommandInjection', 'XSS',
            'BrowserHijacking', 'Backdoor_Malware', 'Uploading_Attack'
        ],
        'BruteForce': ['DictionaryBruteForce'],
        'Spoofing': ['DNS_Spoofing', 'MITM-ArpSpoofing'],
        'Mirai': ['Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain']
    }

    # Required feature columns (12 dimensions for time-series)
    FEATURE_COLUMNS = [
        'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot',
        'fwd_data_pkts_tot', 'bwd_data_pkts_tot',
        'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec',
        'fwd_byts_b_avg', 'bwd_byts_b_avg',
        'fwd_iat_mean', 'bwd_iat_mean'
    ]

    def __init__(self, data_dir: str, cache_dir: Optional[str] = None):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to directory containing CSV files
            cache_dir: Optional path to cache preprocessed data
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._data_cache = {}

        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist.")
            logger.info("Run: python scripts/download_data.py --synthetic")

    def _load_csv_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single CSV file with caching."""
        cache_key = str(file_path)

        if cache_key in self._data_cache:
            logger.debug(f"Using cached data for {file_path.name}")
            return self._data_cache[cache_key]

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading {file_path.name}...")
        df = pd.read_csv(file_path)

        # Validate required columns
        missing_cols = [col for col in self.FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in {file_path.name}: {missing_cols}")

        self._data_cache[cache_key] = df
        return df

    def load_benign_samples(
        self,
        n_samples: int,
        file_pattern: str = "benign*.csv"
    ) -> pd.DataFrame:
        """
        Load benign traffic samples.

        Args:
            n_samples: Number of samples to return
            file_pattern: Glob pattern to match benign files

        Returns:
            DataFrame with n_samples rows
        """
        # Find benign CSV files
        benign_files = list(self.data_dir.glob(file_pattern))

        if not benign_files:
            raise FileNotFoundError(
                f"No benign files found matching '{file_pattern}' in {self.data_dir}"
            )

        logger.info(f"Found {len(benign_files)} benign file(s)")

        # Load and concatenate all benign files
        dfs = []
        for file_path in benign_files:
            df = self._load_csv_file(file_path)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        # Sample n_samples
        if len(combined_df) < n_samples:
            logger.warning(
                f"Requested {n_samples} samples but only {len(combined_df)} available. "
                f"Returning all available samples."
            )
            return combined_df

        sampled_df = combined_df.sample(n=n_samples, random_state=42)
        logger.info(f"Loaded {len(sampled_df)} benign samples")

        return sampled_df[self.FEATURE_COLUMNS]

    def load_attack_samples(
        self,
        attack_type: str,
        n_samples: int
    ) -> pd.DataFrame:
        """
        Load attack samples of a specific type.

        Args:
            attack_type: Attack type (e.g., 'DDoS-TCP_Flood', 'Recon-PortScan')
            n_samples: Number of samples to return

        Returns:
            DataFrame with n_samples rows
        """
        # Find files matching attack type
        attack_files = list(self.data_dir.glob(f"*{attack_type.lower()}*.csv"))

        if not attack_files:
            # Try generic attack pattern
            attack_files = list(self.data_dir.glob("*attack*.csv"))

        if not attack_files:
            raise FileNotFoundError(
                f"No files found for attack type '{attack_type}' in {self.data_dir}"
            )

        logger.info(f"Found {len(attack_files)} file(s) for {attack_type}")

        # Load and filter by label
        dfs = []
        for file_path in attack_files:
            df = self._load_csv_file(file_path)
            if 'label' in df.columns:
                df = df[df['label'].str.contains(attack_type, case=False, na=False)]
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        if len(combined_df) == 0:
            raise ValueError(f"No samples found for attack type: {attack_type}")

        # Sample n_samples
        if len(combined_df) < n_samples:
            logger.warning(
                f"Requested {n_samples} samples but only {len(combined_df)} available."
            )
            sampled_df = combined_df
        else:
            sampled_df = combined_df.sample(n=n_samples, random_state=42)

        logger.info(f"Loaded {len(sampled_df)} {attack_type} samples")

        return sampled_df[self.FEATURE_COLUMNS]

    def get_mixed_batch(
        self,
        n_benign: int,
        n_attack: int,
        attack_types: List[str]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Get a mixed batch of benign and attack samples.

        Args:
            n_benign: Number of benign samples
            n_attack: Number of attack samples (total across all types)
            attack_types: List of attack types to include

        Returns:
            Tuple of (features DataFrame, labels array)
            Labels: 0 = benign, 1 = attack
        """
        # Load benign samples
        benign_df = self.load_benign_samples(n_benign)
        benign_labels = np.zeros(len(benign_df))

        # Load attack samples (distribute across types)
        samples_per_type = n_attack // len(attack_types)
        remaining = n_attack % len(attack_types)

        attack_dfs = []
        for i, attack_type in enumerate(attack_types):
            n = samples_per_type + (1 if i < remaining else 0)
            attack_df = self.load_attack_samples(attack_type, n)
            attack_dfs.append(attack_df)

        attack_combined = pd.concat(attack_dfs, ignore_index=True)
        attack_labels = np.ones(len(attack_combined))

        # Combine and shuffle
        all_data = pd.concat([benign_df, attack_combined], ignore_index=True)
        all_labels = np.concatenate([benign_labels, attack_labels])

        # Shuffle
        indices = np.random.permutation(len(all_data))
        all_data = all_data.iloc[indices].reset_index(drop=True)
        all_labels = all_labels[indices]

        logger.info(
            f"Created mixed batch: {n_benign} benign + {len(attack_combined)} attack"
        )

        return all_data, all_labels

    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about available data."""
        stats = {
            'data_dir': str(self.data_dir),
            'files': [],
            'total_samples': 0
        }

        for csv_file in self.data_dir.glob("*.csv"):
            df = self._load_csv_file(csv_file)
            stats['files'].append({
                'name': csv_file.name,
                'samples': len(df),
                'label': df['label'].iloc[0] if 'label' in df.columns else 'Unknown'
            })
            stats['total_samples'] += len(df)

        return stats
