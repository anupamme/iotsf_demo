"""
Download CICIoT2023 dataset or generate synthetic samples.

The full CICIoT2023 dataset is 548GB. This script provides:
1. Instructions for manual download
2. Synthetic data generation for testing
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from loguru import logger


DATASET_INFO = """
CICIoT2023 Dataset Download Instructions:
==========================================

The full dataset is 548GB and requires manual download:

1. Visit: https://www.unb.ca/cic/datasets/iotdataset-2023.html
2. Fill out the registration form
3. Download recommended subsets:
   - Part 1 (Benign traffic)
   - Part 3 (Mirai and DDoS attacks)
   - Part 5 (Reconnaissance attacks)

4. Extract CSV files to: data/raw/ciciot2023/

For development/testing, use --synthetic flag to generate sample data.
"""

FEATURE_COLUMNS = [
    'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot',
    'fwd_data_pkts_tot', 'bwd_data_pkts_tot',
    'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec',
    'fwd_byts_b_avg', 'bwd_byts_b_avg',
    'fwd_iat_mean', 'bwd_iat_mean'
]


def generate_synthetic_data(output_dir: str, n_samples: int = 10000):
    """
    Generate synthetic network traffic data for testing.

    Creates:
    - benign_traffic.csv (normal IoT traffic)
    - ddos_attack.csv (DDoS attack patterns)
    - recon_attack.csv (Reconnaissance patterns)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {n_samples} synthetic samples...")

    # Benign traffic: lower values, less variance
    benign_data = {
        'flow_duration': np.random.exponential(100, n_samples),
        'fwd_pkts_tot': np.random.poisson(50, n_samples),
        'bwd_pkts_tot': np.random.poisson(40, n_samples),
        'fwd_data_pkts_tot': np.random.poisson(45, n_samples),
        'bwd_data_pkts_tot': np.random.poisson(35, n_samples),
        'fwd_pkts_per_sec': np.random.gamma(2, 10, n_samples),
        'bwd_pkts_per_sec': np.random.gamma(2, 8, n_samples),
        'flow_pkts_per_sec': np.random.gamma(2, 15, n_samples),
        'fwd_byts_b_avg': np.random.normal(1000, 200, n_samples),
        'bwd_byts_b_avg': np.random.normal(800, 150, n_samples),
        'fwd_iat_mean': np.random.exponential(20, n_samples),
        'bwd_iat_mean': np.random.exponential(25, n_samples),
    }
    benign_df = pd.DataFrame(benign_data)
    benign_df['label'] = 'Benign'
    benign_df.to_csv(output_path / 'benign_traffic.csv', index=False)
    logger.info(f"✓ Generated benign_traffic.csv ({n_samples} samples)")

    # DDoS attack: high packet rates, short durations
    ddos_data = {
        'flow_duration': np.random.exponential(10, n_samples),  # Shorter
        'fwd_pkts_tot': np.random.poisson(500, n_samples),  # Much higher
        'bwd_pkts_tot': np.random.poisson(5, n_samples),  # Asymmetric
        'fwd_data_pkts_tot': np.random.poisson(480, n_samples),
        'bwd_data_pkts_tot': np.random.poisson(3, n_samples),
        'fwd_pkts_per_sec': np.random.gamma(10, 50, n_samples),  # Very high
        'bwd_pkts_per_sec': np.random.gamma(1, 2, n_samples),
        'flow_pkts_per_sec': np.random.gamma(10, 60, n_samples),
        'fwd_byts_b_avg': np.random.normal(100, 50, n_samples),  # Smaller packets
        'bwd_byts_b_avg': np.random.normal(50, 30, n_samples),
        'fwd_iat_mean': np.random.exponential(1, n_samples),  # Very short
        'bwd_iat_mean': np.random.exponential(50, n_samples),
    }
    ddos_df = pd.DataFrame(ddos_data)
    ddos_df['label'] = 'DDoS-TCP_Flood'
    ddos_df.to_csv(output_path / 'ddos_attack.csv', index=False)
    logger.info(f"✓ Generated ddos_attack.csv ({n_samples} samples)")

    # Reconnaissance: port scanning patterns
    recon_data = {
        'flow_duration': np.random.exponential(5, n_samples),  # Very short
        'fwd_pkts_tot': np.random.poisson(3, n_samples),  # Few packets
        'bwd_pkts_tot': np.random.poisson(2, n_samples),
        'fwd_data_pkts_tot': np.random.poisson(2, n_samples),
        'bwd_data_pkts_tot': np.random.poisson(1, n_samples),
        'fwd_pkts_per_sec': np.random.gamma(1, 5, n_samples),
        'bwd_pkts_per_sec': np.random.gamma(1, 3, n_samples),
        'flow_pkts_per_sec': np.random.gamma(1, 8, n_samples),
        'fwd_byts_b_avg': np.random.normal(200, 50, n_samples),
        'bwd_byts_b_avg': np.random.normal(150, 40, n_samples),
        'fwd_iat_mean': np.random.exponential(2, n_samples),
        'bwd_iat_mean': np.random.exponential(3, n_samples),
    }
    recon_df = pd.DataFrame(recon_data)
    recon_df['label'] = 'Recon-PortScan'
    recon_df.to_csv(output_path / 'recon_attack.csv', index=False)
    logger.info(f"✓ Generated recon_attack.csv ({n_samples} samples)")

    logger.success(f"Synthetic data saved to {output_path}/")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download or generate CICIoT2023 data")
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate synthetic data for testing'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/sample',
        help='Output directory for synthetic data'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=10000,
        help='Number of samples per class (synthetic only)'
    )

    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_data(args.output, args.samples)
    else:
        print(DATASET_INFO)


if __name__ == '__main__':
    main()
