# Synthetic Data Directory

This directory contains pre-generated synthetic attack samples and demo data for the IoT Security Demo.

## Contents

### Generated Attack Samples
Individual attack samples at different stealth levels:

- `benign_samples.npy` - Baseline benign traffic (5 samples)
- `slow_exfiltration_stealth_85.npy` - Slow data exfiltration at 85% stealth
- `slow_exfiltration_stealth_90.npy` - Slow data exfiltration at 90% stealth
- `slow_exfiltration_stealth_95.npy` - Slow data exfiltration at 95% stealth
- `lotl_mimicry_stealth_85.npy` - Living-off-the-land at 85% stealth
- `lotl_mimicry_stealth_90.npy` - Living-off-the-land at 90% stealth
- `lotl_mimicry_stealth_95.npy` - Living-off-the-land at 95% stealth
- `beacon_stealth_85.npy` - C2 beacon pattern at 85% stealth
- `beacon_stealth_90.npy` - C2 beacon pattern at 90% stealth
- `beacon_stealth_95.npy` - C2 beacon pattern at 95% stealth
- `protocol_anomaly_stealth_85.npy` - Protocol anomaly at 85% stealth
- `protocol_anomaly_stealth_90.npy` - Protocol anomaly at 90% stealth
- `protocol_anomaly_stealth_95.npy` - Protocol anomaly at 95% stealth

**Shape**: Each file contains `(n_samples, 128, 12)` - n samples of 128 timesteps with 12 features each.

### Demo Data Package
`demo_data.npz` - Complete pre-computed demo dataset with detection results.

## demo_data.npz Format

The `demo_data.npz` file contains all samples and detection results needed for the demo presentation.

### Arrays

#### Samples
- **`benign_samples`**: `(3, 128, 12)` - Real benign traffic from CICIoT2023
- **`attack_samples`**: `(3, 128, 12)` - Synthetic attacks (one each: slow_exfiltration, lotl_mimicry, beacon)
- **`attack_labels`**: `(3,)` - String labels for each attack type
- **`labels`**: `(6,)` - Ground truth binary labels (0=benign, 1=attack) → `[0, 0, 0, 1, 1, 1]`

#### Baseline IDS Results
- **`baseline_predictions`**: `(6,)` - Binary predictions (0=benign, 1=attack)
- **`baseline_scores`**: `(6,)` - Anomaly scores in range [0, 1]

#### Moirai Results
- **`moirai_predictions`**: `(6,)` - Binary predictions (0=benign, 1=attack)
- **`moirai_scores`**: `(6,)` - Anomaly scores in range [0, 1]
- **`moirai_forecasts`**: `(6, 28, 12)` - Time-series forecasts (28 prediction steps, 12 features)

#### Metadata
- **`metadata`**: Dictionary containing:
  - `timestamp`: Generation timestamp
  - `n_benign`: Number of benign samples
  - `n_attacks`: Number of attack samples
  - `attack_types`: List of attack type strings
  - `seq_length`: Sequence length (128)
  - `feature_dim`: Number of features (12)
  - `seed`: Random seed used
  - `baseline_params`: Baseline IDS configuration
  - `moirai_params`: Moirai detector configuration

### Expected File Size
- `demo_data.npz`: ~200-500 KB

## Regenerating Data

### Generate Individual Attack Samples
To regenerate attack samples at various stealth levels:

```bash
python scripts/precompute_attacks.py --n-samples 20 --seed 42
```

Options:
- `--n-samples`: Number of samples per attack type (default: 10)
- `--output-dir`: Output directory (default: data/synthetic)
- `--seed`: Random seed for reproducibility (default: 42)

### Generate Demo Data Package
To regenerate the complete demo data with detection results:

```bash
python scripts/precompute_demo_data.py --seed 42
```

Options:
- `--n-benign`: Number of benign samples (default: 3)
- `--n-attacks`: Number of attack samples (default: 3)
- `--attack-types`: Attack types to generate (default: slow_exfiltration lotl_mimicry beacon)
- `--output-dir`: Output directory (default: data/synthetic)
- `--seed`: Random seed (default: 42)

**Note**: Requires Python 3.12 for full Moirai support. With Python 3.13+, Moirai detection will use mock results.

### Verify Data Integrity
After generation, verify the data:

```bash
python scripts/verify_demo_data.py
```

This checks:
- ✅ File exists and is readable
- ✅ All required arrays present with correct shapes
- ✅ No NaN or Inf values
- ✅ Scores in valid range [0, 1]
- ✅ Predictions are binary
- ✅ Attacks statistically different from benign
- ✅ At least some attacks detected
- ✅ File size reasonable

## Loading Demo Data

### In Python Scripts

```python
import numpy as np

# Load data
data = np.load('data/synthetic/demo_data.npz', allow_pickle=True)

# Access samples
benign = data['benign_samples']  # Shape: (3, 128, 12)
attacks = data['attack_samples']  # Shape: (3, 128, 12)
labels = data['labels']           # Shape: (6,) - [0,0,0,1,1,1]

# Access detection results
baseline_pred = data['baseline_predictions']  # Shape: (6,)
baseline_scores = data['baseline_scores']     # Shape: (6,)
moirai_pred = data['moirai_predictions']      # Shape: (6,)
moirai_scores = data['moirai_scores']         # Shape: (6,)
moirai_forecasts = data['moirai_forecasts']   # Shape: (6, 28, 12)

# Access metadata
metadata = data['metadata'].item()
print(f"Generated: {metadata['timestamp']}")
print(f"Attack types: {metadata['attack_types']}")
```

### In Streamlit Demo

```python
import streamlit as st
import numpy as np

@st.cache_data
def load_demo_data():
    """Load pre-computed demo data."""
    data = np.load('data/synthetic/demo_data.npz', allow_pickle=True)
    return {
        'benign': data['benign_samples'],
        'attacks': data['attack_samples'],
        'labels': data['labels'],
        'baseline_pred': data['baseline_predictions'],
        'baseline_scores': data['baseline_scores'],
        'moirai_pred': data['moirai_predictions'],
        'moirai_scores': data['moirai_scores'],
        'moirai_forecasts': data['moirai_forecasts'],
        'metadata': data['metadata'].item()
    }

# Use in app
demo_data = load_demo_data()
st.write(f"Loaded {len(demo_data['benign']) + len(demo_data['attacks'])} samples")
```

## Attack Patterns

### Slow Exfiltration
Gradual increase in data transfer over time to evade detection.
- **Characteristics**: 3% gradual increase in outbound bytes
- **Stealth**: Mimics normal traffic variance
- **Detection**: Requires trend analysis

### Living-off-the-Land (LOTL) Mimicry
Periodic micro-bursts that mimic legitimate polling/periodic communication.
- **Characteristics**: 15% spikes at regular intervals (positions 32, 64, 96)
- **Stealth**: Looks like legitimate periodic tasks
- **Detection**: Requires pattern recognition

### Beacon
Regular C2 (Command & Control) communication pattern.
- **Characteristics**: 2% dips at regular 16-step intervals
- **Stealth**: Very subtle, regular pattern
- **Detection**: Requires periodicity analysis

## Feature Columns (12 Total)

1. `flow_duration` - Duration of network flow
2. `fwd_pkts_tot` - Total forward packets
3. `bwd_pkts_tot` - Total backward packets
4. `fwd_data_pkts_tot` - Forward data packets
5. `bwd_data_pkts_tot` - Backward data packets
6. `fwd_pkts_per_sec` - Forward packet rate
7. `bwd_pkts_per_sec` - Backward packet rate
8. `flow_pkts_per_sec` - Overall packet rate
9. `fwd_byts_b_avg` - Average forward bytes
10. `bwd_byts_b_avg` - Average backward bytes
11. `fwd_iat_mean` - Mean inter-arrival time (forward)
12. `bwd_iat_mean` - Mean inter-arrival time (backward)

All features are normalized using StandardScaler (zero mean, unit variance).

## Detection Methods

### Baseline IDS (Threshold-Based)
- **Method**: Statistical thresholding (mean ± 3σ)
- **Threshold**: 30% of features exceeding boundaries
- **Speed**: Very fast (<1ms per sample)
- **Accuracy**: Good for obvious attacks

### Moirai (Foundation Model)
- **Method**: Forecast-based anomaly detection
- **Model**: Salesforce Moirai-1.0-R-small
- **Context**: 100 time steps
- **Prediction**: 28 steps ahead
- **Speed**: Moderate (~100ms per sample)
- **Accuracy**: Better for subtle attacks

## Troubleshooting

### File Not Found
If `demo_data.npz` doesn't exist, generate it:
```bash
python scripts/precompute_demo_data.py
```

### Import Errors
If you get uni2ts import errors:
- Use Python 3.12: `python3.12 -m venv venv`
- Install dependencies: `pip install -r requirements-py312.txt`

### Verification Failures
If verification fails:
1. Regenerate data: `python scripts/precompute_demo_data.py --seed 42`
2. Check logs for specific errors
3. Ensure CICIoT2023 data exists in `data/raw/sample/`

### Low Detection Rates
If both methods detect 0 attacks:
- Try different attack types: `--attack-types protocol_anomaly beacon lotl_mimicry`
- Adjust detection thresholds in config.yaml
- Regenerate with different seed: `--seed 123`

## References

- **Diffusion-TS**: Time-series diffusion model for synthetic data generation
- **Moirai**: Salesforce foundation model for time-series forecasting
- **CICIoT2023**: Canadian Institute for Cybersecurity IoT dataset
