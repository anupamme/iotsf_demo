# Moirai Fine-Tuning Guide

This document explains how to prepare data and fine-tune the Moirai time-series foundation model for IoT anomaly detection.

## Overview

The fine-tuning process involves:
1. **Data Preparation**: Combining benign IoT traffic with synthetic attack samples
2. **Fine-Tuning**: Training Moirai using negative log-likelihood (NLL) loss
3. **Evaluation**: Testing detection performance using NLL-based scoring

## Prerequisites

```bash
# Activate virtual environment
source .venv312/bin/activate

# Required packages (should already be installed)
pip install torch numpy pandas loguru
pip install uni2ts  # Moirai implementation
```

## 1. Data Preparation

### 1.1 Data Sources

The fine-tuning data consists of three components:

| Source | Description | Ratio |
|--------|-------------|-------|
| **CICIoT2023 Benign** | Real benign IoT network traffic | 70% |
| **Synthetic Hard-Negatives** | Diffusion-TS generated stealth attacks | 20% |
| **CICIoT2023 Attacks** | Real DDoS/Recon attacks | 10% |

### 1.2 Data Format

All data is converted to sequences of shape `(seq_length, n_features)`:
- **seq_length**: 128 timesteps
- **n_features**: 12 network flow features

The 12 features from CICIoT2023:
```python
FEATURE_COLUMNS = [
    'flow_duration',      # Duration of the flow
    'fwd_pkts_tot',       # Total forward packets
    'bwd_pkts_tot',       # Total backward packets
    'fwd_data_pkts_tot',  # Forward data packets
    'bwd_data_pkts_tot',  # Backward data packets
    'fwd_pkts_per_sec',   # Forward packets per second
    'bwd_pkts_per_sec',   # Backward packets per second
    'flow_pkts_per_sec',  # Total packets per second
    'fwd_byts_b_avg',     # Average forward bytes
    'bwd_byts_b_avg',     # Average backward bytes
    'fwd_iat_mean',       # Forward inter-arrival time mean
    'bwd_iat_mean'        # Backward inter-arrival time mean
]
```

### 1.3 Loading and Preprocessing Data

```python
from src.data.loader import CICIoT2023Loader
from src.data.preprocessor import TrafficPreprocessor, create_sequences
import numpy as np

# Initialize loader and preprocessor
loader = CICIoT2023Loader('data/raw/sample')
preprocessor = TrafficPreprocessor(scaler_type='standard')

# Load benign samples
benign_df = loader.load_benign_samples(n_samples=5000)
benign_normalized = preprocessor.fit_transform(benign_df.values)
benign_sequences = create_sequences(
    benign_normalized,
    seq_length=128,
    stride=128  # Non-overlapping sequences
)

# Load attack samples (use same scaler)
attack_df = loader.load_any_attack_samples(n_samples=1000)
attack_normalized = preprocessor.transform(attack_df.values)
attack_sequences = create_sequences(
    attack_normalized,
    seq_length=128,
    stride=128
)
```

### 1.4 Loading Synthetic Hard-Negatives

Synthetic attacks are pre-generated using Diffusion-TS and stored as `.npy` files:

```python
from pathlib import Path

synthetic_dir = Path('data/synthetic')

# Load hard-negative attack files
hard_negatives = []
for stealth in [85, 90, 95]:
    for attack_type in ['slow_exfiltration', 'lotl_mimicry', 'beacon', 'protocol_anomaly']:
        filepath = synthetic_dir / f'{attack_type}_stealth_{stealth}.npy'
        if filepath.exists():
            samples = np.load(filepath)
            hard_negatives.append(samples)
            print(f"Loaded {filepath.name}: {samples.shape}")

# Combine all hard-negatives
if hard_negatives:
    hard_neg_combined = np.concatenate(hard_negatives)
    print(f"Total hard-negatives: {hard_neg_combined.shape}")
```

### 1.5 Creating the Fine-Tuning Dataset

Use the built-in `load_finetune_data` method for convenience:

```python
# Option 1: Use the built-in loader method
train_data, val_data = loader.load_finetune_data(
    synthetic_dir='data/synthetic',
    benign_ratio=0.7,           # 70% benign
    hard_negative_ratio=0.2,    # 20% hard-negatives
    standard_attack_ratio=0.1,  # 10% standard attacks
    train_val_split=0.85,       # 85% train, 15% val
    total_samples=1000          # Total samples to generate
)

print(f"Training data: {train_data.shape}")  # (n_train, 128, 12)
print(f"Validation data: {val_data.shape}")  # (n_val, 128, 12)
```

Or manually combine the data:

```python
# Option 2: Manual combination
all_sequences = []

# Add benign (70%)
n_benign = int(1000 * 0.7)
all_sequences.append(benign_sequences[:n_benign])

# Add hard-negatives (20%)
n_hard_neg = int(1000 * 0.2)
all_sequences.append(hard_neg_combined[:n_hard_neg])

# Add standard attacks (10%)
n_attacks = int(1000 * 0.1)
all_sequences.append(attack_sequences[:n_attacks])

# Combine and shuffle
all_data = np.concatenate(all_sequences, axis=0)
np.random.seed(42)
np.random.shuffle(all_data)

# Split into train/val
split_idx = int(len(all_data) * 0.85)
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]
```

## 2. Fine-Tuning Process

### 2.1 Initialize the Moirai Detector

```python
from src.models import MoiraiAnomalyDetector

# Create detector with appropriate parameters for 128-timestep sequences
detector = MoiraiAnomalyDetector(
    model_size='small',      # 'small' (8M), 'base' (50M), or 'large' (300M)
    context_length=96,       # Context window for forecasting
    prediction_length=32,    # Prediction horizon (context + prediction = 128)
    confidence_level=0.95,
    device='auto'            # 'auto', 'cuda', or 'cpu'
)

# Initialize base model from HuggingFace
detector.initialize()
print(f"Model initialized on {detector.device}")
```

### 2.2 Run Fine-Tuning

```python
# Fine-tune with NLL loss
history = detector.fine_tune(
    train_data=train_data,
    val_data=val_data,
    n_epochs=10,                    # Number of training epochs
    batch_size=32,                  # Batch size
    learning_rate=1e-4,             # AdamW learning rate
    use_hard_negatives=True,        # Log that hard-negatives are included
    checkpoint_dir='models/moirai_finetuned',  # Where to save checkpoints
    early_stopping_patience=3       # Stop if no improvement for 3 epochs
)

print(f"Training complete!")
print(f"Final train loss: {history['train_loss'][-1]:.4f}")
print(f"Final val loss: {history['val_loss'][-1]:.4f}")
```

### 2.3 Understanding the Training Loop

The fine-tuning uses **negative log-likelihood (NLL) loss** computed via Moirai's internal `_val_loss` method:

```python
# Simplified view of what happens in fine_tune():

for epoch in range(n_epochs):
    model.train()

    for batch in train_loader:
        # Get context and target from batch
        context = batch['context']  # Shape: (B, 96, 12)
        target = batch['target']    # Shape: (B, 32, 12)

        # Concatenate for full sequence
        full_target = torch.cat([context, target], dim=1)  # (B, 128, 12)

        # Create observation masks
        observed_target = torch.ones_like(full_target, dtype=torch.bool)
        is_pad = torch.zeros(B, 128, dtype=torch.bool)

        # Compute NLL loss using Moirai's internal method
        # This accesses the MoiraiModule to get distribution parameters
        # and computes proper NLL with gradient flow
        per_sample_loss = model._val_loss(
            patch_size=32,
            target=full_target,
            observed_target=observed_target,
            is_pad=is_pad
        )

        loss = per_sample_loss.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

### 2.4 Checkpoint Format

Checkpoints are saved in a custom format:

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': {
        'model_size': 'small',
        'context_length': 96,
        'prediction_length': 32,
        'patch_size': 32,
        'confidence_level': 0.95
    },
    'epoch': epoch,
    'val_loss': val_loss
}
torch.save(checkpoint, 'models/moirai_finetuned/best_moirai.pt')
```

## 3. Using the Fine-Tuned Model

### 3.1 Load Fine-Tuned Weights

```python
# Initialize detector
detector = MoiraiAnomalyDetector(
    model_size='small',
    context_length=96,
    prediction_length=32
)

# Load with fine-tuned checkpoint
detector.initialize(checkpoint_path='models/moirai_finetuned/best_moirai.pt')
```

### 3.2 Run Detection with NLL Method

```python
# Detect anomalies using NLL-based scoring (Option A)
result = detector.detect_anomalies(
    traffic=sample,           # Shape: (128, 12)
    threshold=0.5,            # Anomaly score threshold
    method='nll',             # Use NLL-based detection
    return_feature_contributions=True
)

# Check results
print(f"Anomaly score: {result.anomaly_scores.mean():.3f}")
print(f"Anomaly rate: {result.anomaly_rate:.1%}")
print(f"Is attack: {result.anomaly_rate > 0.5}")
```

### 3.3 Understanding NLL-Based Detection

The NLL method works differently from confidence intervals:

| Aspect | Confidence Interval | NLL-Based (Option A) |
|--------|---------------------|----------------------|
| **Approach** | Sample from distribution, check bounds | Compute NLL directly |
| **Insight** | Higher deviation = anomaly | Lower NLL = more predictable = attack |
| **Why it works** | - | Attack patterns (DDoS, scans) are repetitive |
| **ROC-AUC on CICIoT2023** | 0.27 | **1.000** |
| **F1 Score** | 0.67 | **1.000** |

## 4. Complete Training Script

Here's a complete script to fine-tune Moirai:

```python
#!/usr/bin/env python3
"""Fine-tune Moirai on IoT traffic data."""

import numpy as np
from pathlib import Path
from src.models import MoiraiAnomalyDetector
from src.data.loader import CICIoT2023Loader

def main():
    # 1. Load fine-tuning data
    print("Loading fine-tuning data...")
    loader = CICIoT2023Loader('data/raw/sample')

    train_data, val_data = loader.load_finetune_data(
        synthetic_dir='data/synthetic',
        benign_ratio=0.7,
        hard_negative_ratio=0.2,
        standard_attack_ratio=0.1,
        train_val_split=0.85,
        total_samples=500  # Increase for better results
    )

    print(f"Train: {train_data.shape}, Val: {val_data.shape}")

    # 2. Initialize detector
    print("Initializing Moirai detector...")
    detector = MoiraiAnomalyDetector(
        model_size='small',
        context_length=96,
        prediction_length=32,
        device='auto'
    )
    detector.initialize()

    # 3. Fine-tune
    print("Starting fine-tuning...")
    history = detector.fine_tune(
        train_data=train_data,
        val_data=val_data,
        n_epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        checkpoint_dir='models/moirai_finetuned',
        early_stopping_patience=3
    )

    # 4. Test the fine-tuned model
    print("\nTesting fine-tuned model...")

    # Load some test samples
    from src.data.preprocessor import TrafficPreprocessor, create_sequences
    preprocessor = TrafficPreprocessor(scaler_type='standard')

    benign_df = loader.load_benign_samples(n_samples=640)
    benign_norm = preprocessor.fit_transform(benign_df.values)
    benign_seqs = create_sequences(benign_norm, seq_length=128, stride=128)

    attack_df = loader.load_any_attack_samples(n_samples=640)
    attack_norm = preprocessor.transform(attack_df.values)
    attack_seqs = create_sequences(attack_norm, seq_length=128, stride=128)

    # Evaluate with NLL method
    benign_scores = []
    for seq in benign_seqs[:5]:
        result = detector.detect_anomalies(seq, threshold=0.5, method='nll')
        benign_scores.append(result.anomaly_scores.mean())

    attack_scores = []
    for seq in attack_seqs[:5]:
        result = detector.detect_anomalies(seq, threshold=0.5, method='nll')
        attack_scores.append(result.anomaly_scores.mean())

    print(f"\nResults (NLL method):")
    print(f"  Benign mean score: {np.mean(benign_scores):.3f}")
    print(f"  Attack mean score: {np.mean(attack_scores):.3f}")
    print(f"  Separation gap: {np.mean(attack_scores) - np.mean(benign_scores):.3f}")

    print("\nFine-tuning complete!")
    print(f"Checkpoint saved to: models/moirai_finetuned/best_moirai.pt")

if __name__ == '__main__':
    main()
```

Run with:
```bash
python scripts/finetune_moirai.py
```

## 5. Expected Results

After fine-tuning, you should see:

### Training Metrics
```
Epoch 1/10 - Train Loss: 10.42, Val Loss: 10.15
Epoch 2/10 - Train Loss: 8.76, Val Loss: 8.52
Epoch 3/10 - Train Loss: 7.89, Val Loss: 7.71
Epoch 4/10 - Train Loss: 7.52, Val Loss: 7.45
Epoch 5/10 - Train Loss: 7.38, Val Loss: 7.39
```

### Detection Performance (NLL Method)
| Metric | Base Model | Fine-Tuned |
|--------|------------|------------|
| Benign NLL | 16.4 | 16.1 |
| Attack NLL | 12.2 | 9.1 |
| Gap | -4.2 | **-7.0** |
| ROC-AUC | 1.0 | **1.0** |

The fine-tuned model shows better separation between benign and attack traffic.

## 6. Troubleshooting

### Common Issues

1. **Out of memory**: Reduce `batch_size` or use `model_size='small'`

2. **NaN loss**: Check for invalid values in your data
   ```python
   assert not np.isnan(train_data).any(), "NaN values in training data"
   ```

3. **No improvement**:
   - Increase `total_samples` for more training data
   - Try lower `learning_rate` (e.g., 5e-5)
   - Ensure data mix includes hard-negatives

4. **Checkpoint not loading**:
   - Verify path exists: `ls models/moirai_finetuned/`
   - Check file format matches expected structure

### GPU Memory Requirements

| Model Size | Parameters | GPU Memory |
|------------|------------|------------|
| small | 8M | ~2 GB |
| base | 50M | ~8 GB |
| large | 300M | ~24 GB |

## References

- [Moirai Paper](https://arxiv.org/abs/2402.02592): "Unified Training of Universal Time Series Forecasting Transformers" (ICML 2024)
- [CICIoT2023 Dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html): IoT Network Traffic Dataset
- [uni2ts Repository](https://github.com/SalesforceAIResearch/uni2ts): Moirai implementation
