# IoT Security Demo

Interactive demonstration of IoT security anomaly detection using time series foundation models.

## Features

- 🔍 Interactive "Spot the Attack" challenge
- 🎯 Comparison of traditional IDS vs. modern ML approaches
- 🤖 **Diffusion-TS for synthetic attack generation** (mock mode included!)
- 📊 Moirai foundation model for anomaly detection
- 📈 Real-time visualization with Plotly

## Quick Start

### Prerequisites
- **Python 3.12** (recommended for full compatibility with Moirai)
- Python 3.13-3.14 supported (but excludes uni2ts/Moirai)
- GPU with CUDA support (optional, will fall back to CPU)

**Python Version Guide:**
- ✅ **Python 3.12** - Full support including Moirai (uni2ts) - **RECOMMENDED**
- ⚠️ **Python 3.13-3.14** - Core features work, but Moirai (uni2ts) not available
- ⚠️ **Python 3.9-3.11** - May work but not tested

### Installation

1. Clone the repository:
```bash
git clone https://github.com/anupamme/iotsf_demo
cd iotsf_demo
```

2. Create virtual environment:
```bash
# For Python 3.12 (recommended):
python3.12 -m venv venv

# Or for Python 3.13+:
python3 -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# For Python 3.12 (includes Moirai):
pip install -r requirements-py312.txt

# For Python 3.13+ (excludes Moirai):
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app/main.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
iotsf_demo/
├── app/                    # Streamlit application
│   ├── main.py            # Entry point
│   ├── pages/             # Multi-page app pages
│   │   ├── 01_challenge.py
│   │   ├── 02_reveal.py
│   │   ├── 03_traditional.py
│   │   ├── 04_pipeline.py
│   │   └── 05_detection.py
│   └── components/        # Reusable UI components
│       ├── plots.py
│       └── metrics.py
├── src/                   # Core source code
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model implementations
│   └── utils/            # Utilities (config, device, logging)
├── data/                  # Data directory
│   ├── raw/              # CICIoT2023 samples
│   ├── processed/        # Preprocessed data
│   └── synthetic/        # Pre-generated attacks
├── models/               # Saved model weights
├── scripts/              # Standalone scripts
├── tests/                # Tests
├── config/
│   └── config.yaml       # Configuration
└── requirements.txt      # Dependencies
```

## Configuration

Edit `config/config.yaml` to customize:
- Data paths and dataset parameters
- Model configurations (Diffusion-TS, Moirai)
- GPU settings
- Demo parameters (number of samples, attack types)
- Visualization theme

## GPU Support

The application automatically detects GPU availability:
- ✅ Uses GPU if CUDA is available
- ⚠️ Falls back to CPU if GPU unavailable
- Override in config: `device.use_gpu: false`

Check GPU status:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Dataset

This demo uses the **CICIoT2023** dataset for IoT security research.
Download instructions will be added in future updates.

## Diffusion-TS Usage

Generate synthetic attacks:

```python
from src.models import IoTDiffusionGenerator
import numpy as np

# Initialize generator (uses mock mode by default)
generator = IoTDiffusionGenerator(seq_length=128, feature_dim=12)
generator.initialize()

# Generate benign-like traffic
samples = generator.generate(n_samples=5)
print(samples.shape)  # (5, 128, 12)

# Generate hard-negative attack
benign_sample = np.random.randn(128, 12)
attack, metadata = generator.generate_hard_negative(
    benign_sample=benign_sample,
    attack_pattern='slow_exfiltration',
    stealth_level=0.95
)
print(f"Attack type: {metadata['attack_type']}")
print(f"Mean difference: {metadata['mean_diff']:.4f}")
```

### Pre-generate Attacks for Demo

```bash
python scripts/precompute_attacks.py --n-samples 20
```

This generates synthetic attacks in `data/synthetic/`:
- `benign_samples.npy` - Baseline benign traffic
- `slow_exfiltration_stealth_XX.npy` - Slow data exfiltration attacks
- `lotl_mimicry_stealth_XX.npy` - Living-off-the-land mimicry
- `protocol_anomaly_stealth_XX.npy` - Protocol timing anomalies
- `beacon_stealth_XX.npy` - C2 beacon patterns

## Development

Run tests:
```bash
pytest tests/ -v
```

Run Diffusion-TS tests specifically:
```bash
pytest tests/test_diffusion_ts.py -v
```

## License

MIT License

## Citation

If you use this demo, please cite the relevant papers:
- Diffusion-TS: [paper link]
- Moirai: [paper link]
- CICIoT2023: [paper link]
