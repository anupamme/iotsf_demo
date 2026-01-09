# IoT Security Demo

Interactive demonstration of IoT security anomaly detection using time series foundation models.

## Features

- 🔍 Interactive "Spot the Attack" challenge
- 🎯 Comparison of traditional IDS vs. modern ML approaches
- 🤖 Diffusion-TS for synthetic attack generation
- 📊 Moirai foundation model for anomaly detection
- 📈 Real-time visualization with Plotly

## Quick Start

### Prerequisites
- Python 3.9 or higher
- GPU with CUDA support (optional, will fall back to CPU)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/anupamme/iotsf_demo
cd iotsf_demo
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
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

## Development

Run tests:
```bash
pytest tests/ -v
```

## License

MIT License

## Citation

If you use this demo, please cite the relevant papers:
- Diffusion-TS: [paper link]
- Moirai: [paper link]
- CICIoT2023: [paper link]
