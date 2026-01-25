# Diffusion-TS Integration Status

## ✅ Successfully Resolved

### 1. Dependency Conflicts
**Problem:** Diffusion-TS requirements.txt had outdated/conflicting versions:
- torch==2.0.1 (iotsf_demo uses 2.4.1)
- pandas==1.5.0 (iotsf_demo uses 2.1.4)
- scikit-learn==1.1.2 (iotsf_demo uses 1.4.0)
- Plus unnecessary dependencies (dm-control, mujoco, gluonts)

**Solution:** Created minimal compatible installation
- Installed only core dependencies: `einops`, `ema-pytorch`, `tqdm`, `pyyaml`
- Cloned Diffusion-TS to `lib/Diffusion-TS` directory
- Created adapter module for path management

### 2. Import Structure
**Problem:** Initial code tried to import non-existent modules:
- `from diffusion_ts.model import DiffusionTS` ❌
- `from diffusion_ts.diffusion import GaussianDiffusion` ❌

**Actual structure:**
- `from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS` ✅
- `from Models.interpretable_diffusion.transformer import Transformer` ✅

**Solution:** Created `src/models/diffusion_ts_adapter.py` to handle imports correctly

### 3. Model Initialization
**Problem:** Incorrect parameter names and structure

**Solution:** Updated to use correct Diffusion_TS API:
```python
model_config = {
    'seq_length': 128,
    'feature_size': 12,  # Note: feature_size, not feature_dim
    'n_layer_enc': 3,
    'n_layer_dec': 6,
    'd_model': 256,
    'timesteps': 1000,
    'sampling_timesteps': None,
    'loss_type': 'l1',
    'beta_schedule': 'cosine'
}
model = Diffusion_TS(**model_config)
```

## ⚠️ Remaining Work

### 4. Generation API (Partial - Needs API Understanding)
**Current Issue:** The `sample()` method has a different signature than expected

**What we know:**
- Model initializes successfully ✅
- Model loads and is ready ✅
- `sample()` method exists but has different parameters ❌

**Next Steps:**
1. Inspect actual `Diffusion_TS.sample()` method signature
2. Update `generate()` method in `diffusion_ts.py` to match
3. Test generation with proper API calls

**Likely needed:**
- Different parameter names
- Different sampling approach (might use `p_sample_loop` or similar)
- May need conditioning/guidance implemented differently

## Installation Instructions

### Quick Install
```bash
source .venv12/bin/activate
bash scripts/install_diffusion_ts_compatible.sh
```

### What Gets Installed
1. **Location:** `lib/Diffusion-TS/` (cloned from GitHub)
2. **Dependencies:** Only core packages (einops, ema-pytorch, tqdm, pyyaml)
3. **No pip install -e:** Uses sys.path management instead to avoid conflicts

### Verification
```bash
python3 << 'EOF'
from src.models import IoTDiffusionGenerator

generator = IoTDiffusionGenerator(seq_length=128, feature_dim=12)
generator.initialize()

print(f"Mock mode: {generator._mock_mode}")
# Should print: Mock mode: False
EOF
```

## Current Capabilities

✅ **Working:**
- Diffusion-TS library imports successfully
- Adapter module handles path management
- Model initialization completes
- No dependency conflicts
- Compatible with Python 3.12 + iotsf_demo environment

⚠️ **Not Yet Working:**
- Generation/sampling (API mismatch)
- Hard-negative generation (depends on sampling)
- Checkpoint loading (not tested)

## For Developers

### File Structure
```
iotsf_demo/
├── lib/
│   └── Diffusion-TS/          # Cloned repo (not in git)
├── src/
│   └── models/
│       ├── diffusion_ts.py            # Main wrapper
│       └── diffusion_ts_adapter.py    # Import adapter (NEW)
└── scripts/
    └── install_diffusion_ts_compatible.sh  # Installation script (NEW)
```

### Key Files Modified
1. `src/models/diffusion_ts.py` - Updated imports and initialization
2. `src/models/diffusion_ts_adapter.py` - New adapter for path/import management
3. `scripts/install_diffusion_ts_compatible.sh` - Compatible installation script

### Mock Mode Fallback
If Diffusion-TS isn't installed or fails, automatically falls back to statistical mock generator:
- trend + seasonality + noise
- Attack pattern injection still works
- Good enough for demo purposes

## Benefits of This Approach

1. **No dependency conflicts** - Doesn't interfere with iotsf_demo packages
2. **Easy to remove** - Just delete `lib/Diffusion-TS` directory
3. **Clean separation** - Adapter isolates Diffusion-TS from main codebase
4. **Fallback ready** - Mock mode works if real model unavailable
5. **No package pollution** - Not installed via pip into site-packages

## Next Actions

To complete integration:
1. Analyze Diffusion_TS.sample() method (see actual implementation)
2. Update generate() method to match Diffusion-TS API
3. Test end-to-end generation
4. Regenerate demo data with real Diffusion-TS:
   ```bash
   python scripts/precompute_demo_data.py --seed 42
   ```

## References

- **Diffusion-TS Repo:** https://github.com/Y-debug-sys/Diffusion-TS
- **Paper:** "Diffusion-TS: Interpretable Diffusion for General Time Series Generation", ICLR 2024
- **Actual Class:** `Models.interpretable_diffusion.gaussian_diffusion.Diffusion_TS`
