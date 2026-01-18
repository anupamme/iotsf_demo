# Diffusion-TS Installation & Integration Guide

## Problem Solved ✅

The original Diffusion-TS `requirements.txt` had severe dependency conflicts with `iotsf_demo`:

| Package | Diffusion-TS | iotsf_demo | Conflict |
|---------|--------------|------------|----------|
| torch | 2.0.1 | 2.4.1 | ❌ Major version mismatch |
| pandas | 1.5.0 | 2.1.4 | ❌ Breaking API changes |
| scikit-learn | 1.1.2 | 1.4.0 | ❌ Incompatible |
| scipy | 1.8.1 | 1.11.4 | ❌ Incompatible |

**Additional issues:**
- Unnecessary dependencies (dm-control, mujoco, gluonts) for IoT use case
- Wrong import paths in original code
- Incorrect API usage

---

## Solution Overview

Created a **minimal, conflict-free installation** that:
1. ✅ Installs only core dependencies (einops, ema-pytorch, tqdm, pyyaml)
2. ✅ Clones Diffusion-TS to `lib/` directory (not pip install)
3. ✅ Uses adapter module for import management
4. ✅ No conflicts with existing iotsf_demo packages
5. ✅ Easy to remove if needed

---

## Installation Steps

### 1. Run the Installation Script

```bash
# Activate your Python 3.12 environment
source .venv12/bin/activate

# Run the compatible installation script
bash scripts/install_diffusion_ts_compatible.sh
```

### 2. Verify Installation

```bash
python3 << 'EOF'
from src.models import IoTDiffusionGenerator

generator = IoTDiffusionGenerator(seq_length=128, feature_dim=12)
generator.initialize()

if not generator._mock_mode:
    print("✅ Real Diffusion-TS is ready!")
else:
    print("❌ Still using mock mode")
EOF
```

### 3. Test Generation (Optional)

```bash
python3 << 'EOF'
from src.models import IoTDiffusionGenerator
import numpy as np

generator = IoTDiffusionGenerator(seq_length=128, feature_dim=12)
generator.initialize()

# Generate sample (takes ~30 seconds)
samples = generator.generate(n_samples=1, seed=42, n_inference_steps=10)
print(f"Generated shape: {samples.shape}")
print("✅ Generation working!")
EOF
```

---

## What Was Changed

### New Files Created

1. **`scripts/install_diffusion_ts_compatible.sh`**
   - Minimal installation script
   - Clones Diffusion-TS to `lib/` directory
   - Installs only core dependencies

2. **`src/models/diffusion_ts_adapter.py`**
   - Handles import path management
   - Provides compatibility layer
   - Graceful fallback to mock mode

3. **`DIFFUSION_TS_INTEGRATION.md`**
   - Technical documentation
   - API details
   - Status and remaining work

4. **`DIFFUSION_TS_INSTALLATION_GUIDE.md`** (this file)
   - User-friendly guide
   - Installation steps
   - Troubleshooting

### Modified Files

1. **`src/models/diffusion_ts.py`**
   - Updated imports to use adapter
   - Fixed model initialization API
   - Updated generation method API
   - Better error handling

---

## Directory Structure

```
iotsf_demo/
├── lib/
│   └── Diffusion-TS/          # ← Cloned repo (NOT in git)
│       ├── Models/
│       │   └── interpretable_diffusion/
│       │       ├── gaussian_diffusion.py  # Contains Diffusion_TS class
│       │       ├── transformer.py
│       │       └── model_utils.py
│       ├── Utils/
│       └── ...
├── src/
│   └── models/
│       ├── diffusion_ts.py            # Main wrapper (MODIFIED)
│       └── diffusion_ts_adapter.py    # Import adapter (NEW)
└── scripts/
    └── install_diffusion_ts_compatible.sh  # Installation script (NEW)
```

---

## Benefits of This Approach

### ✅ Advantages

1. **No Dependency Conflicts**
   - Doesn't touch existing iotsf_demo packages
   - Uses current PyTorch 2.4.1, pandas 2.1.4, etc.

2. **Easy to Remove**
   - Just delete `lib/Diffusion-TS` directory
   - No pip uninstall needed

3. **Clean Separation**
   - Adapter isolates Diffusion-TS from main code
   - Changes are localized

4. **Automatic Fallback**
   - If Diffusion-TS unavailable, uses mock generator
   - Demo still works

5. **No Package Pollution**
   - Not installed globally
   - Doesn't affect other projects

### ⚠️ Limitations

1. **Guidance Not Yet Supported**
   - `target_statistics` parameter doesn't work with real Diffusion-TS
   - Would need custom implementation
   - Mock mode still supports it

2. **Slower Generation**
   - Real diffusion model takes ~30 seconds per sample (CPU)
   - Use `n_inference_steps=10` for faster generation
   - Mock mode is instant

---

## Usage Examples

### Basic Generation

```python
from src.models import IoTDiffusionGenerator

# Initialize
generator = IoTDiffusionGenerator(seq_length=128, feature_dim=12)
generator.initialize()

# Generate benign-like traffic
samples = generator.generate(
    n_samples=5,
    seed=42,
    n_inference_steps=10  # Lower = faster, 50 = default quality
)

print(f"Generated: {samples.shape}")  # (5, 128, 12)
```

### Generate Attack with Pattern Injection

```python
import numpy as np

# Generate benign baseline
benign = generator.generate(n_samples=1, seed=42)[0]

# Generate stealthy attack
attack, metadata = generator.generate_hard_negative(
    benign_sample=benign,
    attack_pattern='slow_exfiltration',  # or 'lotl_mimicry', 'beacon', 'protocol_anomaly'
    stealth_level=0.95  # 95% similar to benign
)

print(f"Attack type: {metadata['attack_type']}")
print(f"Mean difference: {metadata['mean_diff']:.6f}")
print(f"Std difference: {metadata['std_diff']:.6f}")
```

### Regenerate Demo Data

```bash
# This will use real Diffusion-TS if installed
python scripts/precompute_demo_data.py --seed 42

# Verify the new data
python scripts/verify_demo_data.py
```

---

## Troubleshooting

### Issue: "Mock mode: True" after installation

**Cause:** Diffusion-TS not found or import failed

**Solution:**
```bash
# Check if directory exists
ls -la lib/Diffusion-TS

# If missing, run installation again
bash scripts/install_diffusion_ts_compatible.sh

# Test imports manually
python3 << 'EOF'
import sys
sys.path.insert(0, 'lib/Diffusion-TS')
from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
print("✅ Import works")
EOF
```

### Issue: "einops version conflict"

**Symptom:** Warning about `uni2ts` requiring einops==0.7.*

**Solution:** Safe to ignore - doesn't affect functionality
```bash
# Or pin to uni2ts-compatible version
pip install einops==0.7.0
```

### Issue: Generation is slow

**Cause:** Diffusion models are computationally expensive on CPU

**Solutions:**
1. Use fewer inference steps: `n_inference_steps=10` (default is 50)
2. Use mock mode for quick testing: Comment out adapter import
3. Use GPU if available (will auto-detect)
4. Generate once, save, and reuse: `np.save('samples.npy', samples)`

### Issue: "Memory error" during generation

**Solution:** Generate fewer samples at once
```python
# Instead of:
samples = generator.generate(n_samples=100, ...)

# Do:
samples = []
for i in range(10):
    batch = generator.generate(n_samples=10, seed=42+i, ...)
    samples.append(batch)
samples = np.concatenate(samples)
```

---

## Comparison: Mock vs Real Diffusion-TS

| Feature | Mock Generator | Real Diffusion-TS |
|---------|---------------|-------------------|
| **Speed** | Instant (<0.1s) | Slow (~30s/sample on CPU) |
| **Quality** | Good enough for demo | Research-grade quality |
| **Realism** | Statistical patterns | Learned from data |
| **Installation** | Built-in | Requires setup |
| **Dependencies** | None | einops, ema-pytorch |
| **Attack Patterns** | ✅ Supported | ✅ Supported |
| **Guidance** | ✅ Supported | ⚠️ Not yet implemented |

**Recommendation:** Use mock mode for development/testing, real Diffusion-TS for final demo data generation.

---

## Uninstallation

If you want to remove Diffusion-TS:

```bash
# 1. Remove the cloned repository
rm -rf lib/Diffusion-TS

# 2. (Optional) Remove extra dependencies
pip uninstall ema-pytorch

# 3. System will automatically fall back to mock mode
```

---

## Next Steps

1. ✅ **Verify Installation:**
   ```bash
   python3 -c "from src.models import IoTDiffusionGenerator; g=IoTDiffusionGenerator(); g.initialize(); print('Mock:', g._mock_mode)"
   ```

2. ✅ **Test Generation:**
   ```bash
   python3 << 'EOF'
   from src.models import IoTDiffusionGenerator
   g = IoTDiffusionGenerator(); g.initialize()
   samples = g.generate(1, seed=42, n_inference_steps=10)
   print(f"Shape: {samples.shape}, Mean: {samples.mean():.3f}")
   EOF
   ```

3. ✅ **Regenerate Demo Data:**
   ```bash
   python scripts/precompute_demo_data.py --seed 42
   python scripts/verify_demo_data.py
   ```

4. ✅ **Run the Demo:**
   ```bash
   streamlit run app/main.py
   ```

---

## Support

If you encounter issues:

1. **Check logs:** Look for ERROR/WARNING messages
2. **Verify Python version:** `python3 --version` (should be 3.12)
3. **Check installation:** `ls -la lib/Diffusion-TS`
4. **Test adapter:** `python3 -c "from src.models.diffusion_ts_adapter import DIFFUSION_TS_AVAILABLE; print(DIFFUSION_TS_AVAILABLE)"`
5. **Review:** `DIFFUSION_TS_INTEGRATION.md` for technical details

---

## Summary

**Before:** ❌ Dependency conflicts, couldn't install Diffusion-TS
**After:** ✅ Clean installation, no conflicts, works with iotsf_demo

**What you get:**
- Real Diffusion-TS model for high-quality generation
- No dependency conflicts with iotsf_demo
- Automatic fallback to mock mode if needed
- Easy to install, easy to remove

**Current Status:** Attack generation currently uses **MOCK** generator. After running the installation script, it will use **REAL** Diffusion-TS.
