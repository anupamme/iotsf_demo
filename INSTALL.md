# Installation Guide

## Choose Your Python Version

### Python 3.12 (RECOMMENDED)
✅ **Full compatibility** including Moirai (uni2ts)
✅ All packages have prebuilt wheels
✅ Tested and verified

### Python 3.13+
⚠️ Core features work, but Moirai (uni2ts) **NOT available**
✅ All other packages work perfectly
⚠️ Use this only if you don't need Moirai

---

## Installation for Python 3.12 (Full Features)

### Quick Start

1. **Check your Python version:**
   ```bash
   python3.12 --version
   # Should show: Python 3.12.x
   ```

2. **Create a virtual environment:**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install all dependencies (including Moirai):**
   ```bash
   pip install --upgrade pip
   pip install -r requirements-py312.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import uni2ts; print(f'✅ Moirai (uni2ts) {uni2ts.__version__} installed')"
   python -c "from src.utils.device import get_device_info; print(get_device_info())"
   ```

5. **Run the application:**
   ```bash
   streamlit run app/main.py
   ```

### Package Versions (Python 3.12)

When you use `requirements-py312.txt`, you'll get:
- ✅ **numpy 1.26.4** - Compatible with uni2ts
- ✅ **pandas 2.1.4** - Compatible version
- ✅ **scipy 1.11.4** - Prebuilt wheel (no compilation!)
- ✅ **torch 2.4.1** - Stable version
- ✅ **uni2ts 2.0.0** - Moirai foundation model
- ✅ **streamlit 1.52.2** - Latest
- ✅ All other core packages

---

## Installation for Python 3.13+ (Without Moirai)

### Quick Start

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "from src.utils.device import get_device_info; print(get_device_info())"
   ```

4. **Run the application:**
   ```bash
   streamlit run app/main.py
   ```

### Package Versions (Python 3.13+)

When you use `requirements.txt`, you'll get:
- ✅ **numpy 2.4.0** - Latest version
- ✅ **pandas 2.3.3** - Latest
- ✅ **scipy 1.16.3** - Latest
- ✅ **torch 2.9.1** - Latest
- ✅ **streamlit 1.52.2** - Latest
- ⚠️ **No uni2ts** - Not compatible

---

## Why Python 3.12?

**The Issue:** Moirai (uni2ts) requires `scipy~=1.11.4`

- **Python 3.12:** scipy 1.11.4 has prebuilt wheels ✅
- **Python 3.13:** scipy 1.11.4 must compile from source, fails with Cython errors ❌
- **Python 3.14:** scipy 1.11.4 must compile from source, fails with missing dependencies ❌

**Solution:** Use Python 3.12 for Moirai, or skip Moirai and use Python 3.13+

---

## Installing Python 3.12

### macOS (Homebrew)
```bash
brew install python@3.12
```

### Ubuntu/Debian
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
```

### Using pyenv
```bash
pyenv install 3.12.7
pyenv local 3.12.7
```

### Windows
Download from [python.org](https://www.python.org/downloads/)

---

## Troubleshooting

### "command not found: python3.12"
Install Python 3.12 using one of the methods above.

### ImportError: No module named 'uni2ts'
You're using Python 3.13+ with `requirements.txt`. Either:
- Switch to Python 3.12 and use `requirements-py312.txt`
- Or continue without Moirai (core app still works)

### "externally-managed-environment" error
Always use a virtual environment:
```bash
python3.12 -m venv venv
source venv/bin/activate
```

### scipy compilation errors
Don't try to compile scipy manually. Use Python 3.12 which has prebuilt wheels.

---

## Testing Your Installation

### Test Core Functionality
```bash
python -c "from src.utils.device import get_device_info; print(get_device_info())"
```

### Test Moirai (Python 3.12 only)
```bash
python -c "import uni2ts; print(f'Moirai version: {uni2ts.__version__}')"
```

### Run Test Suite
```bash
pytest tests/ -v
```

Expected output:
```
tests/test_device.py::TestGPUUtils::test_get_device_returns_device PASSED
tests/test_device.py::TestGPUUtils::test_is_cuda_available_returns_bool PASSED
tests/test_device.py::TestGPUUtils::test_get_device_info_returns_dict PASSED
tests/test_device.py::TestGPUUtils::test_cpu_fallback PASSED
```

---

## Performance Notes

### CPU vs GPU
The application automatically detects GPU availability:
- **GPU available:** Uses CUDA for acceleration
- **No GPU:** Falls back to CPU (slower but functional)

Check your device:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Summary

| Python Version | Moirai Support | Core Features | Recommendation |
|---------------|----------------|---------------|----------------|
| **3.12** | ✅ Yes | ✅ Yes | **Use this!** |
| 3.13-3.14 | ❌ No | ✅ Yes | Only if you don't need Moirai |
| 3.9-3.11 | ⚠️ Maybe | ⚠️ Maybe | Not tested, use 3.12 instead |

**Bottom Line:** Use Python 3.12 for the best experience!
