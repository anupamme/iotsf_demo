# Anonymous Supplementary Code: Reproducibility Bundle

This anonymous supplementary bundle reproduces the main diagnostic pipeline:
value-gate computation, Moirai fine-tuning, CKA, task-orthogonal probes,
forgetting metrics, and table generation.

**The full cleaned repository will be released upon acceptance.**

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.12 required for uni2ts)
pip install -r requirements.txt
pip install "uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git"

# 2. Apply gradient patch (required for fine-tuning)
cd $(python -c "import uni2ts; print(uni2ts.__path__[0])")/../
git apply /path/to/patches/uni2ts_packed_std_scaler.patch

# 3. Download data
python scripts/00_download_or_prepare_data.py

# 4. Run minimal reproduction (single config, single seed)
bash scripts/run_minimal_repro.sh configs/etth2_small_n500.yaml

# 5. Run full 10-seed sweep
bash scripts/run_minimal_repro.sh configs/etth2_small_n10k.yaml auto seeds/etth2_small_n10k_seeds.txt
```

---

## Pipeline Overview

| Step | Script | What it does |
|------|--------|--------------|
| 0 | `00_download_or_prepare_data.py` | Downloads ETTh2, ILI, M4 datasets |
| 1 | `01_value_gate.py` | Computes R^2_task = 1 - MSE_ZS/MSE_Linear |
| 2 | `02_finetune_moirai.py` | Fine-tunes with full diagnostic protocol |
| 3 | `03_compute_cka.py` | Summarizes CKA between pre/post representations |
| 4 | `04_run_probes.py` | Linear probes for Delta R^2 |
| 5 | `05_compute_forgetting.py` | Aggregates forgetting statistics |
| 6 | `06_make_tables.py` | Generates paper table CSVs |

---

## Reproducing Specific Tables

### Table 2: Sample-size sweep (ETTh2, condition B, h=96)
```bash
python scripts/02_finetune_moirai.py --config configs/etth2_small_n500.yaml \
    --condition B --seeds-file seeds/etth2_small_n10k_seeds.txt --device cuda
python scripts/06_make_tables.py --input runs/ --output expected_outputs/
```

### Table 4: Cross-domain diagnostic summary
Run all configs:
```bash
for cfg in configs/etth2_small_n500.yaml configs/ili_small.yaml configs/chronos_m4.yaml; do
    bash scripts/run_minimal_repro.sh "$cfg" cuda seeds/etth2_small_n10k_seeds.txt
done
python scripts/06_make_tables.py --input runs/ --output expected_outputs/
```

---

## Hardware and Runtime

- **GPU**: NVIDIA A10G (24 GB) — results in paper produced on this architecture
- **Estimated runtimes**:
  - Single config, 4 conditions, 1 seed: ~15 minutes
  - Full 10-seed sweep (one dataset): ~2.5 hours
  - All datasets, all seeds: ~8 hours

---

## Determinism Caveat

Exact deterministic replication is verified on the same GPU architecture
(NVIDIA A10G, Ampere). Results on other architectures (e.g., V100, H100) may
differ at the 4th decimal place due to non-associative floating-point
operations in cuDNN/cuBLAS. The qualitative conclusions (harmful drift at
low-n, stability at frozen-encoder, capacity-buffer at base) are robust
across architectures.

All scripts use `--deterministic` mode by default (fixed seeds, deterministic
algorithms, reproducible DataLoader ordering).

---

## Key Metric Definitions

- **R^2_task** = 1 - MSE_ZS / MSE_Linear (value gate; >0 means gate-pass)
- **Forgetting %** = 100 * (MSE_final - MSE_ZS) / MSE_ZS (>0 = degradation)
- **CKA** = Linear Centered Kernel Alignment (1.0 = identical representations)
- **Delta R^2** = R^2_post - R^2_pre (probe quality change; >0 = beneficial)
- **Weight drift** = L2 distance from pre-trained parameters

---

## Patch Details

The file `patches/uni2ts_packed_std_scaler.patch` fixes an in-place tensor
operation bug in uni2ts 2.0.0 that prevents gradient-based fine-tuning:

```python
# Before (breaks autograd):
loc[sample_id == 0] = 0
scale[sample_id == 0] = 1

# After (gradient-safe):
sample_id_mask = (sample_id == 0).unsqueeze(-1)
loc = torch.where(sample_id_mask, torch.zeros_like(loc), loc)
scale = torch.where(sample_id_mask, torch.ones_like(scale), scale)
```

The same fix is applied at runtime via `src/models.py:apply_uni2ts_gradient_patch()`.

---

## Directory Structure

```
.
├── README_REPRO.md          # This file
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT (anonymous)
├── configs/                 # Experiment configurations
├── scripts/                 # Numbered pipeline scripts
├── patches/                 # uni2ts gradient fix
├── src/                     # Core library (CKA, probes, metrics, models)
├── seeds/                   # Per-experiment seed lists
└── expected_outputs/        # Reference CSVs matching paper tables
```
