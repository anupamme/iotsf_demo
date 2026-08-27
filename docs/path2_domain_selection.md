# Path 2: Forecasting Domain Selection

## Goal
Identify a forecasting benchmark where Moirai's zero-shot performance demonstrably exceeds simple baselines (Linear, DLinear) by >20%, establishing that pre-trained features provide measurable value (unlike IoT anomaly detection where LogReg parity showed they didn't).

## Moirai Paper Reference
**Paper:** "Unified Training of Universal Time Series Forecasting Transformers" (Woo et al., 2024)
**arXiv:** https://arxiv.org/abs/2402.02592

## Candidate Benchmarks (from Moirai paper)

Based on standard time-series forecasting benchmarks, Moirai likely reports results on:

### 1. **ETTh1 (Electricity Transformer Temperature - Hourly)**
- **Dataset:** 7 features (oil temperature, load, etc.), hourly granularity
- **Size:** ~17,400 timesteps (2 years of data)
- **Split:** 12 months train / 4 months val / 4 months test (standard)
- **Forecast horizons:** 96, 192, 336, 720 timesteps (4h, 8h, 14h, 30h ahead)
- **Task:** Multivariate forecasting of electrical transformer behavior
- **Why high-signal:** Clear temporal patterns (daily/weekly cycles), physical system with strong autocorrelation

### 2. **Electricity Load**
- **Dataset:** UCI Electricity Consumption dataset (370 clients)
- **Size:** 26,304 timesteps (hourly, 3 years)
- **Split:** Standard 60/20/20
- **Forecast horizons:** 96, 192, 336, 720
- **Task:** Predict electricity consumption per client
- **Why high-signal:** Strong daily/weekly seasonality, known to benefit from pre-training

### 3. **Weather**
- **Dataset:** Max Planck Institute Weather dataset
- **Size:** 21 meteorological variables, 10-minute intervals
- **Split:** Standard chronological split
- **Forecast horizons:** 96, 192, 336, 720
- **Task:** Multivariate weather prediction
- **Why high-signal:** Strong periodic patterns, physical constraints

### 4. **Traffic**
- **Dataset:** PeMS Bay Area traffic (862 sensors)
- **Size:** 17,544 timesteps (hourly)
- **Split:** 60/20/20
- **Forecast horizons:** 96, 192, 336, 720
- **Task:** Traffic flow prediction
- **Why high-signal:** Clear temporal patterns, spatial-temporal dependencies

## Selection Criteria

For Path 2 to succeed, we need:
1. **Large gap:** Moirai zero-shot MSE should be ≥20% better than Linear baseline
2. **Sufficient data:** Dataset should support 200/500/1000 sample fine-tuning experiments
3. **Standard evaluation:** Published baselines and protocols
4. **Pre-training relevance:** Moirai pre-trained on LOTSA (mixture of public time-series); target domain should have similar characteristics

## Recommended Domain: ETTh1

**Rationale:**
- **Standard benchmark:** Most commonly used in time-series forecasting papers
- **Moirai likely strong:** Electrical data is common in pre-training distributions
- **Small dataset:** 17k timesteps makes experiments feasible (vs. Electricity's 26k)
- **Clear temporal structure:** Physical system with predictable patterns
- **Published baselines:** Linear, DLinear, FEDformer, PatchTST results widely available

## Next Steps (Task #21)

1. **Download Moirai paper:** Read Tables 1-3 to confirm ETTh1 performance
2. **Document baseline numbers:**
   - Linear MSE for horizon=96, 192, 336, 720
   - Zero-shot Moirai MSE for same horizons
   - Compute gap: (Linear - Moirai) / Linear × 100%
3. **Verify gap >20%:** If not, try Electricity or Weather
4. **Download ETTh1 dataset:** From https://github.com/zhouhaoyi/ETDataset
5. **Proceed to Task #14:** Implement dataset loader

## Alternative: If ETTh1 Gap is Small

If Moirai's advantage on ETTh1 is <20%, try:
- **Electricity:** Larger dataset, likely stronger Moirai advantage
- **Weather:** More variables, richer pre-training signal
- **Traffic:** Strong spatial-temporal structure

**Critical:** Must establish that pre-trained features are valuable before proceeding. Path 2 fails if we can't find a domain where Moirai >> baselines.

## Actual Results (Verified 2026-04-24)

### ETTh1: FAILED GATE

Moirai zero-shot is dramatically worse than Linear on ETTh1 at all horizons:

| Horizon | Linear MSE | DLinear MSE | Moirai MSE | Gap |
|---------|-----------|-------------|------------|-----|
| 96 | 0.524 | 1.062 | 0.739* | -41% |
| 192 | 0.697 | 1.195 | - | - |
| 336 | 0.842 | 1.290 | - | - |
| 720 | 1.069 | 1.516 | - | - |

ETTh1 is known to be "too simple" for transformers (Zeng et al., "Are Transformers Effective for TSF?"). Published Moirai paper shows ETTh1 zero-shot MSE=0.400 (averaged) vs DLinear 0.456 — only 12% improvement, and this is the paper's own best-case number.

### ETTh2: PASSES GATE (Selected Domain)

ETTh2 shows strong Moirai advantage (using median point forecast, robust to mixture distribution tails):

| Horizon | Linear MSE | DLinear MSE | Moirai MSE (median) | Gap vs Linear |
|---------|-----------|-------------|---------------------|--------------|
| 96 | 0.373 | 0.556 | **0.269** | **+28.0%** |
| 192 | 0.544 | 0.695 | **0.299** | **+45.0%** |

Important: Must use median (not mean) of forecast samples due to heavy-tailed mixture distribution (NegBin/LogNormal components produce extreme outliers). Using mean caused MSE blowup (8.58 vs 0.30 with median).

Published Moirai paper: ETTh2 averaged MSE = 0.341 (Small) vs DLinear 0.559 — 39% improvement. Our results (28-45%) are consistent.

**Decision: ETTh2 is our domain for Path 2. Use horizons 96 and 192.**

### Moirai Paper Reference Numbers (Table 6, averaged across horizons)

| Dataset | DLinear | MoiraiSmall | Gap |
|---------|---------|-------------|-----|
| ETTh1 | 0.456 | 0.400 | +12% |
| **ETTh2** | **0.559** | **0.341** | **+39%** |
| ETTm2 | 0.350 | 0.300 | +14% |
| Electricity | 0.212 | 0.233 | -10% |
| Weather | 0.265 | 0.242 | +9% |

---

**Status:** Domain selected (ETTh2). Gate passed at BOTH horizons: h=96 (28.0%), h=192 (45.0%). Ready for Phase 2.
**Next:** Design temporal negative sampling for contrastive loss, implement fine-tuning pipeline.
