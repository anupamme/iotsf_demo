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

## Expected Baseline Numbers (Placeholder - Verify with Paper)

**ETTh1 (horizon=96, MSE):**
- Linear: ~0.400
- DLinear: ~0.380
- Zero-shot Moirai: ~0.300 (expected ~25% improvement over Linear)
- Gap: (0.400 - 0.300) / 0.400 = 25% ✓

**If actual gap is confirmed >20%, ETTh1 is our domain.**

---

**Status:** Research in progress. Need to download and read Moirai paper Tables 1-3.
**Next:** Confirm ETTh1 numbers, download dataset, implement loader (Task #14).
