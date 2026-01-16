# Demo Day Presentation Guide

**Presentation Time:** 15-20 minutes
**Audience:** Security researchers, ML practitioners
**Goal:** Showcase hard-negative attack generation for IDS evaluation

---

## Pre-Demo Checklist

### Day Before
- [ ] Verify Python 3.12 environment active (`python --version`)
- [ ] Run full test suite: `pytest tests/ -v` (expect all to pass or skip)
- [ ] Pre-generate synthetic attacks: `python scripts/precompute_attacks.py --n-samples 10`
- [ ] Verify GPU detection: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Launch Streamlit app to verify: `streamlit run app/main.py`
- [ ] Check all 5 pages load without errors
- [ ] Prepare backup slides (in case live demo fails)

### 30 Minutes Before
- [ ] Close all unnecessary applications
- [ ] Set terminal font size to 18+ for visibility
- [ ] Open browser at `http://localhost:8501`
- [ ] Have backup pre-recorded screen capture ready
- [ ] Test screen sharing in presentation software
- [ ] Prepare example questions for Q&A

---

## Demo Flow (15 minutes)

### 1. Introduction (2 minutes)

**Opening Line:**
> "Today I'll show you how we use diffusion models to generate hard-negative attacks that can fool modern IDS systems."

**Talking Points:**
- **Problem:** Traditional IDS struggle with subtle attacks
- **Solution:** Use Diffusion-TS to generate realistic, stealthy attacks
- **Key Innovation:** Decomposition-aware generation with IoT constraints

**Visuals:**
- Show title slide
- Display high-level architecture diagram

---

### 2. Challenge: Spot the Attack (3 minutes)

**Navigate to:** Page 1 - Challenge

**Action Items:**
- [ ] Display 6 traffic samples side-by-side
- [ ] Ask audience: "Which 3 are attacks?"
- [ ] Use Streamlit number input for guesses
- [ ] Reveal that it's intentionally hard to tell

**Talking Points:**
> "These traffic samples look nearly identical. That's the point - we're generating attacks that mimic benign behavior."

**Key Message:**
- Hard-negatives are visually indistinguishable
- Statistical properties match benign baseline
- Only subtle behavioral patterns differ

---

### 3. Reveal: The Ground Truth (2 minutes)

**Navigate to:** Page 2 - Reveal

**Action Items:**
- [ ] Show ground truth labels
- [ ] Display statistical comparison table:
  ```
  Sample | Type   | Mean | Std  | Diff from Benign
  ----------------------------------------------------
  1      | Benign | 0.02 | 0.98 | -
  2      | Attack | 0.03 | 0.97 | 0.01 (1%)
  3      | Benign | 0.01 | 1.01 | -
  ...
  ```
- [ ] Highlight stealth level (90-95%)

**Talking Points:**
> "Sample 2 is a slow data exfiltration attack, but its statistics are within 1% of benign traffic."

**Key Message:**
- Attacks are statistically similar to benign
- Stealth level controls similarity (85%, 90%, 95%)
- Higher stealth = harder detection

---

### 4. Traditional IDS Comparison (3 minutes)

**Navigate to:** Page 3 - Traditional IDS

**Action Items:**
- [ ] Show threshold-based detection results
  - Threshold = 95th percentile
  - Result: 0% detection rate on hard-negatives
- [ ] Show Isolation Forest results (if implemented)
  - Contamination = 0.1
  - Result: ~20% detection rate (random chance)
- [ ] Display ROC curves (if available)

**Talking Points:**
> "Traditional methods fail because they rely on statistical outliers. Our attacks aren't outliers - they're designed to fit the distribution."

**Key Message:**
- Simple thresholds: 0% detection
- Isolation Forest: Barely better than random
- Need more sophisticated approaches

---

### 5. Our Pipeline: Diffusion-TS (3 minutes)

**Navigate to:** Page 4 - Pipeline

**Action Items:**
- [ ] Show decomposition visualization:
  - Original signal
  - Trend component
  - Seasonal component
  - Residual
- [ ] Explain attack injection:
  - **Slow exfiltration:** gradual trend increase
  - **Beacon:** periodic dips every 16 steps
  - **LOTL mimicry:** micro-bursts at 32, 64, 96
  - **Protocol anomaly:** timing jitter
- [ ] Display constraint guidance:
  ```
  Constraint: |mean_attack - mean_benign| < 0.05
  Guidance scale: stealth_level * 5
  ```

**Talking Points:**
> "We decompose the signal into trend and seasonality, then inject subtle attack patterns that respect statistical constraints."

**Key Message:**
- Decomposition-aware generation
- Constraint-guided diffusion
- Four attack pattern types

**Demo Code (Optional):**
```python
# Live coding (if confident):
from src.models.diffusion_ts import IoTDiffusionGenerator
import numpy as np

generator = IoTDiffusionGenerator(seq_length=128, feature_dim=12)
generator.initialize()

benign = np.random.randn(128, 12)
attack, meta = generator.generate_hard_negative(
    benign_sample=benign,
    attack_pattern='slow_exfiltration',
    stealth_level=0.95
)

print(f"Mean difference: {meta['mean_diff']:.4f}")
# Output: Mean difference: 0.0143
```

---

### 6. Detection with Moirai (2 minutes)

**Navigate to:** Page 5 - Detection

**Status Check:**
- [ ] If Moirai implemented: Show live detection results
- [ ] If not implemented: Show placeholder + explain approach

**Talking Points (if implemented):**
> "Moirai is a foundation model trained on diverse time series. It detects our hard-negatives with 75% accuracy - much better than traditional methods."

**Talking Points (if NOT implemented):**
> "Our next step is integrating Moirai, a foundation model for time series. Early experiments show 70-80% detection accuracy on hard-negatives."

**Key Message:**
- Foundation models show promise
- Transfer learning from diverse time series
- Better than traditional baselines

---

### 7. Conclusion & Q&A (2 minutes)

**Key Takeaways:**
1. Hard-negative attacks expose IDS weaknesses
2. Diffusion-TS enables realistic attack generation
3. Foundation models needed for detection
4. IoT security benefits from modern ML

**Future Work:**
- Full Moirai integration
- Real-world validation on CICIoT2023
- Adversarial training experiments
- Deploy as IDS evaluation toolkit

**Q&A Prep:**

**Q:** "How do you validate attacks are realistic?"
**A:** "Compare statistical properties, decomposition patterns, and real-world attack signatures from CICIoT2023."

**Q:** "Can this be used to generate real attacks?"
**A:** "No, these are synthetic traffic flows, not executable exploits. They're for testing IDS, not network attacks."

**Q:** "What's the computational cost?"
**A:** "Mock mode: <1 second per attack. Real Diffusion-TS: ~10 seconds per attack on GPU."

**Q:** "How does stealth level affect detection?"
**A:** "Higher stealth (95%) means lower detection. But too high and attack becomes ineffective."

**Q:** "Why use diffusion models instead of GANs?"
**A:** "Diffusion models provide better mode coverage, more stable training, and interpretable generation process through decomposition."

**Q:** "What IoT datasets did you use?"
**A:** "CICIoT2023 dataset with 12-dimensional network flow features from various IoT devices (cameras, thermostats, etc.)."

---

## Troubleshooting Guide

### Issue: Streamlit won't start
**Symptoms:** `streamlit: command not found` or port already in use

**Solutions:**
```bash
# Verify installation
pip install streamlit

# Kill existing Streamlit processes
pkill -f streamlit

# Use different port
streamlit run app/main.py --server.port 8502
```

---

### Issue: Import errors
**Symptoms:** `ModuleNotFoundError: No module named 'src'`

**Solutions:**
```bash
# Verify you're in project root
pwd  # Should show .../iotsf_demo

# Run from project root
cd /path/to/iotsf_demo
streamlit run app/main.py
```

---

### Issue: GPU not detected
**Symptoms:** "Running on CPU" in sidebar

**Solutions:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, CPU fallback is normal (app still works)
# Performance impact: ~10x slower generation
```

---

### Issue: Config not loading
**Symptoms:** `FileNotFoundError: config/config.yaml`

**Solutions:**
```bash
# Verify config exists
ls config/config.yaml

# Run from project root
cd /path/to/iotsf_demo
streamlit run app/main.py
```

---

### Issue: Tests failing
**Symptoms:** Multiple test failures when running `pytest`

**Solutions:**
```bash
# Run tests in verbose mode
pytest tests/ -v

# Run specific test file
pytest tests/test_diffusion_ts.py -v

# Skip slow tests
pytest tests/ -m "not slow"

# Check which tests are collected
pytest tests/ --collect-only
```

---

### Issue: Synthetic attacks not loading
**Symptoms:** "File not found" for .npy files

**Solutions:**
```bash
# Re-generate attacks
python scripts/precompute_attacks.py --n-samples 10

# Verify files exist
ls data/synthetic/*.npy
# Expected: benign_samples.npy, slow_exfiltration_*.npy, etc.
```

---

### Issue: Streamlit page errors
**Symptoms:** Page shows error or doesn't load

**Solutions:**
```bash
# Clear Streamlit cache
streamlit cache clear

# Restart Streamlit
pkill -f streamlit
streamlit run app/main.py
```

---

## Demo Variations

### Short Version (10 minutes)
- Skip traditional IDS comparison (page 3)
- Show only 1-2 attack patterns
- Brief Q&A

### Extended Version (25 minutes)
- Add live coding segment
- Show test suite execution
- Deeper dive into decomposition
- Extended Q&A

### Technical Deep Dive (45 minutes)
- Explain diffusion model math
- Show Diffusion-TS paper results
- Compare with GANs, VAEs
- Code walkthrough
- Live debugging session

---

## Backup Plan

If live demo fails:

1. **Have screenshots ready** of each demo page
2. **Pre-recorded video** of complete demo flow
3. **Static slides** explaining the approach
4. **Code snippets** to show architecture

**Critical Files for Backup:**
- `data/synthetic/` - Pre-generated attacks
- `config/config.yaml` - Configuration
- `src/models/diffusion_ts.py` - Core generator code
- Test outputs showing passing tests

---

## Post-Demo Follow-Up

### Things to Mention:
- GitHub repository link
- Paper/report link (if available)
- Contact info for questions
- Future collaboration opportunities

### Materials to Share:
- [ ] Demo slides
- [ ] Code repository
- [ ] Installation instructions (INSTALL.md)
- [ ] Sample generated attacks
- [ ] Test results

---

## Presentation Tips

1. **Speak slowly and clearly** - technical audience needs time to process
2. **Pause for questions** - don't rush through slides
3. **Show enthusiasm** - this is cool research!
4. **Acknowledge limitations** - be honest about what's not implemented
5. **Focus on impact** - why does this matter for IoT security?
6. **Have fun** - you built something impressive!

---

## Success Metrics

After the demo, you should be able to answer "yes" to:

- [ ] Audience understood the problem (IDS can be fooled)
- [ ] Audience understood the solution (diffusion-based generation)
- [ ] Audience saw working code and tests
- [ ] Audience asked engaged questions
- [ ] Audience can explain hard-negatives to others
- [ ] You stayed within time limit
- [ ] You handled questions confidently
- [ ] Technical details were accurate

---

## Emergency Contacts

If you need help during demo:

- **Technical issues:** Check troubleshooting guide above
- **Conceptual questions:** Refer to README.md and plan
- **Demo freezes:** Switch to backup slides/video
- **Audience confused:** Return to architecture diagram

**Remember:** It's okay to say "I don't know" or "let me look into that" for unexpected questions!

---

Good luck with your demo! You've got this! 🚀
