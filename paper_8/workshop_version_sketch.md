# Workshop-Paper Sketch (4 pages) — Time-Series Foundation Model Fine-Tuning Diagnostics

**Primary target:** NeurIPS 2026 Workshop "Foundation Models for Temporal Systems: From
Forecasting to World Modeling" (Sydney).
**Secondary:** "Continual Learning in the Era of Foundation Models and Embodied Agents" —
same content, re-angled (see note at end).

**Carve-out strategy.** The main paper over-reached on generality and got dinged for it. The
workshop version flips that: instead of claiming a universal protocol, it presents an
**honest four-backbone taxonomy** of what fine-tuning does to a time-series foundation model's
representations, with the drift–utility dissociation as the provocative centerpiece and the
architecture-specificity as an explicit open question. Sharp, self-contained, discussion-inviting
— exactly what a workshop wants. Non-archival, so it conflicts with neither the NeurIPS main-track
decision nor an ICLR 2027 submission.

---

## Title

**Primary:** "Drift ≠ Damage: A Value-Gated Diagnostic for Representation Change in
Time-Series Foundation Model Fine-Tuning"

**Alt:** "When Does Fine-Tuning Help a Time-Series Foundation Model? Four Backbones, Four Regimes"

---

## Abstract (draft, ~150 words)

Fine-tuning a time-series foundation model (TSFM) can drive large representational change, and
the field often treats such drift as a warning sign of catastrophic forgetting. We show this
is misleading: across four TSFMs (Moirai, Chronos, TimesFM, MOMENT) the *magnitude* of drift
(measured by CKA) does not predict downstream harm. We introduce a lightweight diagnostic
chain — a downstream-only **value gate** (is the pretrained checkpoint worth preserving?), a
**probe-sign asymmetry** test (does drift sharpen the trained objective while sparing
task-orthogonal structure?), and a **frozen-encoder causal control** — that separates four
regimes: value-driven stability, capacity-driven stability, incidental restructuring, and
beneficial specialization. Only Moirai's mixture-of-experts encoder exhibits the
beneficial-specialization signature; the same fine-tuning restructures Chronos generally and
leaves high-capacity TimesFM essentially unchanged. We argue drift and utility are separate
axes, give practitioners a pre-fine-tuning screen, and pose architecture-specific
specialization as an open question.

---

## Contribution bullets (Intro)

1. **Drift ≠ damage.** Empirically, CKA drift does not predict task harm on TSFMs; we
   dissociate the two axes across four backbones.
2. **A diagnostic chain that is cheap and falsifiable:** value gate (ZS-vs-linear, no
   pretraining data), probe-sign asymmetry, frozen-encoder control.
3. **A four-regime taxonomy** of fine-tuning outcomes governed by pretrained value and model
   capacity — with the beneficial-specialization regime found to be architecture-specific (MoE
   encoder), which we flag as an open question rather than a universal law.

---

## Section-by-section outline (4 pages + refs)

**1. Introduction — ~0.75 pp.**
Hook: a Moirai encoder whose representations move by ~70% (CKA ≈ 0.3) *improves* on the target
task. Motivation: practitioners increasingly fine-tune released TSFM checkpoints and reach for
drift metrics / preservation methods (LoRA, EWC) reflexively. Claim: the right question is not
*how much* drift but *what kind*. Contribution bullets above. Keep positioning short — this is
a workshop.

**2. Setup & Diagnostic Chain — ~1 pp.**
- TSFM fine-tuning setup: univariate ETT / M4, lookback 96, horizon 24; MSE head on encoder
  output; conditions B (full fine-tune) and D (frozen-encoder control).
- **Value gate:** R²_task = 1 − MSE_ZS / MSE_Linear on the target set only; gate-pass at >20%
  (report continuously; note 10/20/30% robustness in one line).
- **CKA** for representational drift (pre vs post, held-out windows).
- **Probe-sign asymmetry:** Ridge probe on the trained objective (ΔR²) vs task-orthogonal
  probes (lag-1 autocorr, mean, variance). Signature = ΔR²_trained > 0 **and** ΔR²_ortho ≤ 0.
- **Frozen-encoder control (D):** isolates encoder weight updates as cause (CKA = 1.000,
  ΔR² = 0 by construction).
**Fig. 1 (method-at-a-glance).** A single top-down decision flow: value gate → CKA drift →
probe-sign asymmetry, with the frozen-encoder control verifying causality at every gate-pass
branch, and the four terminal regimes color-matched to Table 1 and Fig. 2. Render script
`fig1_diagnostic_flow.py` (`.pdf`/`.png` beside it).

> **Fig. 1. The diagnostic chain.** A downstream-only value gate plus two representation tests
> classify a fine-tuning cell. The gate (zero-shot vs. a lookback-96 linear baseline) screens
> out checkpoints with no transferable value (e.g., MOMENT). For gate-passing cells, CKA
> measures encoder drift; drift (CKA < 0.95) triggers the probe-sign asymmetry test — a positive
> trained-objective ΔR² with non-positive task-orthogonal ΔR². The four terminal regimes
> (screened out, stability, beneficial specialization, incidental restructuring) are
> color-matched to Table 1 and Fig. 2; a frozen-encoder control (condition D) confirms
> causality at every gate-pass branch.

**3. Results: Four Backbones, Four Regimes — ~1.5 pp.**
Central table (Table 1) — the spine of the paper:

| Backbone × data | Arch. | h | Gate | CKA | Asym. | Regime |
|---|---|---|---|---|---|---|
| Moirai-Base / ETTh2 | Enc-MoE | 96 | 31% | 0.25–0.33 | Yes (9/10) | Beneficial specialization |
| Chronos / M4-Monthly | Enc–Dec | — | 84.5% | 0.98–0.999 | No | Value-driven stability |
| Chronos / ETT | Enc–Dec | 96 | fail (−43%) | — | — | Screened out |
| Chronos / ETT | Enc–Dec | 24 | 42–52% | 0.886–0.949 | No | Incidental restructuring |
| TimesFM / ETT | Decoder | 96 | fail (≈−14%) | — | — | Screened out |
| TimesFM / ETT | Decoder | 24 | 46–53% | 0.995–0.999 | No | Capacity-driven stability |
| MOMENT / ETT | Enc | 24/96 | fail (−14 to −148%) | — | — | Screened out |

*(Provenance-corrected: the value gate is horizon-sensitive — non-Moirai backbones gate-fail at
h=96 (paper) but gate-pass at h=24 (this work). Paper-vetted numbers: Moirai-Base 31%, Chronos/M4
84.5%, MOMENT −14 to −148%. See `workshop_draft.md` §3 for the authoritative prose.)*

Prose, one short paragraph per regime:
- **Beneficial specialization (Moirai):** aggressive drift, forgetting −54% to −57%, probe
  asymmetry 9/10; frozen control confirms the encoder update *causes* the gain.
- **Incidental restructuring (Chronos/ETT):** drift is real (CKA→0.886) but orthogonal probes
  rise too — no asymmetry; frozen D ≈ B on ETTh1 (−14.8% vs −14.5%) ⇒ drift is non-functional.
- **Capacity-driven stability (TimesFM):** at the *same* moderate gate where Chronos drifts,
  d_model = 1280 absorbs the pressure via the head (CKA ≈ 1.0); probe floor is *positive*
  (Ridge R² ≈ 0.72–0.79), so the diagnostic reads cleanly.
- **Value-driven stability (Chronos/M4):** strong gate, encoder essentially unchanged.
- **Screened out (MOMENT):** gate-fail (−37% to −318%) ⇒ no valuable features to preserve.

**Fig. 2 (centerpiece) — the within-model dissociation.** Table 1 is the *cross-backbone*
result; Fig. 2 is its *within-model* complement, and the paper's strongest single visual. A
two-panel sweep on Moirai-Small / ETTh2 across training-set size n (distinct from the
Moirai-Base row above): (top) encoder drift CKA falls **monotonically** 0.95 → 0.52; (bottom)
task forgetting **flips sign four times** (+14, −3, +7, −5 %) and stays high-variance — at
*maximum* drift the model improves. A clean monotonic driver with a sign-flipping outcome is
the dissociation in one glance. Values are exactly Table `tab:sample_sweep`; render script
`fig2_dissociation_sweep.py` (`.pdf`/`.png` beside it).

> **Fig. 2. Drift and task utility are dissociable within a single model.** Moirai-Small
> fine-tuned on ETTh2 (h=96, condition B) across training-set sizes n. *Top:* encoder drift
> (CKA vs. the pre-trained encoder) decreases monotonically as n grows — more data, more drift.
> *Bottom:* task forgetting is non-monotonic, flipping sign four times and remaining
> high-variance; at the largest n (maximum drift) forgetting is negative — the model *improves*.
> Because a monotonic drift curve coexists with a sign-flipping outcome curve, drift magnitude
> alone does not determine whether fine-tuning helps or harms. Error bars ±1 SD across seeds;
> n=10k is the 10-seed early-stopped, CUDA-deterministic run.

**4. Discussion & Open Questions — ~0.5 pp.**
- Practitioner takeaway: run the value gate first (free); if gate-fail, skip preservation
  entirely; if gate-pass, use the probe-sign test to decide whether drift is worth protecting.
- Honest scope: the beneficial-specialization signature is, so far, specific to Moirai's MoE
  encoder. We do not claim it generalizes; we frame *why MoE encoders may specialize where
  dense encoders restructure or stay put* as the key open question.
- Limitations: univariate ETT/M4 focus; probe negative-floor regime on some Moirai cells;
  ETThh1 as a probe-mechanism boundary; forgetting-sign protocol sensitivity (report
  early-stopped).

**References — ~0.5 pp** (trim main-paper bib to ~12–15 core entries).

---

## Figures / tables budget
- **Table 1** — four-backbone regime taxonomy (must-have; the cross-backbone spine).
- **Fig. 1** — diagnostic-chain decision flow (method-at-a-glance).
- **Fig. 2** — within-model dissociation sweep: monotonic CKA vs. sign-flipping forgetting on
  Moirai-Small/ETTh2 (the "drift ≠ damage" money shot; `fig2_dissociation_sweep.py`, values
  from `tables/sample_sweep.tex`).
Two figures + one table is right for 4 pages.
(Note: an earlier CKA-vs-utility *scatter* idea was dropped — with only four gate-passing cells
it reads as a clean monotonic trend and appears to argue the opposite of the claim. The
within-model sweep shows the dissociation without that failure mode; `fig2_drift_utility.py`
is retained only as a rejected alternative.)

## Carry over vs. cut (from the main paper)
**Keep:** value gate, CKA, probe-sign asymmetry, frozen control, the four-backbone table, the
Moirai dissociation, the honest architecture-specificity framing.
**Cut for space:** per-layer CKA / gradient-norm analysis (W4), the full 153-cell CKA-threshold
sweep (compress to one sentence), LoRA/EWC/L2-SP mitigation ladder (mention in one line),
extended ablations, ILI and multivariate detours.

## Re-angle for the Continual-Learning workshop (if targeting secondary)
Same results; change Intro + framing: lead with "not all forgetting is harmful — a dissociation
between representational forgetting and task forgetting," position the value gate as a
*when-to-preserve* trigger for continual learning, and cite the CL/EWC/LoRA literature as the
default reflex the diagnostic refines. Table 1 and figures unchanged.
