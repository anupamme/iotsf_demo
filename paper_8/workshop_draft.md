# Drift ≠ Damage: A Value-Gated Diagnostic for Representation Change in Time-Series Foundation Model Fine-Tuning

*Workshop draft (≈4 pages). Prose expansion of `workshop_version_sketch.md`. Figures:
`fig1_diagnostic_flow`, `fig2_dissociation_sweep`. Provenance: Moirai-Base, Chronos/M4, MOMENT,
and all h=96 gate-fail numbers are from the main paper's tables (`07_analysis.tex`,
`tables/sample_sweep.tex`); the h=24 Chronos/ETT and TimesFM/ETT rows are from this work's
short-horizon runs (GPU instance now offline — sourced from this session's run logs and the
official-comment record). Fig. 2 matches `tables/sample_sweep.tex` exactly.*

---

## Abstract

Fine-tuning a time-series foundation model (TSFM) can drive large representational change, and
the field often treats such drift as a warning sign of catastrophic forgetting to be suppressed
with preservation methods. We show this reflex is misleading. Across four TSFMs — Moirai,
Chronos, TimesFM, and MOMENT — the *magnitude* of encoder drift, measured by CKA, does not
predict downstream harm: an encoder whose representations move by ~70% can forecast markedly
*better*, while a barely-moved encoder can be no better than its zero-shot self. We introduce a
lightweight diagnostic chain — a downstream-only **value gate** (is the pretrained checkpoint
worth preserving?), a **probe-sign asymmetry** test (does drift sharpen the trained objective
while sparing task-orthogonal structure?), and a **frozen-encoder causal control** — that sorts
a fine-tuning cell into one of four regimes: screened-out, value-driven stability,
capacity-driven stability, and beneficial specialization. Only Moirai's mixture-of-experts
encoder exhibits the beneficial-specialization signature; where non-Moirai backbones pass the
(horizon-sensitive) gate at all, the same recipe restructures Chronos without task-specificity and
leaves high-capacity TimesFM essentially unchanged. We argue drift and utility are separate axes,
give practitioners a cheap pre-fine-tuning screen, and pose architecture-specific specialization
as an open question.

---

## 1. Introduction

A Moirai encoder fine-tuned on ETTh2 can have its internal representations change so much that
centered kernel alignment (CKA) with the pretrained encoder falls to ~0.3 — a ~70% shift — and
yet forecast *better* than the zero-shot model on the very same task. If drift were a proxy for
damage, this should not happen. It happens routinely.

Practitioners increasingly fine-tune *released* TSFM checkpoints on their own data, and the
default toolkit for reasoning about what fine-tuning does to a pretrained model is inherited
from continual learning and NLP: measure how far representations moved (CKA, weight norms), read
large movement as catastrophic forgetting, and reach for preservation methods (LoRA, EWC,
L2-SP) to hold the model in place. This reflex bundles three claims that are usually left
implicit — that drift implies forgetting, that forgetting implies harm, and therefore that
drift should be minimized. We show the bundle comes apart on time-series foundation models.

Our central position is that **drift and utility are separate axes**. The useful question is not
*how much* an encoder drifts but *what kind* of drift it is: is the movement functional
specialization toward the task, incidental restructuring that leaves task performance untouched,
or genuine degradation? A single scalar like CKA cannot tell these apart, because all three can
produce the same CKA.

We make three contributions.

1. **Drift ≠ damage.** Empirically, across four architecturally distinct TSFMs, CKA drift does
   not predict task harm. We exhibit the dissociation both across backbones and, most sharply,
   *within* a single model as training data grows (Fig. 2).
2. **A cheap, falsifiable diagnostic chain.** A downstream-only value gate, a probe-sign
   asymmetry test, and a frozen-encoder causal control (Fig. 1) together classify a fine-tuning
   cell — using only the released checkpoint and the target data, no pretraining corpus.
3. **A four-regime taxonomy** of fine-tuning outcomes governed by pretrained value and model
   capacity. Notably, the *beneficial-specialization* regime is, in our study, specific to
   Moirai's mixture-of-experts encoder; we flag *why* this signature is architecture-specific as
   the key open question rather than claiming a universal law.

We are explicit about scope. The drift–utility dissociation as a *phenomenon with a mechanistic
signature* (probe-sign asymmetry) is established on Moirai — a case study. What generalizes, and
what we demonstrate across four backbones, is the *diagnostic protocol*: on every model it
returns the correct, interpretable verdict, and a diagnostic that behaved identically on every
model would carry no information at all.

## 2. Setup and the Diagnostic Chain

**Fine-tuning setup.** We study four TSFMs spanning distinct architectures: **Moirai**
(encoder-only, mixture-of-experts), **Chronos-T5** (encoder–decoder), **TimesFM** (decoder-only,
`d_model`=1280), and **MOMENT** (encoder-only). We fine-tune on univariate series from the ETT
family and M4, with a lookback of 96 and short forecast horizons (h=24 for the cross-backbone
screen; the Moirai sample-size sweep in §3 uses h=96). Fine-tuning attaches a lightweight MSE
regression head to the encoder output. Two conditions run throughout: **B**, full fine-tuning of
the encoder and head; and **D**, a frozen-encoder control in which the encoder is held fixed and
only the head is trained. Comparing B and D isolates what encoder weight updates actually
contribute.

The diagnostic chain (Fig. 1) is one screen followed by two tests, plus a causal control.

**Value gate (the screen).** Before asking whether drift is harmful, we ask whether the
checkpoint has anything worth preserving. The gate is a downstream-only quantity,
R²_task = 1 − MSE_ZS / MSE_Linear, comparing the released model's zero-shot error on the target
set to a lookback-96 linear baseline fit on that same set. A cell is *gate-pass* when the
foundation model beats the linear baseline by more than 20%. Crucially this needs **no
pretraining data or compute** — only the released weights and the practitioner's own series. The
20% cutoff is a practical threshold, not a phase boundary: re-deriving the regime map at 10/20/30%
leaves the gate-pass set essentially unchanged, and we report R²_task continuously.

**CKA drift (test 1).** For gate-passing cells we measure encoder drift as linear CKA between
pretrained and fine-tuned representations on held-out windows. Drift is flagged at CKA < 0.95.
CKA alone is deliberately *not* the verdict — it tells us movement occurred, not whether the
movement helped.

**Probe-sign asymmetry (test 2).** This is the test that separates functional specialization
from incidental restructuring. We fit a Ridge probe from frozen representations to the trained
objective (its improvement is ΔR²_trained) and, separately, to a battery of *task-orthogonal*
targets — lag-1 autocorrelation, window mean, window variance (ΔR²_⊥). The
**beneficial-specialization signature** is a sign asymmetry: ΔR²_trained > 0 **and** ΔR²_⊥ ≤ 0 —
the encoder sharpened exactly the structure the task needs while giving up structure it does not.
When instead *all* probes improve together, the encoder has restructured generally, not
specialized. The diagnostic is the sign pattern, never an absolute magnitude — on some Moirai
cells the absolute probe R² is negative, so only the *relative* change is read.

**Frozen-encoder control (causality).** Condition D verifies that any effect is caused by
encoder weight updates rather than head adaptation or overfitting: freezing the encoder yields,
by construction, CKA = 1.000 and ΔR² = 0. Where D matches B on task performance, the encoder's
drift was non-functional; where freezing eliminates the effect, the drift was causal.

*(Fig. 1 here — the diagnostic-chain decision flow.)*

## 3. Results: A Horizon-Sensitive Gate and Four Regimes

Table 1 is the cross-backbone spine, and one structural fact organizes it: **the value gate is
horizon-sensitive.** The same non-Moirai backbone can gate-*fail* at a long horizon and
gate-*pass* at a short one, so the forecast horizon determines which cells even enter the
diagnostic chain. Where a cell enters, the chain assigns a regime that CKA alone would conflate.
(Rows marked h=24 extend the paper's matched h=96 screen; the h=24 non-Moirai runs are from this
work.)

**Table 1. The gate decides entry (horizon-sensitive); the chain decides the regime.**

| Backbone × data | Arch. | h | Value gate | CKA | Asym. | Regime |
|---|---|---|---|---|---|---|
| Moirai-Base / ETTh2 | Enc-MoE | 96 | 31% | 0.25–0.33 | **Yes (9/10)** | **Beneficial specialization** |
| Chronos / M4-Monthly | Enc–Dec | — | 84.5% | 0.98–0.999 | No | Value-driven stability |
| Chronos / ETT | Enc–Dec | 96 | fail (−43%) | — | — | Screened out |
| Chronos / ETT | Enc–Dec | 24 | 42–52% | 0.886–0.949 | No | Incidental restructuring |
| TimesFM / ETT | Decoder | 96 | fail (≈−14%) | — | — | Screened out |
| TimesFM / ETT | Decoder | 24 | 46–53% | 0.995–0.999 | No | Capacity-driven stability |
| MOMENT / ETT | Enc | 24/96 | fail (−14 to −148%) | — | — | Screened out |

**Beneficial specialization (Moirai-Base / ETTh2, h=96).** Moirai-Base gate-passes (31%), drifts
aggressively (CKA 0.25–0.33), and yet *improves* on the target task — forgetting −56.9% at n=500
and −54.2% at n=10k (10/10 seeds negative). The probe battery shows the specialization signature
in 9/10 seeds (ΔR² = +0.89): the trained-objective probe improves while the task-orthogonal probes
do not. The frozen-encoder control gives zero improvement, confirming the encoder update — not the
head — *causes* the gain. This is the drift–utility dissociation with a mechanistic fingerprint.

**Value-driven stability (Chronos / M4-Monthly).** With a strong gate (84.5%, M4 is in Chronos's
pretraining corpus), the encoder is essentially unchanged (CKA 0.98–0.999) and there is no
distinguishable forgetting (+0.5% at n=10k, 10 seeds): the pretrained features are already good
enough that fine-tuning gains little by moving them.

**Horizon flips gate entry — and reveals two more regimes.** At the paper's matched horizon
(h=96), Chronos, TimesFM, and MOMENT gate-*fail* on every ETT cell (−14% to −148% vs. a linear
fit) and are correctly *screened out* — no transferable value to preserve. Shortening the horizon
to h=24 flips two of them into gate-pass, and the chain then classifies them:

- *Incidental restructuring (Chronos / ETT, h=24, gate 42–52%).* The encoder drifts substantially
  (CKA 0.886–0.949) and the task improves, but the probe battery shows **no** asymmetry — the
  task-orthogonal probes rise alongside the trained one, i.e. general restructuring, not
  specialization. The frozen control is decisive: on ETTh1, condition D nearly matches B (−14.8%
  vs −14.5%), so the drift is largely *non-functional*.
- *Capacity-driven stability (TimesFM / ETT, h=24, gate 46–53%).* At the same moderate gate where
  Chronos drifts, TimesFM barely moves (CKA 0.995–0.999, 10 seeds; ΔR² ≈ 0). Its larger hidden
  dimension (`d_model`=1280 vs 512) absorbs the fine-tuning pressure through the head without
  restructuring the encoder; here the probe floor is even *positive* (Ridge R² ≈ 0.72–0.79).

**Screened out (MOMENT).** MOMENT gate-fails on every ETT cell at both horizons (within the −14%
to −148% band): zero-shot is worse than a linear fit, so there are no valuable features to
preserve and any preservation method is correctly skipped before it starts.

**The within-model dissociation (Fig. 2).** The cross-backbone table shows drift outcomes differ
*between* models; the sharpest evidence that drift does not determine outcome comes from *within*
one model. Fig. 2 sweeps Moirai-Small on ETTh2 across training-set size n. Encoder drift is a
clean, near-monotonic function of data (CKA 0.95 → 0.52 as n grows from 500 to 10k), with tight
error bars. Task forgetting, over the *same* sweep, is neither monotonic nor tightly determined:
it runs +14%, +13%, −3%, +7%, −5%, flipping sign repeatedly, and at the largest n — maximum drift
— the model *improves*. A clean monotonic driver producing a sign-flipping, high-variance outcome
is the dissociation in one picture: you cannot read the task result off the amount of drift.

*(Fig. 2 here — the within-model dissociation sweep.)*

**What the chain buys.** The value gate governs *entry* — and is horizon-sensitive, so a cell can
move between screened-out and gate-pass with the forecast horizon. Among gate-passing cells, the
probe-sign test then distinguishes *functional* specialization (Moirai) from *incidental*
restructuring (Chronos/ETT) — a distinction invisible to CKA, since both cross the 0.95 line —
while capacity and pretrained value separate the two stability regimes. The protocol thus returns
four distinct, correct verdicts across three architecturally different backbones, and flags the
horizon dependence of the screen itself.

## 4. Discussion and Open Questions

**A practitioner recipe.** The chain is a decision procedure. Run the value gate first — it is
free and uses only your own data. If the cell gate-fails, skip preservation entirely: there are
no valuable features to lose, and applying LoRA/EWC to worthless representations is wasted effort.
If it gate-passes, use the probe-sign test to decide whether the drift you observe is functional
specialization worth protecting or incidental restructuring you can ignore. CKA by itself should
not trigger a mitigation.

**The open question.** The beneficial-specialization signature — drift that is functional and
task-aligned — appears, in our study, only for Moirai's mixture-of-experts encoder. The same
fine-tuning recipe restructures Chronos's dense encoder without task-specificity and leaves
high-capacity TimesFM essentially still. We do not claim the phenomenon is universal; the title's
"case study" is deliberate. Why an MoE encoder specializes where dense encoders restructure or
stay put — whether experts provide a natural substrate for task-specific reorganization — is the
question we most want the community to take up.

**Limitations.** Our study is univariate and focused on the ETT/M4 families; multivariate and
other domains remain open. The value gate is *horizon-sensitive*: non-Moirai backbones that
gate-fail at h=96 gate-pass at h=24, so the gate reflects transferable value at a given horizon
rather than an intrinsic property of the checkpoint — useful for a practitioner (screen at your
own horizon) but a caveat for cross-study comparison, and the reason the non-Moirai regimes in
Table 1 are horizon-scoped. On some Moirai cells the absolute probe R² is negative, so the probe
signal is only interpretable as a relative sign change (we report the positive-floor task-native
metric alongside it). ETTh1 is a genuine probe-mechanism boundary rather than evidence for the
asymmetry, and we scope the claim to exclude it. Finally, the *sign* of forgetting is
protocol-dependent at large n (early-stopped vs final-epoch checkpoints can disagree, as Fig. 2's
variance hints); we report early-stopped, the standard choice, and note that the dissociation
signal itself is robust to this while one secondary metric is not.

---

## References

*(Trim the main-paper bibliography to ~12–15 core entries: TSFM backbones — Moirai, Chronos,
TimesFM, MOMENT; CKA; the continual-learning / catastrophic-forgetting and LoRA/EWC/L2-SP
preservation literature the diagnostic refines; linear-probing methodology.)*
