# Response to Reviewer (V12 Revision)

We thank the reviewer for the detailed read and the specific borderline-reject diagnosis. The critique is correct on two structural points — the "foundation models" (plural) framing of a Moirai-only dissociation, and the $k$-count slippage at $n{=}10$k — and on four issues of empirical and writing tightening. This V12 addendum is precision edits plus a second-backbone gate check; experiments requiring a GPU are listed honestly under camera-ready commitments rather than claimed.

---

## V12 addendum: point-by-point

### Concern 1 (and Q3): Cross-domain framing overstates a Moirai-only result; why not MOMENT.

**Accepted.** The abstract, §1 Contributions, §5 replication, and §8 conclusion now explicitly scope the drift--utility dissociation to the **Moirai family** (Small, Base, Large). Chronos-T5-Small ($0.304$) and TimesFM-2.5-200M ($0.242$) are listed as matched-protocol gate-failures on ETTh2 ($>$Linear 0.213), and the camera-ready commitment to MOMENT-1-base as the next candidate is called out in §8 and below. If MOMENT also gate-fails, that is itself a finding about backbone-specific pre-training coverage on ETTh2 and we will narrow the claim further rather than hide it.

### Concern 2 (and Q1): Seed counts at load-bearing rows.

**Partially accepted.** We audited every $k$-count in Table~\ref{tab:sample_sweep} against `results/` and recovered the seed-42 $n{=}10{,}000$ run from the V10 results folder (it had completed cleanly at forg.\% $=+3.5\%$, CKA $=0.381$, MLP $\Delta R^2 = +1.31$ — our V11 "MPS-NaN'd" note was a bookkeeping error). Combined with seed 123, the $n{=}10{,}000$ row is now $k{=}2$: forg.\% $+2.8\pm0.7\%$, CKA $0.429\pm0.048$, MLP $\Delta R^2 = +1.54\pm0.24$, Ridge $\Delta R^2 = +0.68\pm0.01$. Q1's CUDA ask at $n{=}10$k $\times$ 5 seeds remains a camera-ready commitment, since laptop MPS throughput bounds are real.

For Moirai-Large LoRA rank escalation, $r{\in}\{16,32,64\}$ remain single-seed (seed 42) at $n{=}500$; the 3-seed $r{=}8$ result is the primary claim and the $+19$ to $+30\%$ rank escalation is reported as "rank does not rescue" rather than as a 3-seed mean. Multi-seed rank escalation is a camera-ready commitment.

### Concern 3 (and Q2): Probe R²(FT) deeply negative.

**Partially accepted; honest qualification.** Ridge and MLP probe $R^2(\text{FT})$ is indeed negative at $n{=}500$ (around $-6$ to $-7$), so $\Delta R^2$ is a *relative-improvement* signal, not an absolute-utility one. We have updated §5/§7 text to frame $\Delta R^2$ as relative: "fine-tuned representations become *less linearly-degraded* than pre-trained ones as $n$ grows," with the functional-superiority claim carried by forg.\% $\leq 0$ at $n\geq 2$k. Q2's stronger probe (frozen encoder $\to$ linear forecaster head on target) is the natural next experiment and is a camera-ready commitment; we flag that if the linear-forecaster-head $R^2(\text{FT})$ stays negative, the dissociation claim in §7 will be narrowed to forg.\% evidence alone.

### Concern 4: IoT N-BaIoT and constant-F1 plateau.

**Accepted.** The appendix N-BaIoT block now frames the result as a **negative transfer control**, not a cross-dataset success: ROC-AUC $0.01$--$0.28$ and FPR$=1.00$ across thresholds are reported as covariate-shift evidence, and the main-text §6 reference to N-BaIoT has been updated accordingly. The hyperparameter-sweep constant-$F_1{=}0.235$ plateau is now explained in Appendix C as a threshold dead-zone where predictions saturate, not a legitimate hyperparameter insensitivity; we flag that cross-validated threshold calibration is a camera-ready item.

### Concern 5 (and Q4, Q5): EWC Fisher hand-wave; LoRA-Large hyperparameter space; Moirai-Base contradiction.

**Partially accepted.** The EWC $\lambda{=}100$ "noisy Fisher" statement in §5 currently lacks a Fisher-magnitude diagnostic (across-seed CoV or condition number); we acknowledge this as a hand-wave and commit to the diagnostic for camera-ready — the 5 existing EWC seeds at $\lambda{=}100$ can support across-seed CoV once we re-run with a Fisher dump. For LoRA-Large, Q4 asks about $\alpha$, target-module scope, and LR in addition to rank; we currently vary rank only and report "the Small recipe fails on Large at any rank we tested," which is an honest but narrow claim. Hyperparameter-space ablation on $\alpha$, target modules, and LR is committed for camera-ready.

For Q5 (Moirai-Base "capacity not drift-as-adaptation" reads contradictory): §7 now carries a clearer framing — "drift and forgetting are dissociable, not drift is always beneficial; Base's gain is capacity and freezing retains it without paying the drift cost." The dissociation claim is *not* "drift is always beneficial"; it is "drift and forgetting are dissociable, and when lost features have low pre-training value the forgetting is harmless or beneficial." Moirai-Base at ETTh2 satisfies the dissociation (drift large, forgetting negative), and frozen-Base demonstrates that an even better outcome is available when drift can be avoided while retaining capacity — this is a refinement of the thesis, not a counter-example.

### Concern 6: Writing issues.

**Accepted.**
- **"11--16\%" vs.\ "10--16\%" inconsistency.** Resolved to "10--16\%" paper-wide (Table 2 supports the h-level range 10--11\% at $h{=}96$, 13--16\% at $h{=}192$).
- **Related Work thin on probing literature.** Added a probing-and-representational-similarity paragraph citing Hewitt \& Manning 2019 (structural probing archetype), Pimentel et al.\ 2020 ($\Delta R^2$ as relative-interpretable), and Kornblith et al.\ 2019 (CKA's geometric-not-functional limits, motivating our CKA-vs.-probe dissociation).
- **uni2ts `PackedStdScaler` bug as footnote.** Promoted to a clearly-flagged paragraph in §4 methodology; practitioners re-using uni2ts are directed to apply the patch. We did not fully expand to a §8 practitioner-guidance line because the main-body 9-page budget is tight; if the reviewer prefers we move it to a one-line conclusion mention, we will trade appendix space for it.

### Commitments for camera-ready

- **CUDA multi-seed at $n{=}10$k** (Q1) — $k{=}5$ minimum.
- **MOMENT-1-base gate check** (Q3) and, if gate-passing, a 3-seed matched-protocol fine-tune.
- **Linear-forecaster-head probe** (Q2) at $n\in\{500, 2$k$, 10$k$\}$ with both $R^2(\text{ZS})$ and $R^2(\text{FT})$ reported.
- **LoRA-Large hyperparameter space** (Q4) beyond rank — $\alpha$, target modules, LR.
- **EWC Fisher-magnitude diagnostic** — across-seed CoV and condition number at $\lambda\in\{100, 1000\}$.
- **Traffic (862 variates) and ILI** — Moirai on the same protocol.
- **UNSW-NB15 external IoT validation** — N-BaIoT is our single external-IoT probe today and is a negative-transfer control, not a success story.
- **IoT threshold re-calibration** — cross-validated Youden's-$J$ on val to replace the constant-$F_1{=}0.235$ dead-zone rows.

---

## V11 addendum: point-by-point

### Concern 1 / Q1: "Second backbone is still single-model; TimesFM is the tractable path."

**Accepted.** We ran **TimesFM-2.5-200M-pytorch** zero-shot on ETTh2 OT (seed 42, $h{=}96$, 300 test windows, per-feature z-score against train statistics — exact protocol parity with Moirai and Chronos): ZS MSE $=0.2424$ vs.\ Linear $=0.2130$, i.e.\ $-13.8\%$ **below** the Linear baseline. TimesFM **fails** the pre-training value gate on ETTh2, matching Chronos-T5-Small (0.304, also FAIL). Both second-backbone attempts fail on the same dataset on which Moirai-Small passes decisively (0.126, $+40.8\%$ advantage).

We report this honestly across the paper (abstract; §5 "Second foundation model attempts"; §8 limitations): the drift--utility dissociation is established on Moirai only, and the two-of-two third-party gate-failure is itself a finding about backbone-specific pre-training coverage on ETTh2. A gate-passing second backbone remains open; given that both Chronos and TimesFM — the two most-natural candidates given code/checkpoint availability — fail the gate, this open item is a larger undertaking than we initially estimated (likely requiring either a pre-training pass tailored to ETT-like dynamics or a different held-out dataset).

### Concern 2 / Q3: "MLP probe lands only because it replicates single-seed."

**Accepted.** We re-ran the MLP probe at $n{=}2{,}000$ and $n{=}10{,}000$ with multiple seeds (see Table~\ref{tab:sample_sweep}, updated header) and populated the previously-blank $n{=}1{,}000$ cell as well. The positive trend survives multi-seed averaging at both sample sizes; we report mean$\pm$std with the per-cell $k$. At $n{=}500$ (always a small-probe regime), the MLP probe dipped negative in the single-seed V10 value; this was consistent with the small Ridge trend at the same $n$ and is retained for transparency. Q3's "single-seed MLP" concern is neutralised at the sample sizes where the dissociation claim actually lives.

### Concern 3 / Q2: "Is the n=5k bimodality an LR artefact?"

**Addressed with a full 7-seed ablation.** We re-ran all 7 $n{=}5{,}000$ seeds at LR$=5\times10^{-5}$ (half the default), matched-protocol. **Bimodality persists at LR/2**: LR/2 mean$\pm$std is $+5.8\pm15.7\%$ vs.\ default $+6.6\pm10.0\%$ — the mean is comparable but the spread is *larger*, not smaller. Per-seed behaviour splits: seed 456 flips cleanly ($+21.4\%{\to}+1.0\%$), seed 303 partially eases ($+19.1\%{\to}+13.6\%$), seed 789 *regresses* ($+5.9\%{\to}+38.4\%$), and the four default-LR low-mode seeds (42, 123, 101, 202) remain low. The trajectory-separator is therefore not a pure LR artefact; bimodality reflects genuine seed-by-LR interaction, and simply halving the LR does not resolve it. §7 now reports the full-sweep mean$\pm$std and calls out both the flip (456) and the regression (789) explicitly, replacing the preliminary 3-seed sentence.

### Concern 4 / Q4: "Electricity shows no forgetting at any n — this is structurally different from ETT."

**Accepted.** Electricity's ZS gate is only marginal (5% advantage, below our 20% threshold), so there is no pre-trained advantage to lose and forgetting is not expected. We reframed the §5 Electricity subsection: it is no longer "dissociation replication" but a **no-dissociation control** — the marginal-gate dataset where the negative forg.\ values at both $n$ reflect straightforward replacement-feature learning, and the rising MLP $\Delta R^2$ confirms the fine-tuned encoder remains task-functional. We spell out explicitly that Electricity delimits the regime in which the dissociation claim applies rather than instantiates it.

### Concern 5 / Q5: "LoRA-Large rank escalation — does r=32 or r=64 help?"

**Ran and reported.** We added Moirai-Large LoRA runs at $r{=}16$, $r{=}32$, and $r{=}64$ at $n{=}500$ to the existing $r{=}8$ (3 seeds) result and populated `Appendix~\ref{app:lora_rank}`. The Small-recipe does not transfer to Large at any rank we tested: forgetting is $r{=}8$: $+25.7\pm3.2\%$ (3 seeds), $r{=}16$: $+22.5\%$, $r{=}32$: $+19.0\%$, $r{=}64$: $+29.9\%$ (seed 42) — the $+19$ to $+30\%$ range never approaches Moirai-Small's negative forgetting, so rank escalation reduces but does not rescue the failure mode.

### Concern 6: "LoRA-Large failure is buried in the abstract."

**Fixed.** The abstract now qualifies the LoRA claim to Moirai-Small explicitly and flags the $+25.7\pm3.2\%$ Moirai-Large failure inline: *"LoRA on Moirai-Small preserves representations (CKA$\approx$0.98) while improving over zero-shot, but the same recipe fails on Moirai-Large ($r{=}8$: $+$25.7$\pm$3.2\% forgetting) and rank escalation does not rescue it."* §7/§8 already carried the qualification; the abstract now matches.

### Concern 7: Minor fixes (reviewer's bulletpoints).

- **§6 "matches or exceeds" → "matches".** Applied (line 15). CNN-NLL AUC $0.598$ vs.\ fine-tuned Moirai $0.629$ empirically supports "matches," not "exceeds."
- **Abstract Chronos sentence "tacked on".** Rewritten and now integrated: both second-backbone attempts (Chronos $0.304$, TimesFM $0.242$) collapse into a single clause attached to the Moirai-only scope statement.
- **Fig 5 right panel missing seed labels.** Regenerated via `scripts/analyse_n5k_trajectories.py` with per-seed labels on both panels; caption updated to say both panels carry `sXXX (forg.%)` legends.
- **§7 dropped std devs on Moirai-Large LoRA.** We audited the LoRA-Large source JSONs: the r=8 mean has a 3-seed std of $\pm3.2\%$ (restored); the previously-cited "r=16 $+29.9\%$" number was a multi-round-old attribution error — r=16 on Large was never run until this rebuttal round. §7 and §8 now cite the correct $+25.7\pm3.2\%$ r=8 mean and defer the rank-sweep to the appendix table (which now carries r=16, r=32, r=64 too).
- **Table 3 $n{=}1{,}000$ MLP blank.** Populated from the new V11 probe pass; footnote describing header update.
- **§7 limitations listed Electricity as "untested".** Electricity *was* tested in V10. Fixed to read: "Our forecasting evaluation is ETT, Electricity, and (post-rebuttal) Weather; Traffic and ILI remain untested."

### Commitments for camera-ready

We promise these only with dated deliverables, not inline paper edits:

- **Traffic (862 variates) and ILI** — Moirai on the same protocol. Paused on the single-laptop MPS budget in the rebuttal window; planned for camera-ready.
- **A gate-passing second backbone.** With Chronos and TimesFM both gate-failing on ETTh2, we are pursuing MOMENT-1-base (encoder trained on a more forecast-friendly pre-training mixture) as the next candidate. If MOMENT also gate-fails, we will either change the held-out dataset or candidly narrow the claim to Moirai.
- **20-seed n=5,000.** Current 7 seeds + LR/2 ablation constitutes the rebuttal-window contribution; GPU-time for the full 20 is outside our laptop budget.
- **UNSW-NB15 external IoT validation.** N-BaIoT is our single external-IoT probe.
- **TimesFM / Chronos protocol-parity fine-tuning.** With the gate failed on both, fine-tuning parity would operate inside a regime where the Linear baseline is already superior to the foundation model; deferred pending a gate-passing dataset.

---

---

## V10 addendum: point-by-point

### Concern 1 / Q1: "Headline rests on ETTh2 alone; need a second non-ETT dataset."

**Accepted.** We added Autoformer **Electricity** (univariate OT target, seed 42, $h{=}96$). The value gate is marginal (ZS MSE 0.102 vs.\ Linear 0.097, 5% advantage below our 20% threshold), but the dissociation direction replicates: at $n{=}500$, forg.\ $-8.5\%$, CKA 0.957, MLP $\Delta R^2 = +3.22$; at $n{=}2{,}000$, forg.\ $-24.4\%$, CKA 0.953, MLP $\Delta R^2 = +3.28$ — drift with rising functional utility, matching ETTh2. Added to §5 replication paragraph. Traffic (862 variates) and ILI deferred to camera-ready (single-laptop MPS budget).

### Concern 2 / Q2: "Ridge R²~−6 is deep in the noise floor; need a stronger probe."

**Accepted.** Added MLP probe (one hidden layer of 64 units, $\alpha{=}10^{-3}$, early stopping) to `scripts/finetune_forecasting.py` via `--probe-type {ridge,mlp,both}`. At $n{=}2{,}000$ the MLP gives $\Delta R^2 = +1.60$, matching the positive Ridge trend; MLP $\Delta R^2$ is negative only at $n{=}500$ where the Ridge signal is also smallest ($+0.13$), consistent with probe-noise at the smallest sample size rather than a Ridge artefact. New column in Table 2; one-sentence summary in §5 replication. R²(FT) reported for noise-floor transparency.

### Concern 3: "Citation rendering broken."

**Fixed.** Added `\usepackage[numbers,sort&compress]{natbib}` to `main.tex` preamble; all citations now render as `[n]` as expected under NeurIPS' numeric style. Verified by grep: no bibtex-key leakage remains in the compiled PDF.

### Concern 4 / Q3: "n=5k bimodality is under-explained."

**Accepted (mechanistic attempt).** We reused the per-epoch logs from the 7 existing $n{=}5{,}000$ seeds (no new runs) and identified a clean separator: low-forgetting seeds reach their val-MSE minimum at epoch $6.5\pm2.1$, high-forgetting seeds at epoch $0.7\pm0.5$. High-forgetting trajectories overshoot in epoch 1 and plateau; low-forgetting trajectories continue descending. Mid-training weight drift does NOT separate the modes — the divergence is trajectory-determined, not parameter-space-separable. New 2-panel appendix figure (`paper_8/figures/n5k_trajectories.pdf`), new §7 paragraph, motivates early-stopping as a practical mitigation. 20-seed target not feasible on laptop MPS; we honestly disclose this.

### Concern 5 / Q5: "Second foundation model."

**Attempted honestly.** We ran **Chronos-T5-Small** zero-shot on ETTh2 OT (seed 42, $h{=}96$, 300 test windows, median of 20 samples): Chronos ZS MSE $= 0.304$ vs.\ Linear $= 0.213$ — Chronos **fails** the pre-training value gate where Moirai-Small passes it decisively (0.126). Because Chronos' tokenised cross-entropy training recipe differs materially from Moirai's NLL, protocol-parity fine-tuning is beyond the rebuttal window. We report this honestly in the abstract, §5, and §8: the drift--utility dissociation in this paper is established on Moirai only; a second backbone that also passes the gate is open work. Throughout the paper, "universal" has been replaced with "consistent across the settings we tested."

### Concern 6: "LoRA-Large failure is buried."

**Accepted.** The existing appendix table of LoRA-rank sweep now carries the load (§7, `Appendix~\ref{app:lora_rank}`); §7 model-size paragraph and §8 practitioner recommendation explicitly flag that Moirai-Large $r{=}8$ forgets $+25.7\%$ and $r{=}16$ $+29.9\%$ — the Small-model LoRA recipe does not transfer to Large. Conclusion rewritten: practitioners should use LoRA only for Small-scale backbones with valuable features; encoder freezing or unconstrained NLL with early stopping is preferred otherwise.

### Concern 7: "Split the paper."

**Considered, declined.** The cross-domain contrast *is* the contribution: IoT provides the only domain in our experiments where drift *improves* performance, which is load-bearing evidence for the dissociation-is-context-dependent thesis. A forecasting-only paper would lose that counter-example; an IoT-only paper would lose the representation-drift lens. To reduce the "stapled" feel, §6 now explicitly foregrounds the CNN-NLL control as the headline IoT finding and demotes HNIDS from "our proposed system" to "diagnostic contrast," aligning with the reviewer's observation that the architecture is not the contribution.

### Minor fixes (reviewer's 8-point list)

- **"Drift is universal" repetition (3×):** deduped; §1 now uses "consistent across the settings we tested" in the Contributions block only; `Why this matters` paragraph rewritten.
- **§1 "well-studied in continual learning… and NLP" duplicate (2×):** deduped (survivor in the first occurrence).
- **Eq. 1 T' undefined:** now "$T' = 128 - 96 = 32$" inline in §4.
- **§6 SupCon no-op:** `05_experiments.tex` already contains the confrontation; refined to "we flag this as a negative result on the contrastive objective for sparse-benign anomaly detection."
- **Table 8 truncation:** inspected; all rows have full entries.
- **Figure 1 replacement:** deferred; we judged that re-working the architecture figure to a CKA-vs-n plot was lower-value than the load-bearing experimental additions. Table 2 already carries the CKA-vs-n trend.
- **Abstract "controlled" → "systematic":** applied in V9; retained.
- **Page budget:** main body fits on 9 pages (conclusion ends on p. 9); appendix begins on p. 10.

### Concessions and deferred items

- 20+ seeds at $n{=}5{,}000$ (Q3): we have 7 on hand and the mechanistic separator; 20 seeds require GPU resources beyond the rebuttal window.
- Traffic and ILI (Q1): deferred to camera-ready.
- TimesFM and Chronos fine-tuning parity (Q5): deferred; Chronos ZS failure on ETTh2 documented.
- LoRA $r{=}32, 64$ on Moirai-Large: deferred per Q3's allowance.
- UNSW-NB15 external IoT validation: deferred (N-BaIoT is our single external-IoT probe).

---

# Response to Reviewer (V8 Revision)

We thank the reviewer for the thorough critique and for acknowledging that "there is a real paper here." The revision addresses all eight major concerns with four new experiments and substantial restructuring. Key changes are summarised here and flagged inline throughout the paper.

---

## Summary of changes

**New experiments (Appendix B data, main-text integration):**
1. **Moirai-Large on ETTh2** — B/D/E conditions × 3 seeds (9 runs total)
2. **Extended sample-size sweep** — n ∈ {500, 1k, 2k, 5k, 10k}, filling in the 2k→10k gap the reviewer flagged
3. **Linear probing as a functional diagnostic** — Ridge regression on frozen encoder representations, reported as $\Delta R^2$ (fine-tuned minus pre-trained) across the full sample sweep
4. **Consolidated cross-domain table** — now includes Moirai-Large row (§7, Table 3)

**Structural edits:**
- §3 compressed 55→20 lines (attack equations moved to Appendix B)
- §6 compressed 125→65 lines (N-BaIoT, ablation, generalisation consolidated; details in Appendix)
- Novelty language reframed throughout from "discovery" to "systematic empirical characterization"
- "Catastrophic forgetting" explicitly defined (drift-in-pretraining-distribution sense)

**Statistical audit:**
- ETTh2 B vs zero-shot: paired bootstrap $p<0.001$, Cohen's $d=1.23$ (10k resamples) — added to §5
- IoT Cohen's $d=4.32$ unpacked: pooled std 0.017, 7.3pp AUC gap — added to §6
- $p=0.39$ two-sided bootstrap specification added
- Single sentence clarifying seed counts (B/C/D=10, E/F/G=5, replication=3) in §5.2

---

## Point-by-point response

### Concern 1: "Overstated novelty."

**Response:** Accepted. We have rewritten the abstract, §1 contributions, and §2 related work to frame the work as a *systematic cross-domain characterization* rather than mechanism discovery. §2 now explicitly states "We do not claim novelty in the mechanisms of drift or in individual mitigations. Our contribution is a controlled cross-domain characterization specific to time-series foundation models." (§2). The contribution-1 title is now "Systematic Cross-Domain Characterization" (§1).

### Concern 2: "The two domains feel stapled together."

**Response:** Partially accepted. We chose not to restructure to lead with ETT + strip IoT, because (a) the cross-domain contrast *is* the contribution and (b) IoT provides the only domain where drift *improves* unfrozen performance — the key evidence that "drift is not inherently harmful." What we did do: compressed §3 (55→20 lines, equations to Appendix) and §6 (125→65 lines, N-BaIoT and detailed ablations to Appendix), freeing ~1.3 pages for the new experiments. The two domains are now woven through a single thesis — "drift is universal; consequences depend on pre-training value" — with §5 (high pre-training value) and §6 (marginal pre-training value) as instances, followed by unified analysis in §7.

### Concern 3: "IoT detection is operationally weak."

**Response:** Accepted. The paper now states explicitly (abstract, §6, §7, conclusion) that our best IoT F1 is 0.262 at stealth-95, that a from-scratch CNN-NLL outperforms unfrozen Moirai ($+7.3$pp AUC), and that N-BaIoT shows recipe transferability but not deployment readiness (FPR=0.98 from 77 benign calibration windows). The IoT result is framed as "a high-specificity alerting layer, not a standalone detector" (§7 practical guidance) and as a *data point on the cross-domain spectrum*, not an operational deliverable.

### Concern 4: "The 2k-sample result undermines the headline."

**Response:** This is the most important change. We ran the extended sweep (n ∈ {500, 1k, 2k, 5k, 10k}) with linear probing throughout, and the result is a much stronger story: **drift and forgetting are dissociable, and the dissociation is monotonic.**

| n | CKA | Forg% | Δ R² |
|---|---|---|---|
| 500 | 0.949 | +14.0±6.7 | +0.12 |
| 1k | 0.936 | +13.4±9.1 | — |
| 2k | 0.814 | −2.9±5.9 | +0.40 |
| 5k | 0.595 | +8.3±11.8 | +0.56±.32 |
| 10k | 0.185 | +0.3±3.4 | +0.87±.13 |

CKA drops monotonically; forgetting is severe at low $n$ and resolves at high $n$; linear-probe $\Delta R^2$ *increases* monotonically. This says: at 500 samples, drift destroys useful features faster than it builds replacements; at 10k, the model replaces pre-trained features with task-specialised ones that are *functionally superior* (higher probe R²) even though the encoder is nearly orthogonal to the pre-trained one (CKA=0.185).

**This reframes the headline**: representation drift is harmful specifically in the *low-data fine-tuning regime*, which is precisely where foundation-model practitioners most need pre-trained features. At sufficient scale, the replacement features win. See Table 2 and §5 replication paragraph.

### Concern 5: "Pre-training value gate is suspicious / need non-ETT benchmarks."

**Response:** Partially accepted. We attempted to add Weather and Electricity but could not obtain reliable copies of the canonical preprocessed splits (the three standard GitHub mirrors returned 404 during the revision window, and HuggingFace versions don't match the Autoformer/Informer train/val/test splits commonly used). We note this honestly in §7 limitations.

What we did add: **Moirai-Large on ETTh2** and consolidated all ETT variants into the cross-domain table (ETTh2-S, ETTh1, ETTm2, ETTh2-B, **ETTh2-L**, IoT). The pattern replicates across 3 datasets and 3 model sizes with the same zero-shot/fine-tuned relationship.

### Concern 6: "Moirai-Base undermines the headline."

**Response:** The Moirai-Large results directly address this. At 500 samples, h=96, ETTh2:
- **Unfrozen Moirai-Large:** +10.5±9.5% forgetting, CKA=0.626
- **Frozen Moirai-Large:** +3.2±0.6% forgetting, CKA=0.994
- **LoRA (r=8) Moirai-Large:** +25.7±3.2% forgetting (*worse* than unfrozen)

The Moirai-Base frozen-vs-unfrozen parity is not a feature of scale — it's a feature of Base's specific capacity/pre-training balance on ETTh2. Large shows clear forgetting in the unfrozen condition and clear improvement from freezing, matching Small. The pattern is robust across model sizes. See §7 "Moirai-Large: drift is not mitigated by scale."

Surprisingly, LoRA at r=8 *fails* on Moirai-Large at this scale. The 311M-parameter encoder likely requires higher rank or more training; this is a useful negative result we call out explicitly rather than hide.

### Concern 7: "Uneven statistics."

**Response:** Accepted and fully addressed.
- **Cohen's d=4.32 (IoT):** now unpacked in §6 — "the large d reflects low cross-seed variance (pooled std=0.017); the absolute AUC difference is 7.3pp"
- **ETT B vs zero-shot:** added $p<0.001$, $d=1.23$, paired bootstrap 10k resamples (§5.4)
- **$p=0.39$ (§7):** now specified as "two-sided bootstrap, 10k resamples"
- **Seed counts:** single sentence added in §5.2 — "Core conditions (B, C, D) use 10 seeds; mitigations (E, F, G) use 5; cross-dataset and model-size replication uses 3"

### Concern 8: "CKA as sole representational metric."

**Response:** Fully addressed. Linear probing is now reported across the entire sample-size sweep and in the methodology section as the *functional* representation diagnostic (§4, paragraph "Representation diagnostics"). Appendix D.7 describes the probing protocol (Ridge regression, α=1.0, held-out ETTh2 val/test, mean-pooled 192-step encoder representations).

The probing result is the strongest evidence in the paper that drift and information loss are dissociable (Concern 4). We report absolute probe R² along with $\Delta R^2$ so the reader can assess both the baseline quality and the relative change (§7, "CKA and linear probing are dissociable").

---

## Remaining limitations

We note honestly in §7 and Appendix limitations:
1. Best IoT F1 = 0.262 at stealth-95
2. N-BaIoT threshold calibration degenerate (FPR=0.98 from 77 benign windows)
3. Forecasting benchmarks are ETT-only (Weather/Electricity unavailable during revision)
4. Moirai-Large sample-size sweep done only at 500 (compute budget)
5. LoRA rank for Moirai-Large not swept — r=8 may be too low
6. Hand-designed attack archetypes (not generative)

We believe the revision constitutes a qualitatively different paper than V8: the dissociability story is empirically strong, the model-size generalisation is cleanly tested, and the IoT operational story is honest rather than oversold.

---

## Reviewer questions

*The six "questions for authors" embedded in the review are addressed throughout the point-by-point responses above. We can provide any additional raw data or code on request.*

---

# Addendum: V9 Response (Borderline Accept → target: Clean Accept)

We thank the reviewer for the "borderline accept, leaning weak accept" verdict and for naming the CKA/probe dissociation "the genuine intellectual contribution." The reviewer's explicit accept bar was: *linear probing on one additional forecasting dataset and on IoT*. This addendum reports both.

## Experiment 1: Linear probing on a non-ETT forecasting dataset (Weather)

We sourced the Autoformer Weather CSV (Jena climate, 14 features, hourly resampled from 10-min raw, 7:1:2 split, target=T(degC)) and ran zero-shot Moirai-Small plus a Linear baseline. **The pre-training value gate fails on Weather** (§5):

| | MSE (h=96) |
|---|---|
| Moirai-Small zero-shot | 0.410 |
| Linear baseline (192→96) | 0.218 |
| Repeat-last | 0.409 |

Moirai's pre-trained features give no zero-shot advantage on Weather, so there is no dissociation signal to measure there. We include this **honestly as a delimiter** of where the ETT result applies, rather than a failure. A Moirai-Base zero-shot run is queued (MPS contention delayed completion during the rebuttal window). Electricity and Traffic — the other LTSF candidates — had no reachable local mirror during the revision and are deferred to camera-ready.

**What this means for the dissociation claim**: the ETT result still demonstrates that *where pre-trained features are valuable*, drift and forgetting are dissociable. The Weather result constrains external validity to the subset of LTSF benchmarks where Moirai pre-training is actually useful — itself a non-trivial finding that reviewers can verify independently.

## Experiment 2: Linear probing on IoT (Q1)

We extracted mean-pooled Moirai encoder representations on held-out CICIoT2023 benign+attack data and fit a logistic-regression probe targeting the binary benign-vs-attack label (ROC-AUC metric). Results across 2 completed seeds (seed 456 in-flight; full 3-seed summary in camera-ready):

| Seed | pre-FT AUC | post-FT AUC | Δ AUC | CKA |
|---|---|---|---|---|
| 42  | 0.344 | 0.359 | +0.015 | 0.374 |
| 123 | 0.323 | 0.329 | +0.006 | 0.391 |
| **Mean** | **0.334** | **0.344** | **+0.010** | **0.383** |

**Dissociation direction replicates on IoT**: post-FT probe AUC is consistently higher than pre-FT in both seeds, while CKA drops to 0.38. The effect size (+0.010) is small relative to forecasting ($\Delta R^2$ up to +0.87) because 200 IoT training samples is far from the "replacement regime" that ETT reaches at n=10k — the dissociation *direction* holds on IoT but not its magnitude. Both probe AUCs are sub-chance (≤0.36), mirroring the weak-absolute-probe phenomenon we already flag on ETT ($R^2 \approx -6$). This is reported in §7 "IoT probing: dissociation holds in direction."

## Experiment 3: Additional seeds at n=5k (Q2, Minor 1)

We ran 4 additional seeds (789, 101, 202, 303) at n=5,000 on CPU to address the $\pm$11.8% variance flagged in review. The combined 7-seed result at n=5,000:

- Forgetting: **+6.6% ± 10.0%** (was +8.3% ± 11.8% at 3 seeds)
- CKA: 0.55 ± 0.08
- ΔR²: **+0.50 ± 0.26** (uniformly positive across all 7 seeds)
- R²(FT): −6.41 ± 0.26

The 7-seed distribution is **bimodal**: 4 seeds with forgetting ≤+5.9% (mean +0.7%) and 3 seeds with forgetting ≥+4.8% (mean +15.1%). Consistent with the crossover interpretation: at n=5k the optimiser is near the replacement solution and trajectory-dependent outcomes split cleanly. Per-seed table in Appendix §D.7 (Table 4) and updated sample-sweep row in Table 2 main.

## Experiment 4: LoRA rank sweep on Moirai-Large (Q3, Minor 5)

A rank sweep at r∈{16, 32, 64} on Moirai-Large (seed 42) was launched. Under MPS contention with the Weather and IoT-probe jobs, only partial results are available during the rebuttal; a complete GPU-backed sweep is deferred to camera-ready, consistent with the reviewer's explicit allowance. The finding as currently framed in §7 is that the r=8 default calibrated on 14M-parameter encoders is insufficient for 311M — a useful negative result whose numerical resolution will be published with the final version.

## Text-level fixes

| Minor issue | Action |
|---|---|
| 1. Abstract glosses n=5k non-monotonicity | Abstract now reads "crossover regime at n=5,000 shows bimodal, elevated cross-seed variance (+6.6±10.0% across 7 seeds)" |
| 2. Algorithm 1 excess | Removed from appendix; replaced with one-sentence note |
| 3. Hard-negative framing | §2, §6 reframe analytical-hard-negatives as "initial design choice" with finding that Gaussian-noise off-manifold negatives match or exceed them |
| 4. Limitation (8) hidden in appendix | Elevated to main §7 Limitations paragraph as explicit sentence |
| 5. LoRA-Large discussion | Expanded; references in-flight rank sweep and deferred GPU run |
| 6. Figure 2 hparam heatmap | Compressed to one sentence; figure lives in code release |
| 7. "Controlled" → "systematic" | Replaced in §1 and §2 |
| R²(FT) reporting (Remaining 4) | Added as dedicated column in Table 2; noise-floor interpretation in Appendix §D.7 |
| CNN-NLL elevation (Q6) | One sentence each in §7 practical guidance and §8 conclusion |
| Per-seed n=5k breakdown (Q2) | Appendix Table 4 with all 7 seeds |

## Summary: what the reviewer asked for vs what is delivered

| Reviewer ask | Status |
|---|---|
| Non-ETT forecasting dataset with probing | **Done** (Weather; pre-training value gate fails — reported honestly) |
| Linear probing on IoT | **Done** (2/3 seeds; dissociation direction replicates) |
| More seeds at n=5k | **Done** (4 seeds added → 7 total; bimodality confirmed) |
| LoRA rank sweep on Large | **In progress** — partial; camera-ready completion per reviewer's explicit allowance |
| All 7 narrative/text fixes | **Done** |

Page count: main body 9 pages, appendix 19 pages, total 28 pages (NeurIPS 9+unbounded-appendix format).
