#!/usr/bin/env python3
"""
The intervention matrix: every paired B/D cell in the repo, scored on held-out windows.

WHY THIS EXISTS
---------------
The paper's claims are cross-cell claims, so they need one table built by one rule from every cell
we have, not a hand-assembled list. This module discovers paired B/D cells, attaches the correct
zero-shot TEST denominator to each, and emits B-D on both validation and held-out windows.

THE DENOMINATOR MATTERS MORE THAN IT LOOKS. B-D divides by a shared zero-shot term. Each run stores
`zeroshot_mse`, which is measured on VALIDATION. Using it against a test-window MSE difference mixes
scales: on Moirai-Small/ETTh2 h=96 the test zero-shot is 0.492 against validation's 0.129, so the
val denominator inflates held-out B-D by ~3.8x. Cells without a test reference are reported as
sign-only and excluded from the quantitative analyses.

Run:  python3 scripts/cell_matrix.py
"""
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Run dirs whose cells carry their OWN condition A, laid out as <dir>/<cell>/condition_{A,B,D}/.
# One list, read by new_moirai_cells() and by the strict-freeze pairing, so a cell that is scorable
# in one is scorable in the other. They diverged once and the consequence was silent: two cells sat
# in the matrix but could not be paired with a condition-H arm.
OWN_A_DIRS = ("results/v43_moirai_matrix", "results/v47_prospective")

# Condition-H (strict freeze) run dirs, all laid out as <dir>/<cell>/condition_H/.
# v45 covers ETTh1/Weather/ILI; v51 the five cells clearing the fitted-linear gate; v53 the four
# Moirai/ETTm2 cells that clear it under another rung of the baseline ladder.
STRICT_FREEZE_DIRS = ("results/v45_strict_freeze", "results/v51_strictfreeze_etth2",
                      "results/v53_strictfreeze_ettm2")

# Condition-E (LoRA, r=8, alpha=16) run dirs, same <dir>/<cell>/condition_E/ layout. v54 is the arm
# that took LoRA off Moirai-Small/ETTh2: before it, E existed on two cells and every "LoRA is the
# cheap win" sentence in this project generalised from one of them.
LORA_DIRS = ("results/v54_lora_valuecells",)


def _sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


GATE_SPLIT = "val"           # set from --gate-split; see _gates()


def _gates():
    """cell-ref -> gate R2_task, from scripts/gate_all_cells.py.

    GATE_SPLIT="val" is the PRIMARY gate and the default as of 20 Sep 2026: the gate is scored on
    the SELECTION windows, which are disjoint from the held-out windows B-D is scored on. Scoring
    it on the held-out windows instead -- what this file did through 20 Sep 2026 -- makes cell
    INCLUSION a retrospective judgement, the criterion applied after the outcome split has been
    looked at, so "N cells have something to preserve" was never an out-of-sample statement. That
    circularity is what a reviewer caught and it is not a framing problem: on the selection split
    7 of 31 cells clear 0.20, not 5, and two of the seven are ETTm2 rather than ETTh2.

    GATE_SPLIT="test" reproduces that retrospective variant, which the paper still reports as a
    sensitivity analysis. Forgetting terms stay test-side either way, so only the inclusion
    decision changes -- and no cell meets the three-clause definition under EITHER split, which is
    the one thing the flip does not move.

    The pre-registered prospective predictor is deliberately NOT re-split: it was frozen in git
    against the retrospective gate, and re-scoring it on the selection split now would be a
    post-hoc change to a registered rule. See preregister_prospective*.py and score_prospective.py.
    """
    p = ROOT / f"results/gate_{GATE_SPLIT}_side.json"
    if not p.exists():
        return {}
    return {k: v["r2_task"] for k, v in json.load(open(p)).items()}


def _own_a_glob(name, cond):
    """Every run dir whose cells carry their own condition A, globbed for one cell and condition."""
    return [f for d in OWN_A_DIRS
            for f in glob.glob(str(ROOT / d / name / f"{cond}/*.json"))]


def _b_arm_for(name, refs):
    """
    (seed -> condition-B run, zero-shot test MSE) for a strict-freeze cell.

    The six degradation cells do not all live in one results dir: Weather and Moirai-Base/ETTh1 were
    run for this revision under results/v43_moirai_matrix (each with its own condition A), while
    Moirai-Small/ETTh1 comes from the earlier results/v5_etth1 sweep and takes its denominator from
    the shared zero-shot test references. Pairing only against v43 silently dropped the two
    Small/ETTh1 cells from the strict-freeze report even though condition H had run on them, which is
    exactly the kind of quiet omission this table exists to prevent.

    It happened a second time, the same way. This searched v43 only, while new_moirai_cells() reads
    v43 AND v47_prospective -- so the two Moirai/Base/ETTm2 cells were in the matrix but invisible
    here, and condition H on them would have run and then silently failed to appear in any table.
    Both now iterate OWN_A_DIRS, so a cell that is scorable in one place is scorable in the other.
    """
    B = {json.load(open(f))["seed"]: json.load(open(f))
         for f in _own_a_glob(name, "condition_B")}
    A = [json.load(open(f)) for f in _own_a_glob(name, "condition_A")]
    A = [x for x in A if "zeroshot_test_mse" in x]
    if B and A:
        return B, st.mean(x["zeroshot_test_mse"] for x in A)
    size, ds, h = name.split("_")
    h = int(h.lstrip("h"))
    for (rel, sz, hz, _n), seeds in moirai_cells().items():
        if sz == size and hz == h and DATASET_OF.get(rel) == ds:
            B = {s: v["B"] for s, v in seeds.items() if "B" in v and "test_mse" in v["B"]}
            if B:
                return B, refs.get(f"{size}_{ds}_h{h}")
    return {}, None


def _d_arm_for(name):
    """seed -> condition-D run for a strict-freeze cell, from whichever dir holds it."""
    D = {json.load(open(f))["seed"]: json.load(open(f))
         for f in _own_a_glob(name, "condition_D")}
    if D:
        return D
    size, ds, h = name.split("_")
    h = int(h.lstrip("h"))
    for (rel, sz, hz, _n), seeds in moirai_cells().items():
        if sz == size and hz == h and DATASET_OF.get(rel) == ds:
            D = {s: v["D"] for s, v in seeds.items() if "D" in v and "test_mse" in v["D"]}
            if D:
                return D
    return {}


def strict_freeze_cells():
    """
    B-H, the strict-freeze control: ref -> held-out B-H beside B-D.

    Condition D freezes the encoder's weights but leaves in_proj and mask_encoding trainable, so the
    encoder's OUTPUT is not a fixed function of its input and D's CKA lands at 0.76-0.99 rather than
    1.0. Condition H freezes those too, so only param_proj trains and CKA is 1.0 by construction.
    The reading rule was fixed before the runs: if B-H agrees in sign and magnitude with B-D, the
    "freezing wins" reading is not an artifact of the input projection re-fitting; divergence on a
    cell means part of that cell's gap is input re-fitting and gets said so.
    """
    out = {}
    refs = _zs_test_refs()
    # STRICT_FREEZE_DIRS: several directories, one reading rule. The cell-name spaces are disjoint
    # by dataset, and the assert keeps them that way: a collision would silently overwrite one arm's
    # numbers with the other's rather than fail.
    sf_dirs = [f for d in STRICT_FREEZE_DIRS
               for f in sorted(glob.glob(str(ROOT / d / "*")))]
    for d in sf_dirs:
        name = Path(d).name
        if name == "ili":
            continue
        assert name not in out, f"strict-freeze cell {name} appears in two result directories"
        H = {json.load(open(f))["seed"]: json.load(open(f))
             for f in glob.glob(d + "/condition_H/*.json")}
        B, zs_t = _b_arm_for(name, refs)
        D = _d_arm_for(name)
        seeds = sorted(set(B) & set(H) & set(D))
        if not (zs_t and seeds):
            continue
        bh = [(B[s]["test_mse"] - H[s]["test_mse"]) / zs_t * 100 for s in seeds]
        # B-D restricted to the seeds H ran on, so the comparison is like for like: Small/ETTh1's
        # published B-D averages 5 seeds while condition H ran 3.
        bd = [(B[s]["test_mse"] - D[s]["test_mse"]) / zs_t * 100 for s in seeds]
        fh = [(H[s]["test_mse"] - zs_t) / zs_t * 100 for s in seeds]
        out[name] = dict(bh_test=st.mean(bh), bh_test_sd=_sd(bh), seeds=len(seeds),
                         bd_same=st.mean(bd), bd_same_sd=_sd(bd),
                         forg_h=st.mean(fh), forg_h_sd=_sd(fh), forg_h_pos=sum(x > 0 for x in fh),
                         cka_h=st.mean(H[s]["final_cka"] for s in seeds))
    # ILI runs through finetune_ili.py, which stores aggregate percentages rather than MSEs.
    H = {json.load(open(f))["seed"]: json.load(open(f))
         for f in glob.glob(str(ROOT / "results/v45_strict_freeze/ili/condition_H_seed*.json"))}
    B = {json.load(open(f))["seed"]: json.load(open(f))
         for f in glob.glob(str(ROOT / "results/v40_ili_heldout/condition_B_seed*.json"))}
    Di = {json.load(open(f))["seed"]: json.load(open(f))
          for f in glob.glob(str(ROOT / "results/v40_ili_heldout/condition_D_seed*.json"))}
    seeds = sorted(set(B) & set(H) & set(Di))
    if seeds:
        bh = [B[s]["aggregate"]["forgetting_pct_test"] - H[s]["aggregate"]["forgetting_pct_test"]
              for s in seeds]
        bd = [B[s]["aggregate"]["forgetting_pct_test"] - Di[s]["aggregate"]["forgetting_pct_test"]
              for s in seeds]
        fh = [H[s]["aggregate"]["forgetting_pct_test"] for s in seeds]
        out["ili"] = dict(bh_test=st.mean(bh), bh_test_sd=_sd(bh), seeds=len(seeds),
                          bd_same=st.mean(bd), bd_same_sd=_sd(bd),
                          forg_h=st.mean(fh), forg_h_sd=_sd(fh), forg_h_pos=sum(x > 0 for x in fh),
                          cka_h=st.mean(H[s]["aggregate"]["cka"] for s in seeds))
    return out


def lora_value_cells():
    """
    B-E, the constrained-adaptation arm: ref -> held-out B-E beside B-D on the same seeds.

    WHAT E IS AND IS NOT. Conditions B and D bracket the adaptation axis -- everything trains, or the
    encoder body does not. E sits between them: LoRA adds trainable low-rank updates to the encoder's
    attention projections, so the encoder's function changes while its base weights do not. That is
    why final_weight_drift is exactly 0.0 in every one of these records and is not a bug: l2 drift
    measures movement of the base parameters, and LoRA moves none. CKA is the measure that still
    reads on E, and it is the reason E belongs in the drift analysis at all.

    Delta_encoder stays B - D. B - E answers the practitioner's question instead -- would a cheap
    constrained adaptation have beaten the extreme this cell actually ran -- and it is reported beside
    B - D on the SAME seeds, for the reason spelled out in strict_freeze_cells(): E ran 3 seeds where
    Small/ETTh1 and Small/ETTm2 published B-D over 5.

    Sign convention matches B-D and B-H throughout: positive means full fine-tuning did WORSE than
    the comparison arm, i.e. the constrained option won.
    """
    out = {}
    refs = _zs_test_refs()
    for d in [f for dd in LORA_DIRS for f in sorted(glob.glob(str(ROOT / dd / "*")))]:
        name = Path(d).name
        assert name not in out, f"LoRA cell {name} appears in two result directories"
        E = {json.load(open(f))["seed"]: json.load(open(f))
             for f in glob.glob(d + "/condition_E/*.json")}
        B, zs_t = _b_arm_for(name, refs)
        D = _d_arm_for(name)
        seeds = sorted(set(B) & set(E) & set(D))
        if not (zs_t and seeds):
            continue
        be = [(B[s]["test_mse"] - E[s]["test_mse"]) / zs_t * 100 for s in seeds]
        bd = [(B[s]["test_mse"] - D[s]["test_mse"]) / zs_t * 100 for s in seeds]
        fe = [(E[s]["test_mse"] - zs_t) / zs_t * 100 for s in seeds]
        # D-E, paired per seed: "did the constrained middle beat the frozen extreme". Kept as its own
        # paired contrast rather than left to be read off the difference of the B-E and B-D means,
        # because on Small/ETTh1 those two means sit 0.2 pp apart with a SEM of 4.9 -- a difference of
        # means invites "E beat D" where the paired contrast shows the cell cannot tell them apart.
        de = [(D[s]["test_mse"] - E[s]["test_mse"]) / zs_t * 100 for s in seeds]
        # bd_same_e, NOT bd_same: strict_freeze_cells() already publishes bd_same, restricted to the
        # seeds condition H ran. Six of these eight cells have BOTH arms, so reusing the name would let
        # whichever ran second overwrite the other's column -- silently changing the B-D printed in
        # strictfreeze.tex, which the prose checker has claims registered against.
        out[name] = dict(be_test=st.mean(be), be_test_sd=_sd(be), be_seeds=len(seeds),
                         bd_same_e=st.mean(bd), bd_same_e_sd=_sd(bd), e_seedset=seeds,
                         forg_e=st.mean(fe), forg_e_sd=_sd(fe), forg_e_pos=sum(x > 0 for x in fe),
                         be_pos=sum(x > 0 for x in be),
                         de_test=st.mean(de), de_test_sd=_sd(de), de_pos=sum(x > 0 for x in de),
                         cka_e=st.mean(E[s]["final_cka"] for s in seeds),
                         drift_e=st.mean(E[s]["final_weight_drift"] for s in seeds),
                         lora_rank=E[seeds[0]]["lora_rank"])
    return out


def _per_seed(seeds, d_enc, d_ft, d_frozen, ft_paired, zs_seeds=None, zs_sem=None):
    """The per-seed contrasts a row is built from, kept rather than averaged away.

    Every cell source already computes these lists to take their means; discarding them forced the
    paired analyses to re-derive them from the run files under a second set of conventions, which is
    how a denominator drifts. All three are in the published units -- percent of the cell's zero-shot
    reference -- so a mean over d_enc reproduces bd_test exactly, and a mean over d_ft reproduces
    forg_b.

    ft_paired records whether the ZERO-SHOT term is per-seed. It is on Chronos and TimesFM, whose run
    files carry a seed's own zero-shot test MSE, and it is NOT on the Moirai cells, which divide by a
    per-cell mean over condition-A seeds. Where it is False, d_ft and d_frozen vary only through the
    intervention arm and the reference's own measurement error is not propagated into their intervals;
    zs_seeds and zs_sem say how large that unpropagated term is instead of leaving it implicit.
    """
    return dict(seeds=list(seeds), d_enc=list(d_enc), d_ft=list(d_ft), d_frozen=list(d_frozen),
                ft_paired=ft_paired, zs_seeds=zs_seeds, zs_sem=zs_sem)


def _sem(xs):
    return _sd(xs) / (len(xs) ** 0.5) if len(xs) > 1 else 0.0


def _zs_test_by_seed():
    """dataset-key -> {seed: zero-shot test MSE}, every condition-A measurement on disk.

    Split out of _zs_test_refs so the paired analyses can pair on seed wherever the zero-shot arm ran
    the same seeds as the intervention arm. Zero-shot test MSE is NOT seed-invariant: the evaluation
    windows are sampled per seed, and base_ETTh2_h192 spans 0.4224-0.4434 across its three. Averaging
    that away and then calling the resulting contrast "paired" would claim a pairing the data does not
    have. Keying by seed is lossless here -- no dataset-key receives two measurements of the same seed
    from two directories, checked before this refactor -- so the mean below is unchanged.
    """
    refs = defaultdict(dict)

    def put(key, path):
        d = json.load(open(path))
        if "zeroshot_test_mse" in d:
            refs[key][d.get("seed")] = d["zeroshot_test_mse"]

    for f in glob.glob(str(ROOT / "results/v41_zs_test/*/condition_A_*.json")):
        put(Path(f).parent.name, f)            # e.g. small_ETTh2_h96
    for p in OWN_A_DIRS:
        for f in glob.glob(str(ROOT / p / "*/condition_A/*.json")):
            put(Path(f).parent.parent.name, f)
    # the Moirai-Base n=1k cell measured earlier, under its own directory
    for f in glob.glob(str(ROOT / "results/v39_moirai_zs_test/h96/condition_A/*.json")):
        put("base_ETTh2_h96", f)
    return refs


def _zs_test_refs():
    """dataset-key -> zero-shot test MSE, averaged over the seeds we measured it on."""
    return {k: st.mean(v.values()) for k, v in _zs_test_by_seed().items()}


def moirai_cells():
    """Every paired Moirai B/D cell with test_mse stored, keyed by (dir, size, horizon, n)."""
    cells = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(str(ROOT / "results/**/*.json"), recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("condition") not in ("B", "D"):
            continue
        if "final_val_mse" not in d or "test_mse" not in d:
            continue
        rel = Path(f).relative_to(ROOT).parts[1]
        cells[(rel, d.get("model_size") or "small", d.get("horizon"),
               d.get("max_train_samples"))][d.get("seed")][d["condition"]] = d
    return cells


DATASET_OF = {   # results dir -> dataset, for attaching the right zero-shot reference
    "forecasting_finetune_20ep": "ETTh2", "v5_etth1": "ETTh1", "v5_ettm2": "ETTm2",
    "v5_etth2_base": "ETTh2", "v8_etth2_large": "ETTh2",
}


def new_moirai_cells(dirs=OWN_A_DIRS, extra_a_dirs=()):
    """Moirai cells that carry their own condition A.

    Two directories share one layout: results/v43_moirai_matrix (the cells added for the previous
    revision) and results/v47_prospective (the pre-registered prospective arm). They are read by
    the same code so a prospective cell is never scored differently from a published one.

    `extra_a_dirs` exists for batch 3 of the prospective arm, whose condition-A records are split
    across three directories by history: the pool screen measured six cells' zero-shot in
    results/v48_prospective2 (an abandoned batch 2) and three in results/v56_pool3 before the batch
    was registered, and the two extra reference seeds were then run into results/v57_prospective3
    beside the outcomes. A cell's reference is the union of its condition-A records across the cell's
    own directory and these, looked up BY CELL NAME. The alternative was to copy those records into
    the batch-3 tree, which would have put two copies of one measurement in the release, or to write a
    second reader, which is how a denominator drifts. Default is empty, so nothing published moves.
    """
    out = []
    for d in sorted(f for p in dirs for f in glob.glob(str(ROOT / p / "*"))):
        if not Path(d).is_dir():
            continue
        name = Path(d).name                       # e.g. small_Weather_h96
        a_files = glob.glob(d + "/condition_A/*.json")
        for p in extra_a_dirs:
            a_files += glob.glob(str(ROOT / p / name / "condition_A*" / "*.json"))
        A = [json.load(open(f)) for f in sorted(set(a_files))]
        A = [x for x in A if "zeroshot_test_mse" in x]
        # One measurement per seed. Two directories holding the same seed would weight that seed twice
        # in the reference mean and shrink its SEM, which is the quantity the B-ZS and D-ZS intervals
        # are widened by -- so this is asserted rather than deduplicated silently.
        a_seeds = [x.get("seed") for x in A]
        assert len(set(a_seeds)) == len(a_seeds), f"{name}: duplicate condition-A seeds {a_seeds}"
        B = {json.load(open(f))["seed"]: json.load(open(f))
             for f in glob.glob(d + "/condition_B/*.json")}
        D = {json.load(open(f))["seed"]: json.load(open(f))
             for f in glob.glob(d + "/condition_D/*.json")}
        seeds = sorted(set(B) & set(D))
        if not (A and seeds):
            continue
        zs_t = st.mean(x["zeroshot_test_mse"] for x in A)
        bv = [(B[s]["final_val_mse"] - D[s]["final_val_mse"]) / B[s]["zeroshot_mse"] * 100
              for s in seeds]
        bt = [(B[s]["test_mse"] - D[s]["test_mse"]) / zs_t * 100 for s in seeds]
        fb = [(B[s]["test_mse"] - zs_t) / zs_t * 100 for s in seeds]
        fd = [(D[s]["test_mse"] - zs_t) / zs_t * 100 for s in seeds]
        zs_vals = [x["zeroshot_test_mse"] for x in A]
        ps = _per_seed(seeds, bt, fb, fd, ft_paired=False,
                       zs_seeds=len(zs_vals), zs_sem=_sem(zs_vals) / zs_t * 100)
        out.append(dict(per_seed=ps,
                        cell=f"Moirai-{name}", seeds=len(seeds), pos=sum(x > 0 for x in bt),
                        bd_val=st.mean(bv), bd_val_sd=_sd(bv),
                        bd_test=st.mean(bt), bd_test_sd=_sd(bt),
                        forg_b=st.mean(fb), forg_b_sd=_sd(fb), forg_b_pos=sum(x > 0 for x in fb),
                        forg_d=st.mean(fd), forg_d_sd=_sd(fd), forg_d_neg=sum(x < 0 for x in fd),
                        cka=st.mean(B[s]["final_cka"] for s in seeds),
                        drift=st.mean(B[s].get("final_weight_drift", float("nan")) for s in seeds),
                        n_train=_n_train([B[s] for s in seeds] + [D[s] for s in seeds], name),
                        ref=name, has_ref=True))
    return out


def chronos_cells(root="results/v44_chronos_guarded", horizon=24):
    """
    Chronos cells. Per seed the headline condition-D estimator is the one VALIDATION preferred --
    the rule pre-committed before these datasets were run. That matters here: on series with
    near-degenerate windows the closed-form ridge can return a broken fit at an interior alpha,
    so 'interior' alone is not a sufficiency check and the comparison against the AdamW leg is.
    """
    out = []
    for ds in ("etth1", "etth2", "weather", "ettm2", "electricity"):
        B = {json.load(open(f))["seed"]: json.load(open(f))
             for f in glob.glob(str(ROOT / root / f"cond_B/mse_{ds}/seed*/condition_B_s*.json"))}
        D = {json.load(open(f))["seed"]: json.load(open(f))
             for f in glob.glob(str(ROOT / root / f"cond_D/mse_{ds}/seed*/condition_D_s*.json"))}
        seeds = sorted(set(B) & set(D))
        if not seeds:
            continue
        bv, bt, cka, bad, fb, fd = [], [], [], 0, [], []
        for s in seeds:
            b, d = B[s], D[s]
            ridge_ok = d["ridge_optimum"]["best_val_loss"] <= d["best_val_loss"] * 1.05
            if not ridge_ok:
                bad += 1
            dv = (d["best_val_loss_ols"] if ridge_ok else d["best_val_loss"]) / horizon
            dt = (d["test_mse_per_element_ols"] if ridge_ok
                  else d["test_mse_per_element_adamw"])
            zst = (d["zs_mse_test"] + b["zs_mse_test"]) / 2
            bv.append((b["best_val_loss"] / horizon - dv) / d["zs_mse"] * 100)
            bt.append((b["test_mse_per_element"] - dt) / zst * 100)
            cka.append(b["final_cka"])
            # Confounded by the head/decoder mismatch (limitation iv) -- kept so predicts_bd() can
            # show WHERE the forg_B rule breaks, never quoted as a forgetting measurement.
            fb.append((b["test_mse_per_element"] - zst) / zst * 100)
            fd.append((dt - zst) / zst * 100)
        out.append(dict(per_seed=_per_seed(seeds, bt, fb, fd, ft_paired=True),
                        cell=f"Chronos/{ds} h{horizon}", seeds=len(seeds),
                        pos=sum(x > 0 for x in bt), bd_val=st.mean(bv), bd_val_sd=_sd(bv),
                        bd_test=st.mean(bt), bd_test_sd=_sd(bt), cka=st.mean(cka),
                        forg_b=st.mean(fb), forg_b_sd=_sd(fb), forg_b_pos=sum(x > 0 for x in fb),
                        forg_d=st.mean(fd), forg_d_sd=_sd(fd), forg_d_neg=sum(x < 0 for x in fd),
                        forg_confounded=True,
                        n_train=_n_train([B[s] for s in seeds] + [D[s] for s in seeds],
                                         f"Chronos/{ds}"),
                        drift=float("nan"), ref=f"chronos_{ds}", has_ref=True,
                        note=f"{bad}/{len(seeds)} seeds fell back to AdamW" if bad else ""))
    return out


def timesfm_cells(root="results/v46_timesfm", horizon=24, ref_suffix=""):
    """
    TimesFM 2.5 cells -- the third backbone (scripts/finetune_timesfm.py).

    Unlike the Chronos arm, no head is attached: A, B and D all score through TimesFM's OWN output
    head via the same differentiable native path, so forg_B and forg_D are genuine per-condition
    forgetting numbers here and are NOT flagged forg_confounded. That is what makes this arm a
    second, independent test of the forg_B predictor rule from within-backbone data.

    The zero-shot denominator comes from the run files themselves (`zeroshot_test_mse`), which
    finetune_timesfm.py computes on the SAME build_windows(test, 96, 24, max_windows=200, seed=0)
    set that gate_all_cells.timesfm_gates() screens on -- so gate, ZS reference and B-D share one
    window set by construction rather than by convention. B and D are paired on seed, as everywhere
    else; an unpaired seed contributes nothing.

    `ref` carries no horizon by default, because the published TimesFM cells are all at h=24 and the
    gate cache keys them that way. Batch 3 of the prospective arm runs the same five datasets at
    h=48, whose gate keys DO carry the horizon (timesfm_etth1_h48), so that caller passes
    ref_suffix="_h48". Without it two cells of one dataset at different horizons would share a ref and
    the second would silently take the first's gate.
    """
    out = []
    for ds in ("ETTh1", "Weather", "ETTm2", "ETTh2", "Electricity"):
        base = ROOT / root / f"{ds}_h{horizon}"
        B = {json.load(open(f))["seed"]: json.load(open(f))
             for f in glob.glob(str(base / f"condition_B/condition_B_h{horizon}_s*.json"))}
        D = {json.load(open(f))["seed"]: json.load(open(f))
             for f in glob.glob(str(base / f"condition_D/condition_D_h{horizon}_s*.json"))}
        seeds = sorted(set(B) & set(D))
        if not seeds:
            continue
        bv, bt, cka, drift, fb, fd = [], [], [], [], [], []
        for s in seeds:
            b, d = B[s], D[s]
            zs_t, zs_v = b["zeroshot_test_mse"], b["zeroshot_mse"]
            bv.append((b["final_val_mse"] - d["final_val_mse"]) / zs_v * 100)
            bt.append((b["test_mse"] - d["test_mse"]) / zs_t * 100)
            cka.append(b["final_cka"])
            drift.append(b["final_weight_drift"])
            fb.append((b["test_mse"] - zs_t) / zs_t * 100)
            fd.append((d["test_mse"] - zs_t) / zs_t * 100)
        out.append(dict(per_seed=_per_seed(seeds, bt, fb, fd, ft_paired=True),
                        cell=f"TimesFM/{ds} h{horizon}", seeds=len(seeds),
                        pos=sum(x > 0 for x in bt), bd_val=st.mean(bv), bd_val_sd=_sd(bv),
                        bd_test=st.mean(bt), bd_test_sd=_sd(bt), cka=st.mean(cka),
                        forg_b=st.mean(fb), forg_b_sd=_sd(fb), forg_b_pos=sum(x > 0 for x in fb),
                        forg_d=st.mean(fd), forg_d_sd=_sd(fd), forg_d_neg=sum(x < 0 for x in fd),
                        n_train=_n_train([B[s] for s in seeds] + [D[s] for s in seeds],
                                         f"TimesFM/{ds}"),
                        drift=st.mean(drift), ref=f"timesfm_{ds.lower()}{ref_suffix}", has_ref=True))
    return out


# finetune_ili.py has no --max-train-samples: it trains on the WHOLE training split, so its run
# records carry no n field and the number cannot be read out of them the way every other arm's can.
# Derived instead from the split the script performs: national_illness.csv is 966 rows, train_ratio
# 0.6 -> 579 rows, and make_sequences() with lookback 104 and horizon 24 yields 579-104-24+1 = 452
# windows. Hardcoded rather than recomputed because the CSV is gitignored and this emitter has to run
# from a clean clone (TIER A of scripts/rederive_all.sh); a recomputation here would move the table
# into TIER B.
ILI_N_TRAIN = 452


def ili_cell():
    """Moirai-Small/ILI. Separate script path (finetune_ili.py), so it needs its own reader."""
    B = {json.load(open(f))["seed"]: json.load(open(f))
         for f in glob.glob(str(ROOT / "results/v40_ili_heldout/condition_B_seed*.json"))}
    D = {json.load(open(f))["seed"]: json.load(open(f))
         for f in glob.glob(str(ROOT / "results/v40_ili_heldout/condition_D_seed*.json"))}
    seeds = sorted(set(B) & set(D))
    if not seeds:
        return []
    bv = [B[s]["aggregate"]["forgetting_pct"] - D[s]["aggregate"]["forgetting_pct"] for s in seeds]
    bt = [B[s]["aggregate"]["forgetting_pct_test"] - D[s]["aggregate"]["forgetting_pct_test"]
          for s in seeds]
    fb = [B[s]["aggregate"]["forgetting_pct_test"] for s in seeds]
    fd = [D[s]["aggregate"]["forgetting_pct_test"] for s in seeds]
    return [dict(per_seed=_per_seed(seeds, bt, fb, fd, ft_paired=True),
                 cell="Moirai-small/ILI h24", seeds=len(seeds), pos=sum(x > 0 for x in bt),
                 bd_val=st.mean(bv), bd_val_sd=_sd(bv), bd_test=st.mean(bt), bd_test_sd=_sd(bt),
                 forg_b=st.mean(fb), forg_b_sd=_sd(fb), forg_b_pos=sum(x > 0 for x in fb),
                 forg_d=st.mean(fd), forg_d_sd=_sd(fd), forg_d_neg=sum(x < 0 for x in fd),
                 cka=st.mean(B[s]["aggregate"]["cka"] for s in seeds),
                 n_train=ILI_N_TRAIN,
                 drift=float("nan"), ref="ili", has_ref=True)]


def _n_train(records, label):
    """The training-set size the records agree on, or None if they do not record it.

    Used only for display, but it asserts, because a row that averages two training-set sizes is the
    defect this project keeps producing and the n column exists to make visible.
    """
    got = {r.get("max_train_samples") for r in records}
    got.discard(None)                      # condition-A records never train, so they never carry it
    if not got:
        return None
    if len(got) > 1:
        raise SystemExit(f"{label}: records disagree on max_train_samples: {sorted(got)}")
    return got.pop()


def _display(cell, n_train=None):
    """'Moirai-small/ETTh1 h96 n1000' -> 'Moirai-S / ETTh1 ($h{=}96$, $n{=}$1000)', for LaTeX.

    EVERY row shows n, and that is a correctness matter rather than a cosmetic one. n is part of the
    protocol and it is not constant across these tables -- the ETTh2 spectrum cells ran at 500 while
    their same-dataset neighbours ran at 1,000 -- so a table where some rows name n and others stay
    silent reads as though the silent rows share their neighbour's n. They did not always. Where the
    cell key carries the token it is used; otherwise the caller passes what the run records say.
    """
    short = {"small": "S", "base": "B", "large": "L"}
    pretty = {"etth1": "ETTh1", "etth2": "ETTh2", "ettm2": "ETTm2",
              "weather": "Weather", "electricity": "Electricity"}
    suffix = f", $n{{=}}${n_train}" if n_train is not None else ""
    if cell.startswith(("Chronos", "TimesFM")):
        arm, rest = cell.split("/")
        ds, h = rest.split()
        return f"{arm} / {pretty.get(ds.lower(), ds)} ($h{{=}}{h[1:]}${suffix})"
    if "ILI" in cell:
        return f"Moirai-S / ILI ($h{{=}}24${suffix})"
    c = cell.replace("Moirai-", "").replace("_", "/", 1).replace("_h", " h")
    parts = c.replace("/", " ").split()
    size = short.get(parts[0], parts[0])
    ds, h = parts[1], parts[2][1:]
    if len(parts) > 3 and parts[3].startswith("n"):
        suffix = f", $n{{=}}${parts[3][1:]}"
    return f"Moirai-{size} / {ds} ($h{{=}}{h}${suffix})"


def emit_latex(rows, path=ROOT / "paper_8/tables/heldout_all.tex"):
    """
    The held-out appendix table, all cells, generated rather than transcribed.

    The previous revision's version of this table listed four cells while the body's Table 1 quoted
    held-out values for all 19 -- the appendix had simply not been updated when the re-scoring was
    extended, and a reviewer read the mismatch as the body overclaiming. Emitting it from the same
    function that computes the body's numbers is the only way that stays fixed.

    Every dispersion here is SEM, matching the body; the earlier table mixed SD and SEM.
    """
    def fmt(m, sd, k, bold=False):
        s = f"{m:+.1f}{{\\pm}}{sd / max(k, 1) ** 0.5:.1f}"
        return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"

    lines = [
        "% GENERATED by scripts/cell_matrix.py --latex -- do not edit by hand.",
        "\\begin{center}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
        "Cell & Seeds & Gate $\\Vb{\\text{ridge}}$ & B$-$D validation & B$-$D held-out & Reverses? \\\\",
        "\\midrule",
    ]
    for r in sorted(rows, key=lambda r: r["bd_test"]):
        rev = (r["bd_val"] > 0) != (r["bd_test"] > 0)
        gate = "---" if r["gate"] is None else (
            f"${r['gate']:+.3f}$" + ("\\,\\textsuperscript{f}" if r["gate"] < 0.20 else ""))
        lines.append(
            f"{_display(r['cell'], r.get('n_train'))} & {r['seeds']} & {gate} & "
            f"{fmt(r['bd_val'], r['bd_val_sd'], r['seeds'])} & "
            f"{fmt(r['bd_test'], r['bd_test_sd'], r['seeds'], bold=True)} & "
            f"{'yes' if rev else 'no'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path.relative_to(ROOT)}  ({len(rows)} cells)")


def emit_strictfreeze_latex(rows, path=ROOT / "paper_8/tables/strictfreeze.tex"):
    """
    The strict-freeze appendix table, generated so B-H cannot drift from the runs.

    Columns: cell, seeds condition H ran, B-D on THOSE seeds, B-H, condition H's own forgetting, and
    CKA_H (1.0000 by construction -- printed because a reader should be able to check the construction
    held rather than take it on trust).
    """
    have = [r for r in rows if r.get("bh_test") is not None]
    # MIN_SEEDS applies to the H arm in its own right. build_rows() already enforces it on the B/D
    # arms, but the H arm is joined in afterwards and had no guard of its own: while a strict-freeze
    # arm was still running, a cell with one or two finished H seeds would print here with a SEM
    # divided by that k and no indication the cell was incomplete. Excluded cells are named rather
    # than dropped silently, so the table is safe to regenerate mid-arm.
    thin = [r for r in have if (r["bh_seeds"] or 0) < MIN_SEEDS]
    if thin:
        for r in sorted(thin, key=lambda r: r["cell"]):
            print(f"  EXCLUDED from strictfreeze.tex (condition H has only {r['bh_seeds']} "
                  f"seed(s), needs {MIN_SEEDS}): {r['cell']}")
    have = [r for r in have if (r["bh_seeds"] or 0) >= MIN_SEEDS]
    if not have:
        print("\n  no condition-H runs with >= %d seeds yet; skipped strictfreeze.tex" % MIN_SEEDS)
        return

    def fmt(m, sd, k, bold=False):
        s = f"{m:+.1f}{{\\pm}}{sd / max(k, 1) ** 0.5:.1f}"
        return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"

    lines = [
        "% GENERATED by scripts/cell_matrix.py --latex -- do not edit by hand.",
        "\\begin{center}", "\\small", "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
        "Cell & Seeds & B$-$D & B$-$H & forg$_\\text{H}$ & CKA$_\\text{H}$ \\\\",
        "\\midrule",
    ]
    for r in sorted(have, key=lambda r: -r["bd_same"]):
        k = r["bh_seeds"]
        lines.append(
            f"{_display(r['cell'], r.get('n_train'))} & {k} & {fmt(r['bd_same'], r['bd_same_sd'], k)} & "
            f"{fmt(r['bh_test'], r['bh_test_sd'], k, bold=True)} & "
            f"{fmt(r['forg_h'], r['forg_h_sd'], k)} & ${r['cka_h']:.4f}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(ROOT)}  ({len(have)} cells)")


def emit_lora_latex(rows, path=ROOT / "paper_8/tables/lora_valuecells.tex"):
    """
    The LoRA arm across the value-cells, generated so B-E cannot drift from the runs.

    Columns: cell, seeds E ran, B-D on THOSE seeds, B-E, E's own forgetting with its seed count, and
    CKA_E. l2 drift is omitted as a column and stated in the caption instead: it is exactly 0.0 on
    every one of these records by construction, and a column of zeros invites the reader to conclude
    LoRA left the encoder's FUNCTION unchanged, which CKA_E here flatly contradicts.
    """
    have = [r for r in rows if r.get("be_test") is not None]
    thin = [r for r in have if (r["be_seeds"] or 0) < MIN_SEEDS]
    if thin:
        for r in sorted(thin, key=lambda r: r["cell"]):
            print(f"  EXCLUDED from lora_valuecells.tex (condition E has only {r['be_seeds']} "
                  f"seed(s), needs {MIN_SEEDS}): {r['cell']}")
    have = [r for r in have if (r["be_seeds"] or 0) >= MIN_SEEDS]
    if not have:
        print("\n  no condition-E runs with >= %d seeds yet; skipped lora_valuecells.tex" % MIN_SEEDS)
        return
    ranks = {r["lora_rank"] for r in have}
    assert ranks == {8}, f"lora_valuecells.tex assumes r=8 throughout, got ranks {ranks}"

    def fmt(m, sd, k, bold=False):
        s = f"{m:+.1f}{{\\pm}}{sd / max(k, 1) ** 0.5:.1f}"
        return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"

    lines = [
        "% GENERATED by scripts/cell_matrix.py --latex -- do not edit by hand.",
        # scriptsize, and a tighter tabcolsep than the strict-freeze table's 5pt: this table carries a
        # seventh column and does not fit the text block at larger sizes. Measured rather than
        # guessed, since an overfull hbox is the only signal a generated table has outgrown the page:
        # \small was 75.8pt too wide, \footnotesize 57.8pt, \scriptsize fits.
        "\\begin{center}", "\\scriptsize", "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
        "Cell & Seeds & B$-$D & B$-$E & D$-$E & forg$_\\text{E}$ & CKA$_\\text{E}$ \\\\",
        "\\midrule",
    ]
    # Ordered by D-E, the contrast a practitioner choosing between LoRA and freezing reads.
    for r in sorted(have, key=lambda r: -r["de_test"]):
        k = r["be_seeds"]
        lines.append(
            f"{_display(r['cell'], r.get('n_train'))} & {k} & "
            f"{fmt(r['bd_same_e'], r['bd_same_e_sd'], k)} & "
            f"{fmt(r['be_test'], r['be_test_sd'], k)} & "
            f"{fmt(r['de_test'], r['de_test_sd'], k, bold=True)} ({r['de_pos']}/{k}) & "
            f"{fmt(r['forg_e'], r['forg_e_sd'], k)} ({r['forg_e_pos']}/{k}) & "
            f"${r['cka_e']:.4f}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    path.write_text("\n".join(lines) + "\n")
    neg = sum(1 for r in have if r["be_test"] < 0)
    ck = [r["cka_e"] for r in have]
    print(f"wrote {path.relative_to(ROOT)}  ({len(have)} cells; B-E negative on {neg}, "
          f"CKA_E {min(ck):.3f}-{max(ck):.3f})")


MIN_SEEDS = 3     # see build_rows()


def build_rows(verbose=False):
    """Every intervention cell as a dict, with gate and strict-freeze fields attached.

    Split out of main() so figure scripts consume the same rows the tables and statistics do --
    fig1_diagnostic_flow.py previously hard-coded its bar values, which is how an earlier version
    came to plot 15 bars that silently omitted a cell.

    UNDER-SEEDED CELLS ARE DROPPED. Every cell the paper reports has 3-10 seeds. A cell that is
    still being produced has fewer, and one seed is not a measurement: its SEM is 0.00 by
    construction, so the "every seed agrees" clause of degradation_cells() passes trivially and
    |forg_B|/SEM is infinite. Scoring a half-finished cell alongside finished ones therefore
    manufactures a degradation cell out of a single run and shifts every rank statistic. This was
    not hypothetical -- it happened while results/v47_prospective was mid-flight, adding a
    1-seed cell to the degradation list. Cells below MIN_SEEDS are excluded and named, not
    silently skipped.
    """
    refs = _zs_test_refs()
    zs_seed_map = _zs_test_by_seed()
    if verbose:
        print("=" * 96)
        print("INTERVENTION MATRIX -- B-D on validation and on held-out windows")
        print("B-D > 0 means the FROZEN encoder is better (preservation-needed).")
        print("=" * 96)
        print(f"  zero-shot test references available: {len(refs)}")
    rows = []
    for key, seeds in sorted(moirai_cells().items(), key=str):
        rel, size, h, n = key
        ds = DATASET_OF.get(rel)
        if ds is None:
            continue
        paired = {s: v for s, v in seeds.items() if {"B", "D"} <= set(v)}
        if not paired:
            continue
        refkey = f"{size}_{ds}_h{h}"
        zs_t = refs.get(refkey)
        bv, bt, cka, drift = [], [], [], []
        for s, v in paired.items():
            zs_v = v["B"]["zeroshot_mse"]
            bv.append((v["B"]["final_val_mse"] - v["D"]["final_val_mse"]) / zs_v * 100)
            if zs_t:
                bt.append((v["B"]["test_mse"] - v["D"]["test_mse"]) / zs_t * 100)
            cka.append(v["B"]["final_cka"])
            drift.append(v["B"].get("final_weight_drift", float("nan")))
        pos = sum(x > 0 for x in bt) if bt else None
        fb = [(v["B"]["test_mse"] - zs_t) / zs_t * 100 for v in paired.values()] if zs_t else []
        fd = [(v["D"]["test_mse"] - zs_t) / zs_t * 100 for v in paired.values()] if zs_t else []
        zs_by_seed = list(zs_seed_map.get(refkey, {}).values())
        ps = _per_seed(paired.keys(), bt, fb, fd, ft_paired=False,
                       zs_seeds=len(zs_by_seed), zs_sem=_sem(zs_by_seed) / zs_t * 100 if zs_t else None)
        rows.append(dict(per_seed=ps, cell=f"Moirai-{size}/{ds} h{h} n{n}", seeds=len(paired), pos=pos,
                         bd_val=st.mean(bv), bd_val_sd=_sd(bv),
                         bd_test=st.mean(bt) if bt else None,
                         bd_test_sd=_sd(bt) if bt else None,
                         forg_b=st.mean(fb) if fb else None, forg_b_sd=_sd(fb),
                         forg_b_pos=sum(x > 0 for x in fb),
                         forg_d=st.mean(fd) if fd else None, forg_d_sd=_sd(fd),
                         forg_d_neg=sum(x < 0 for x in fd),
                         cka=st.mean(cka), drift=st.mean(drift), ref=refkey, has_ref=bool(zs_t)))
    rows += new_moirai_cells() + chronos_cells() + timesfm_cells() + ili_cell()

    gates, sf, lo = _gates(), strict_freeze_cells(), lora_value_cells()
    for r in rows:
        r["gate"] = gates.get(r["ref"])
        r["bh_test"] = sf.get(r["ref"], {}).get("bh_test")
        r["bh_test_sd"] = sf.get(r["ref"], {}).get("bh_test_sd")
        r["bh_seeds"] = sf.get(r["ref"], {}).get("seeds")
        r["cka_h"] = sf.get(r["ref"], {}).get("cka_h")
        for k in ("bd_same", "bd_same_sd", "forg_h", "forg_h_sd", "forg_h_pos"):
            r[k] = sf.get(r["ref"], {}).get(k)
        for k in ("be_test", "be_test_sd", "be_seeds", "be_pos", "bd_same_e", "bd_same_e_sd",
                  "de_test", "de_test_sd", "de_pos",
                  "forg_e", "forg_e_sd", "forg_e_pos", "cka_e", "drift_e", "lora_rank"):
            r[k] = lo.get(r["ref"], {}).get(k)
        # Where both arms ran the same seeds, their two independently-computed B-D values must agree.
        # They are built from the same B and D records by the same formula, so a disagreement means one
        # arm paired against a different run dir -- the failure _b_arm_for()'s docstring records twice.
        e_seeds = lo.get(r["ref"], {}).get("e_seedset")
        if e_seeds is not None and r["bh_seeds"] == len(e_seeds) and r["bd_same"] is not None:
            assert abs(r["bd_same"] - r["bd_same_e"]) < 1e-9, (
                f"{r['ref']}: B-D is {r['bd_same']:.6f} via the strict-freeze arm but "
                f"{r['bd_same_e']:.6f} via the LoRA arm, on seed sets of equal size")

    partial = [r for r in rows if r["seeds"] < MIN_SEEDS]
    if partial and verbose:
        for r in partial:
            print(f"  EXCLUDED (only {r['seeds']} seed(s), needs {MIN_SEEDS}): {r['cell']}")
    return [r for r in rows if r["seeds"] >= MIN_SEEDS]


def main():
    import sys
    global GATE_SPLIT
    for a in sys.argv[1:]:
        if a.startswith("--gate-split="):
            GATE_SPLIT = a.split("=", 1)[1]
    assert GATE_SPLIT in ("test", "val"), f"--gate-split must be test|val, got {GATE_SPLIT!r}"
    if GATE_SPLIT == "test":
        print("\n*** GATE TAKEN FROM THE TEST SPLIT -- the RETROSPECTIVE variant, scored on the "
              "same windows as the outcome; the paper's primary gate is --gate-split=val ***")
    else:
        print(f"\n*** gate from the {GATE_SPLIT.upper()} (selection) split -- the paper's primary; "
              f"forgetting terms remain test-side ***")
    rows = build_rows(verbose=True)

    for r in sorted(rows, key=lambda r: (r["bd_test"] is None, r["bd_test"] or 0)):
        if r["bd_test"] is not None:
            sem = r["bd_test_sd"] / max(r["seeds"], 1) ** 0.5
            sig = "" if abs(r["bd_test"]) > 2 * sem else " [<2SEM]"
            t = (f"{r['bd_test']:+7.2f}±{sem:5.2f}SEM {r['pos']}/{r['seeds']}+{sig}")
        else:
            t = f"   no ref ({r['ref']})"
        g = f"gate={r['gate']:+.3f}{'' if (r['gate'] or 0) >= 0.20 else 'FAIL'}" \
            if r["gate"] is not None else "gate=  ?   "
        bh = ""
        if r["bh_test"] is not None:
            bh = (f"  B-H={r['bh_test']:+7.2f}±"
                  f"{r['bh_test_sd'] / max(r['bh_seeds'], 1) ** 0.5:.2f}SEM"
                  f" k={r['bh_seeds']} CKA_H={r['cka_h']:.3f}")
        tag = "PRESERVE" if (r["bd_test"] or 0) > 0 else "adapt"
        print(f"  {r['cell']:34s} k={r['seeds']:2d}  CKA={r['cka']:.3f}  l2={r['drift']:5.2f}  "
              f"{g:16s}  val={r['bd_val']:+7.2f}±{r['bd_val_sd']:5.2f}  test={t}  "
              f"{tag if r['bd_test'] is not None else ''}{bh}  {r.get('note','')}")
    missing = sorted({r["ref"] for r in rows if not r["has_ref"]})
    if missing:
        print(f"\n  missing zero-shot test references: {missing}")
    quantified = [r for r in rows if r["bd_test"] is not None]
    if len(quantified) >= 4:
        degradation_cells(quantified)
        cross_cell_stats(quantified)
        within_backbone(quantified)
        predicts_bd(quantified)
        strict_freeze_report(quantified)
    if "--latex" in sys.argv:
        emit_latex(quantified)
        emit_strictfreeze_latex(quantified)
        emit_lora_latex(quantified)


def degradation_cells(rows, gate_threshold=0.20):
    """
    Apply the paper's own definition of a degradation cell, uniformly, from the numbers.

    Definition (Table 1 caption): a GATE-PASSING cell where full fine-tuning ends up worse than the
    un-tuned model (forg_B > 0) and freezing improves on it (forg_D < 0). Hand-assembling this list
    is how the earlier revision came to report four cells while its own definition selected six --
    two Moirai-Small/ETTh1 cells were labelled `preserve` instead. This function is the list.

    The bare inequalities are not enough, and the cell that shows why is Moirai-Small/ETTh2 h192:
    gate 0.790, forg_D negative in 10/10 seeds, but forg_B is +2.0 with only 5/10 seeds positive --
    a coin flip that the mean happens to place on the harmful side. "Fine-tuning ends up worse than
    the un-tuned model" has to be a measurement, not a sign accident, so both terms must agree
    across EVERY seed. That rule is stated here rather than applied by eye; on these cells it
    coincides exactly with |mean| > 2 SEM, and it excludes only the ETTh2 h192 cell, which
    Table 1 already reports as directional.
    """
    print(f"\n{'='*96}\nDEGRADATION CELLS BY THE PAPER'S OWN DEFINITION")
    print(f"gate >= {gate_threshold} AND forg_B > 0 in every seed AND forg_D < 0 in every seed")
    hits, near = [], []
    for r in sorted(rows, key=lambda r: -(r["bd_test"] or 0)):
        if r.get("forg_b") is None or r["gate"] is None or r.get("forg_confounded"):
            continue
        if not (r["gate"] >= gate_threshold and r["forg_b"] > 0 and r["forg_d"] < 0):
            continue
        unan = r["forg_b_pos"] == r["seeds"] and r["forg_d_neg"] == r["seeds"]
        sem = r["forg_b_sd"] / max(r["seeds"], 1) ** 0.5
        (hits if unan else near).append(r)
        print(f"  {'' if unan else 'excluded: '}{r['cell']:34s} gate={r['gate']:+.3f}  "
              f"forg_B={r['forg_b']:+6.2f}±{sem:5.2f}SEM "
              f"({r['forg_b_pos']}/{r['seeds']} pos)  "
              f"forg_D={r['forg_d']:+6.2f}±{r['forg_d_sd'] / max(r['seeds'], 1) ** 0.5:5.2f}SEM "
              f"({r['forg_d_neg']}/{r['seeds']} neg)  B-D={r['bd_test']:+6.2f}  "
              f"|forg_B|/SEM={abs(r['forg_b']) / sem if sem else float('inf'):.1f}")
    if hits:
        h = sorted(r["forg_b"] for r in hits)
        print(f"  => {len(hits)} cells; harm range +{h[0]:.1f}% to +{h[-1]:.1f}% above zero-shot")
    if near:
        print(f"  => {len(near)} excluded for seed disagreement: "
              f"{', '.join(r['cell'] for r in near)}")
    return hits


def predicts_bd(rows, n_boot=10000, seed=0):
    """
    Can a cell's reading be predicted WITHOUT running the frozen control?

    The paper's "Open question" asserted this was open. It is not, within the Moirai arm: condition
    B's own held-out forgetting orders B-D almost perfectly, and that is a rule needing conditions A
    and B only -- no condition D, no CKA.

    The coupling is stated here rather than hidden, because it IS the finding: B-D == forg_B - forg_D
    exactly (both divide by the same zero-shot test reference), so forg_B predicts B-D to the extent
    that forg_D is constant across cells. This function prints the SD of each so the reader can see
    which term carries the cross-cell variation, and verifies the identity numerically rather than
    asserting it.

    Chronos is reported separately and never pooled: its per-condition forgetting is confounded by
    the head/decoder mismatch (limitation iv), so its forg_B is not a forgetting measurement. It is
    exactly where the rule fails, which is the scope limit, not a footnote.

    INTERVALS ARE CLUSTERED AND P-VALUES ARE GONE. Within a group these cells still reuse series
    across sizes and horizons, so the interval is taken over (backbone, dataset) clusters as well as
    over cells; both are printed. The gate-vs-B-D correlation reported in the body comes from here,
    and it previously carried a cell-level p-value, which over-states the evidence because it assumes
    31 independent draws where there are roughly 10 independent series-backbone combinations.
    """
    import numpy as np

    import cluster_keys as ck
    rng = np.random.default_rng(seed)

    print(f"\n{'='*96}\nCAN THE READING BE PREDICTED WITHOUT CONDITION D?")
    groups = [("Moirai only (forg_B is a clean measurement)",
               [r for r in rows if r["cell"].startswith("Moirai") and r.get("forg_b") is not None]),
              # TimesFM attaches no head, so forg_B is a clean measurement here too -- this is the
              # second, independent test of the rule, on a backbone the rule was not derived on.
              ("TimesFM only (forg_B is a clean measurement)",
               [r for r in rows if r["cell"].startswith("TimesFM") and r.get("forg_b") is not None]),
              ("Chronos only (forg_B confounded, limitation iv)",
               [r for r in rows if r["cell"].startswith("Chronos") and r.get("forg_b") is not None])]
    for name, g in groups:
        if len(g) < 4:
            print(f"  {name:44s} n={len(g):2d}  too few cells")
            continue
        y = np.array([r["bd_test"] for r in g])
        cl = ck.clusters_for([r["cell"] for r in g])
        print(f"  {name}  n={len(g)}  clusters={len(set(cl))}")
        for key, label in (("forg_b", "forg_B (cond. B only)"), ("forg_d", "forg_D"),
                           ("gate", "gate R2_task"), ("cka", "CKA"), ("drift", "l2 drift")):
            x = np.array([r[key] if r[key] is not None else np.nan for r in g], dtype=float)
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 4:
                print(f"    {label:24s} n={ok.sum()}  too few")
                continue
            rho, lo, hi, _ = ck.cell_bootstrap_spearman(
                x[ok], y[ok], n_boot=n_boot, rng=np.random.default_rng(seed))
            _, clo, chi, ckept, ncl = ck.cluster_bootstrap_spearman(
                x[ok], y[ok], cl[ok], n_boot=n_boot, rng=rng)
            excl = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
            cexcl = "EXCLUDES 0" if (clo > 0 or chi < 0) else "includes 0"
            print(f"    {label:24s} n={ok.sum():2d}  rho={rho:+.3f}  "
                  f"cells CI[{lo:+.3f},{hi:+.3f}] {excl:10s}  "
                  f"clusters CI[{clo:+.3f},{chi:+.3f}] {cexcl:10s} "
                  f"({ncl} cl, kept {ckept}/{n_boot})")
        fb = np.array([r["forg_b"] for r in g]); fd = np.array([r["forg_d"] for r in g])
        dev = float(np.max(np.abs((fb - fd) - y)))
        print(f"    identity check: max|(forg_B - forg_D) - (B-D)| = {dev:.3f}  "
              f"(SD forg_B {fb.std(ddof=1):.1f} vs SD forg_D {fd.std(ddof=1):.1f}: "
              "the variation is on the B side)")


def strict_freeze_report(rows):
    """
    B-H beside B-D on every cell where the strict-freeze control ran.

    Reading rule, fixed before the runs (scripts/run_strict_freeze.sh): agreement in sign means the
    "freezing wins" reading is not an artifact of in_proj/mask_encoding re-fitting under condition D;
    divergence on a cell means part of that cell's gap is input re-fitting and is reported as such.
    """
    have = [r for r in rows if r.get("bh_test") is not None]
    print(f"\n{'='*96}\nSTRICT-FREEZE CONTROL (condition H: in_proj + mask_encoding frozen too)")
    if not have:
        print("  no condition-H runs yet (results/v45_strict_freeze)")
        return
    print(f"  {'cell':34s} kH {'B-D(same seeds)':>16s} {'B-H':>16s} {'forg_H':>16s}  CKA_H  agrees?")
    for r in sorted(have, key=lambda r: -(r["bd_same"] or 0)):
        k = r["bh_seeds"]
        sem = lambda v: v / max(k, 1) ** 0.5
        agree = "yes" if (r["bh_test"] > 0) == (r["bd_same"] > 0) else "NO -- SIGN FLIP"
        # forg_H > 0 means the strictly-frozen model does NOT beat the un-tuned one, so that cell's
        # "freezing improves on zero-shot" clause was carried by the input projections re-fitting.
        flag = "" if r["forg_h"] < 0 else f"   forg_H>0 in {r['forg_h_pos']}/{k} seeds"
        # The console shows partial arms on purpose -- it is how a run in progress gets watched -- but
        # it says they are partial, because the table excludes them.
        if (k or 0) < MIN_SEEDS:
            flag += f"   [PARTIAL: {k}/{MIN_SEEDS} seeds, excluded from the table]"
        print(f"  {r['cell']:34s} {k:2d} {r['bd_same']:+8.2f}±{sem(r['bd_same_sd']):5.2f} "
              f"{r['bh_test']:+8.2f}±{sem(r['bh_test_sd']):5.2f} "
              f"{r['forg_h']:+8.2f}±{sem(r['forg_h_sd']):5.2f}  {r['cka_h']:.4f}  {agree}{flag}")


def within_backbone(rows):
    """
    The pooled CKA-vs-B-D correlation is not a licensed analysis.

    Section 2 of the paper states that CKA is "compared only within a backbone, never across
    architectures" -- and for good reason: CKA magnitude is representation-dependent, so values
    from different architectures are not on a common scale. Every Chronos cell here sits at CKA
    0.09-0.23 and every Moirai cell at 0.40-0.97, so pooling them can manufacture a correlation
    out of the backbone split alone. This function reports the pooled figure and the
    within-backbone figures side by side so the difference is visible rather than assumed.
    """
    import numpy as np

    import cluster_keys as ck

    groups = [("POOLED (violates the within-backbone rule of §2)", rows),
              ("Moirai only", [r for r in rows if r["cell"].startswith("Moirai")]),
              ("Chronos only", [r for r in rows if r["cell"].startswith("Chronos")]),
              ("TimesFM only", [r for r in rows if r["cell"].startswith("TimesFM")])]
    print(f"\n{'='*96}\nIS THE CKA CORRELATION A BACKBONE ARTIFACT?")
    print("Intervals over (backbone, dataset) clusters; descriptive, no p-values.")
    for name, g in groups:
        if len(g) < 4:
            print(f"  {name:48s} n={len(g):2d}  too few cells"); continue
        x = np.array([r["cka"] for r in g]); y = np.array([r["bd_test"] for r in g])
        cl = ck.clusters_for([r["cell"] for r in g])
        rho, lo, hi, _ = ck.cell_bootstrap_spearman(x, y, rng=np.random.default_rng(0))
        _, clo, chi, kept, ncl = ck.cluster_bootstrap_spearman(
            x, y, cl, rng=np.random.default_rng(0))
        verdict = "EXCLUDES 0" if (clo > 0 or chi < 0) else "includes 0"
        print(f"  {name:48s} n={len(g):2d}  CKA {min(x):.2f}-{max(x):.2f}  rho={rho:+.3f}  "
              f"cells CI[{lo:+.3f},{hi:+.3f}]  clusters CI[{clo:+.3f},{chi:+.3f}] {verdict}  "
              f"({ncl} cl, kept {kept}/10000)")


def cross_cell_stats(rows, n_boot=10000, seed=0):
    """
    Do the observational diagnostics order the intervention?

    Two predictors, not one. CKA is the measure the paper is about; l2 weight drift is a second,
    entirely different observational summary of the same fine-tuning. If BOTH fail to order B-D,
    the claim is about observational measures in general rather than about CKA specifically -- a
    materially stronger and more useful statement. If l2 succeeds where CKA fails, the honest
    conclusion is the narrower one, that CKA is the wrong summary. Reported either way.

    TWO INTERVALS, NOT ONE. The cell-level bootstrap resamples cells as if they were independent,
    which they are not: these cells reuse 3 backbones and 7 series, so runs sharing a series share
    its split, its normalisation constants and its checkpoint. The clustered interval resamples
    (backbone, dataset) pairs instead, which is the level the dependence lives at. Both are printed
    because the gap between them is the evidence that the cell-level one was too narrow. No p-value
    is printed for either: a p-value from a resampling scheme that mis-states the dependence is not
    conventional inferential evidence, and the paper's own framing of this analysis is descriptive.
    """
    import numpy as np
    from scipy import stats  # noqa: F401  (used via cluster_keys)

    import cluster_keys as ck
    rng = np.random.default_rng(seed)
    y = np.array([r["bd_test"] for r in rows])
    cl = ck.clusters_for([r["cell"] for r in rows])
    print(f"\n{'='*96}\nDO OBSERVATIONAL DIAGNOSTICS ORDER THE INTERVENTION?  (n={len(rows)} cells)")
    print("Spearman rho of predictor vs held-out B-D. Descriptive: no p-values, two intervals --")
    print("over cells (too narrow, shown for contrast) and over (backbone, dataset) clusters.")
    for name in ("cka", "drift"):
        x = np.array([r[name] for r in rows])
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 4:
            continue
        rho, lo, hi, kept = ck.cell_bootstrap_spearman(
            x[ok], y[ok], n_boot=n_boot, rng=np.random.default_rng(seed))
        crho, clo, chi, ckept, ncl = ck.cluster_bootstrap_spearman(
            x[ok], y[ok], cl[ok], n_boot=n_boot, rng=rng)
        assert abs(crho - rho) < 1e-12, "the two bootstraps must share one point estimate"
        label = "CKA" if name == "cka" else "l2 weight drift"
        print(f"  {label:16s} rho={rho:+.3f}")
        print(f"    {'over cells':22s} 95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"{'includes 0' if lo <= 0 <= hi else 'EXCLUDES 0'}  (kept {kept}/{n_boot})")
        print(f"    {'over clusters':22s} 95% CI [{clo:+.3f}, {chi:+.3f}]  "
              f"{'includes 0' if clo <= 0 <= chi else 'EXCLUDES 0'}  "
              f"(kept {ckept}/{n_boot}, {ncl} clusters)")
    print("  (a CI that includes 0 means these cells cannot pin the sign down -- not that the\n"
          "   diagnostic is uninformative, and not a population claim)")


if __name__ == "__main__":
    main()
