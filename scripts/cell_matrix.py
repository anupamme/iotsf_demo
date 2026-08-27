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


def _sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


GATE_SPLIT = "test"          # set from --gate-split; see _gates()


def _gates():
    """cell-ref -> gate R2_task, from scripts/gate_all_cells.py.

    GATE_SPLIT="test" is the gate as reported: recomputed on the same held-out windows B-D is
    scored on. That makes cell INCLUSION a retrospective judgement -- the criterion is applied
    after the test split has been looked at. GATE_SPLIT="val" answers the counterfactual a
    reviewer is entitled to ask: which cells would a gate fixed BEFORE any test window was
    consulted have admitted? Forgetting terms stay test-side either way, so only the inclusion
    decision changes.
    """
    p = ROOT / f"results/gate_{GATE_SPLIT}_side.json"
    if not p.exists():
        return {}
    return {k: v["r2_task"] for k, v in json.load(open(p)).items()}


def _b_arm_for(name, refs):
    """
    (seed -> condition-B run, zero-shot test MSE) for a strict-freeze cell.

    The six degradation cells do not all live in one results dir: Weather and Moirai-Base/ETTh1 were
    run for this revision under results/v43_moirai_matrix (each with its own condition A), while
    Moirai-Small/ETTh1 comes from the earlier results/v5_etth1 sweep and takes its denominator from
    the shared zero-shot test references. Pairing only against v43 silently dropped the two
    Small/ETTh1 cells from the strict-freeze report even though condition H had run on them, which is
    exactly the kind of quiet omission this table exists to prevent.
    """
    B = {json.load(open(f))["seed"]: json.load(open(f))
         for f in glob.glob(str(ROOT / "results/v43_moirai_matrix" / name / "condition_B/*.json"))}
    A = [json.load(open(f)) for f in
         glob.glob(str(ROOT / "results/v43_moirai_matrix" / name / "condition_A/*.json"))]
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
         for f in glob.glob(str(ROOT / "results/v43_moirai_matrix" / name / "condition_D/*.json"))}
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
    for d in sorted(glob.glob(str(ROOT / "results/v45_strict_freeze/*"))):
        name = Path(d).name
        if name == "ili":
            continue
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


def _zs_test_refs():
    """dataset-key -> zero-shot test MSE, averaged over the seeds we measured it on."""
    refs = defaultdict(list)
    for f in glob.glob(str(ROOT / "results/v41_zs_test/*/condition_A_*.json")):
        key = Path(f).parent.name              # e.g. small_ETTh2_h96
        d = json.load(open(f))
        if "zeroshot_test_mse" in d:
            refs[key].append(d["zeroshot_test_mse"])
    for pat in ("results/v43_moirai_matrix/*/condition_A/*.json",
                "results/v47_prospective/*/condition_A/*.json"):
        for f in glob.glob(str(ROOT / pat)):
            key = Path(f).parent.parent.name
            d = json.load(open(f))
            if "zeroshot_test_mse" in d:
                refs[key].append(d["zeroshot_test_mse"])
    # the Moirai-Base n=1k cell measured earlier, under its own directory
    for f in glob.glob(str(ROOT / "results/v39_moirai_zs_test/h96/condition_A/*.json")):
        d = json.load(open(f))
        if "zeroshot_test_mse" in d:
            refs["base_ETTh2_h96"].append(d["zeroshot_test_mse"])
    return {k: st.mean(v) for k, v in refs.items()}


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


def new_moirai_cells():
    """Moirai cells that carry their own condition A.

    Two directories share one layout: results/v43_moirai_matrix (the cells added for the previous
    revision) and results/v47_prospective (the pre-registered prospective arm). They are read by
    the same code so a prospective cell is never scored differently from a published one.
    """
    out = []
    for d in sorted(glob.glob(str(ROOT / "results/v43_moirai_matrix/*"))
                    + glob.glob(str(ROOT / "results/v47_prospective/*"))):
        if not Path(d).is_dir():
            continue
        name = Path(d).name                       # e.g. small_Weather_h96
        A = [json.load(open(f)) for f in glob.glob(d + "/condition_A/*.json")]
        A = [x for x in A if "zeroshot_test_mse" in x]
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
        out.append(dict(cell=f"Moirai-{name}", seeds=len(seeds), pos=sum(x > 0 for x in bt),
                        bd_val=st.mean(bv), bd_val_sd=_sd(bv),
                        bd_test=st.mean(bt), bd_test_sd=_sd(bt),
                        forg_b=st.mean(fb), forg_b_sd=_sd(fb), forg_b_pos=sum(x > 0 for x in fb),
                        forg_d=st.mean(fd), forg_d_sd=_sd(fd), forg_d_neg=sum(x < 0 for x in fd),
                        cka=st.mean(B[s]["final_cka"] for s in seeds),
                        drift=st.mean(B[s].get("final_weight_drift", float("nan")) for s in seeds),
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
        out.append(dict(cell=f"Chronos/{ds} h{horizon}", seeds=len(seeds),
                        pos=sum(x > 0 for x in bt), bd_val=st.mean(bv), bd_val_sd=_sd(bv),
                        bd_test=st.mean(bt), bd_test_sd=_sd(bt), cka=st.mean(cka),
                        forg_b=st.mean(fb), forg_b_sd=_sd(fb), forg_b_pos=sum(x > 0 for x in fb),
                        forg_d=st.mean(fd), forg_d_sd=_sd(fd), forg_d_neg=sum(x < 0 for x in fd),
                        forg_confounded=True,
                        drift=float("nan"), ref=f"chronos_{ds}", has_ref=True,
                        note=f"{bad}/{len(seeds)} seeds fell back to AdamW" if bad else ""))
    return out


def timesfm_cells(root="results/v46_timesfm", horizon=24):
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
        out.append(dict(cell=f"TimesFM/{ds} h{horizon}", seeds=len(seeds),
                        pos=sum(x > 0 for x in bt), bd_val=st.mean(bv), bd_val_sd=_sd(bv),
                        bd_test=st.mean(bt), bd_test_sd=_sd(bt), cka=st.mean(cka),
                        forg_b=st.mean(fb), forg_b_sd=_sd(fb), forg_b_pos=sum(x > 0 for x in fb),
                        forg_d=st.mean(fd), forg_d_sd=_sd(fd), forg_d_neg=sum(x < 0 for x in fd),
                        drift=st.mean(drift), ref=f"timesfm_{ds.lower()}", has_ref=True))
    return out


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
    return [dict(cell="Moirai-small/ILI h24", seeds=len(seeds), pos=sum(x > 0 for x in bt),
                 bd_val=st.mean(bv), bd_val_sd=_sd(bv), bd_test=st.mean(bt), bd_test_sd=_sd(bt),
                 forg_b=st.mean(fb), forg_b_sd=_sd(fb), forg_b_pos=sum(x > 0 for x in fb),
                 forg_d=st.mean(fd), forg_d_sd=_sd(fd), forg_d_neg=sum(x < 0 for x in fd),
                 cka=st.mean(B[s]["aggregate"]["cka"] for s in seeds),
                 drift=float("nan"), ref="ili", has_ref=True)]


def _display(cell):
    """'Moirai-small/ETTh1 h96 n1000' -> 'Moirai-S / ETTh1 ($h{=}96$)', for the LaTeX table."""
    short = {"small": "S", "base": "B", "large": "L"}
    pretty = {"etth1": "ETTh1", "etth2": "ETTh2", "ettm2": "ETTm2",
              "weather": "Weather", "electricity": "Electricity"}
    if cell.startswith(("Chronos", "TimesFM")):
        arm, rest = cell.split("/")
        ds, h = rest.split()
        return f"{arm} / {pretty.get(ds.lower(), ds)} ($h{{=}}{h[1:]}$)"
    if "ILI" in cell:
        return "Moirai-S / ILI ($h{=}24$)"
    c = cell.replace("Moirai-", "").replace("_", "/", 1).replace("_h", " h")
    parts = c.replace("/", " ").split()
    size = short.get(parts[0], parts[0])
    ds, h = parts[1], parts[2][1:]
    n = f", $n{{=}}${parts[3][1:]}" if len(parts) > 3 and parts[3].startswith("n") else ""
    return f"Moirai-{size} / {ds} ($h{{=}}{h}${n})"


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
        "Cell & Seeds & Gate $R^2_\\text{task}$ & B$-$D validation & B$-$D held-out & Reverses? \\\\",
        "\\midrule",
    ]
    for r in sorted(rows, key=lambda r: r["bd_test"]):
        rev = (r["bd_val"] > 0) != (r["bd_test"] > 0)
        gate = "---" if r["gate"] is None else (
            f"${r['gate']:+.3f}$" + ("\\,\\textsuperscript{f}" if r["gate"] < 0.20 else ""))
        lines.append(
            f"{_display(r['cell'])} & {r['seeds']} & {gate} & "
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
    if not have:
        print("\n  no condition-H runs yet; skipped strictfreeze.tex")
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
            f"{_display(r['cell'])} & {k} & {fmt(r['bd_same'], r['bd_same_sd'], k)} & "
            f"{fmt(r['bh_test'], r['bh_test_sd'], k, bold=True)} & "
            f"{fmt(r['forg_h'], r['forg_h_sd'], k)} & ${r['cka_h']:.4f}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path.relative_to(ROOT)}  ({len(have)} cells)")


def build_rows(verbose=False):
    """Every intervention cell as a dict, with gate and strict-freeze fields attached.

    Split out of main() so figure scripts consume the same rows the tables and statistics do --
    fig1_diagnostic_flow.py previously hard-coded its bar values, which is how an earlier version
    came to plot 15 bars that silently omitted a cell.
    """
    refs = _zs_test_refs()
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
        rows.append(dict(cell=f"Moirai-{size}/{ds} h{h} n{n}", seeds=len(paired), pos=pos,
                         bd_val=st.mean(bv), bd_val_sd=_sd(bv),
                         bd_test=st.mean(bt) if bt else None,
                         bd_test_sd=_sd(bt) if bt else None,
                         forg_b=st.mean(fb) if fb else None, forg_b_sd=_sd(fb),
                         forg_b_pos=sum(x > 0 for x in fb),
                         forg_d=st.mean(fd) if fd else None, forg_d_sd=_sd(fd),
                         forg_d_neg=sum(x < 0 for x in fd),
                         cka=st.mean(cka), drift=st.mean(drift), ref=refkey, has_ref=bool(zs_t)))
    rows += new_moirai_cells() + chronos_cells() + timesfm_cells() + ili_cell()

    gates, sf = _gates(), strict_freeze_cells()
    for r in rows:
        r["gate"] = gates.get(r["ref"])
        r["bh_test"] = sf.get(r["ref"], {}).get("bh_test")
        r["bh_test_sd"] = sf.get(r["ref"], {}).get("bh_test_sd")
        r["bh_seeds"] = sf.get(r["ref"], {}).get("seeds")
        r["cka_h"] = sf.get(r["ref"], {}).get("cka_h")
        for k in ("bd_same", "bd_same_sd", "forg_h", "forg_h_sd", "forg_h_pos"):
            r[k] = sf.get(r["ref"], {}).get(k)
    return rows


def main():
    import sys
    global GATE_SPLIT
    for a in sys.argv[1:]:
        if a.startswith("--gate-split="):
            GATE_SPLIT = a.split("=", 1)[1]
    assert GATE_SPLIT in ("test", "val"), f"--gate-split must be test|val, got {GATE_SPLIT!r}"
    if GATE_SPLIT != "test":
        print(f"\n*** GATE TAKEN FROM THE {GATE_SPLIT.upper()} SPLIT -- prospective-inclusion "
              f"counterfactual; forgetting terms remain test-side ***")
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
    """
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(seed)

    def boot(x, y):
        rho, p = stats.spearmanr(x, y)
        b = []
        for _ in range(n_boot):
            i = rng.integers(0, len(x), len(x))
            if len(set(x[i])) > 2:
                b.append(stats.spearmanr(x[i], y[i]).statistic)
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        return rho, p, lo, hi

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
        print(f"  {name}  n={len(g)}")
        for key, label in (("forg_b", "forg_B (cond. B only)"), ("forg_d", "forg_D"),
                           ("gate", "gate R2_task"), ("cka", "CKA"), ("drift", "l2 drift")):
            x = np.array([r[key] if r[key] is not None else np.nan for r in g], dtype=float)
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 4:
                print(f"    {label:24s} n={ok.sum()}  too few")
                continue
            rho, p, lo, hi = boot(x[ok], y[ok])
            excl = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
            print(f"    {label:24s} n={ok.sum():2d}  rho={rho:+.3f}  p={p:.2e}  "
                  f"CI[{lo:+.3f},{hi:+.3f}]  {excl}")
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
    from scipy import stats
    rng = np.random.default_rng(0)

    def boot(x, y):
        rho, p = stats.spearmanr(x, y)
        b = []
        for _ in range(10000):
            i = rng.integers(0, len(x), len(x))
            if len(set(x[i])) > 2:
                b.append(stats.spearmanr(x[i], y[i]).statistic)
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        return rho, p, lo, hi

    groups = [("POOLED (violates the within-backbone rule of §2)", rows),
              ("Moirai only", [r for r in rows if r["cell"].startswith("Moirai")]),
              ("Chronos only", [r for r in rows if r["cell"].startswith("Chronos")]),
              ("TimesFM only", [r for r in rows if r["cell"].startswith("TimesFM")])]
    print(f"\n{'='*96}\nIS THE CKA CORRELATION A BACKBONE ARTIFACT?")
    for name, g in groups:
        if len(g) < 4:
            print(f"  {name:48s} n={len(g):2d}  too few cells"); continue
        x = np.array([r["cka"] for r in g]); y = np.array([r["bd_test"] for r in g])
        rho, p, lo, hi = boot(x, y)
        verdict = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
        print(f"  {name:48s} n={len(g):2d}  CKA {min(x):.2f}-{max(x):.2f}  "
              f"rho={rho:+.3f} p={p:.4f} CI[{lo:+.3f},{hi:+.3f}]  {verdict}")


def cross_cell_stats(rows, n_boot=10000, seed=0):
    """
    Do the observational diagnostics order the intervention?

    Two predictors, not one. CKA is the measure the paper is about; l2 weight drift is a second,
    entirely different observational summary of the same fine-tuning. If BOTH fail to order B-D,
    the claim is about observational measures in general rather than about CKA specifically -- a
    materially stronger and more useful statement. If l2 succeeds where CKA fails, the honest
    conclusion is the narrower one, that CKA is the wrong summary. Reported either way.
    """
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(seed)
    y = np.array([r["bd_test"] for r in rows])
    print(f"\n{'='*96}\nDO OBSERVATIONAL DIAGNOSTICS ORDER THE INTERVENTION?  (n={len(rows)} cells)")
    print("Spearman rho of predictor vs held-out B-D, with a bootstrap CI over cells.")
    for name in ("cka", "drift"):
        x = np.array([r[name] for r in rows])
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 4:
            continue
        rho, p = stats.spearmanr(x[ok], y[ok])
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, ok.sum(), ok.sum())
            if len(set(x[ok][idx])) < 3:
                continue
            boots.append(stats.spearmanr(x[ok][idx], y[ok][idx]).statistic)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        crosses = "includes 0" if lo <= 0 <= hi else "EXCLUDES 0"
        label = "CKA" if name == "cka" else "l2 weight drift"
        print(f"  {label:16s} rho={rho:+.3f}  p={p:.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {crosses}")
    print("  (a CI that includes 0 means the diagnostic does not order the intervention here;\n"
          "   at these cell counts this is a statement about our cells, not a population claim)")


if __name__ == "__main__":
    main()
