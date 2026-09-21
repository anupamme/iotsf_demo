#!/usr/bin/env python3
"""
Emit the pre-registered positive control: results/positive_control.json and paper_8/tables/positive_control.tex.

WHAT THE ARM IS FOR. The main matrix contains no cell that loses a pre-trained capability, so every
statement it makes about CKA's ability to ORDER damage is made in a regime where there is little damage
to order. A reviewer's objection to that is correct and structural: a diagnostic cannot be shown to fail
at detection if the thing to detect is absent. This arm constructs the missing regime deliberately --
task A is the strongest cell the screen admits, task B is a synthetic series whose optimal forecast is
the NEGATION of the seasonal-naive prior -- and then asks whether CKA tracks what was lost.

NOTHING IN THIS FILE DECIDES ANYTHING. The grid, the thresholds, the primary checkpoint, where CKA is
measured, the validity conditions and all three outcome sentences were written to
results/positive_control/preregistration.json and committed before the first run existed. This script
reads that file, reads the run records, applies the registered rules, and reports which outcome fired.
Where an outcome sentence appears below it is quoted from the registration, not composed here.

WHY BOTH CHECKPOINTS ARE REPORTED. Early stopping is on task B's validation MSE, so it can halt before
the damage to task A has finished developing. The registration names the early-stopped state primary
(for comparability with the main matrix, which early-stops the same way) and the final-epoch state the
maximal-destruction bound. When early stopping never fires the two states are IDENTICAL, and then the
gap between their two retention numbers is not redundancy -- it is a direct read of the evaluator's own
sampling noise, since Moirai's forecast is sampled. That quantity is computed and reported, because
every retention number here has to be read against it.

A PARTIAL GRID IS REPORTED AS PARTIAL. Missing cells are listed by name in the JSON and in the table's
caption. Silence about a missing cell would let a reader take a half-run ladder for a finished one.

TIER C: the retention JSONs it reads come from checkpoints (*.pt, gitignored). Regenerate them with
scripts/run_positive_control.sh; this script itself needs only the JSONs, which are committed.

Usage:  .venv12/bin/python scripts/emit_positive_control.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PC = ROOT / "results/positive_control"
REG = PC / "preregistration.json"
OUT_JSON = ROOT / "results/positive_control.json"
OUT_TEX = ROOT / "paper_8/tables/positive_control.tex"

COND_NAME = {"B": "full fine-tune", "D": "frozen encoder"}


def cell_dir(lr, cond, seed):
    """The directory name run_positive_control.sh builds, which is the registration's own float repr.

    Kept in ONE place on purpose: an earlier version of the runner wrote lr1e-2 while the registration
    serialised 0.01, so the idempotent skip missed the pilot and would have produced a second record of
    the same registered cell.
    """
    return PC / f"lr{lr}_{cond}_s{seed}"


def read_cell(lr, cond, seed, horizon):
    d = cell_dir(lr, cond, seed)
    run = d / f"condition_{cond}_h{horizon}_s{seed}.json"
    if not run.exists():
        return None
    r = json.loads(run.read_text())
    row = dict(lr=lr, condition=cond, seed=seed, dir=str(d.relative_to(ROOT)),
               # forgetting_pct in the runner is measured on the series it TRAINED on, which here is
               # task B. Negated so the column reads as "task B learned by x%".
               taskb_learned_pct=-r["forgetting_pct"],
               taskb_mse_zeroshot=r["zeroshot_mse"], taskb_mse_finetuned=r["test_mse"],
               cka_on_task_b_inputs=r["final_cka"], weight_drift=r["final_weight_drift"],
               best_epoch=r["early_stopping"]["best_epoch"], epochs=r["epochs"],
               early_stopping_fired=r["early_stopping"]["best_epoch"] < r["epochs"])
    for ck in ("best_encoder", "final_state"):
        f = d / f"retention_A_{ck}.json"
        if not f.exists():
            row[f"retention_A_{ck}"] = None
            continue
        q = json.loads(f.read_text())
        row[f"retention_A_{ck}"] = q["retention_A_pct"]
        row[f"cka_on_task_a_{ck}"] = q["cka_on_task_a_inputs"]
        row[f"mse_A_zeroshot_{ck}"] = q["mse_zeroshot"]
        row[f"mse_A_finetuned_{ck}"] = q["mse_finetuned"]
        row[f"n_eval_windows_A"] = q["n_eval_windows"]
    return row


def diverged_cells():
    """Cells whose run FAILED, with the reason read out of the runner's own log.

    A cell with no record is ambiguous -- never launched, or launched and crashed -- and the two are not
    the same fact. The registered grid completed 18/18, but the declared lr=1e-1 extension's THREE
    frozen-encoder cells diverged numerically: with the encoder frozen every gradient lands on the
    decoder, and its likelihood head leaves the domain its distribution requires. That is a result about
    the rung, not an absence of one, and reporting it as "missing" would read as work not done.

    Derived, never hand-typed: the failing cells come from the status file the runner appends a verdict
    to, and the reason from the exception in that cell's log. If a future failure has a different cause
    this reports THAT cause, rather than a sentence about the negative binomial that has stopped
    being true.
    """
    tsv = PC / "grid_status.tsv"
    if not tsv.exists():
        return {}
    out = {}
    for line in tsv.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3 or not parts[2].startswith("FAILED"):
            continue
        cell, stage = parts[0], parts[1]
        log = ROOT / f"logs/pc_{cell}.log"
        reason = "the runner recorded a failure; its log is not present"
        if log.exists():
            # The LAST exception line, not the first: training catches and skips individual bad batches
            # (logged as warnings), and the run dies later on the uncaught one. The first match would be
            # a survived batch, which is the opposite of the fatal cause.
            hits = [ln.strip() for ln in log.read_text(errors="replace").splitlines()
                    if ln.startswith(("ValueError:", "RuntimeError:", "TypeError:", "AssertionError:",
                                      "KeyError:", "OSError:", "torch.OutOfMemoryError:"))]
            if hits:
                reason = hits[-1][:300]
        out[cell] = dict(stage=stage, status=parts[2], reason=reason)
    return out


def dispersion(xs):
    """std at both ddofs and the SEM, every time.

    This project has mixed conventions on record -- SEM in the body, std in two tables, ddof=0 in stored
    JSON -- so all three are stored and the consumer picks. A single number here would be the ambiguity
    that has already had to be chased down once.
    """
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if a.size == 0:
        return dict(n=0, mean=None, std_ddof0=None, std_ddof1=None, sem=None)
    return dict(n=int(a.size), mean=float(a.mean()),
                std_ddof0=float(a.std(ddof=0)),
                std_ddof1=float(a.std(ddof=1)) if a.size > 1 else None,
                sem=float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else None)


def main():
    if not REG.exists():
        sys.exit(f"no {REG.relative_to(ROOT)}: the arm may not be reported before it is registered")
    reg = json.loads(REG.read_text())
    g, thr = reg["grid"], reg["thresholds"]
    hz = reg["task_a"]["horizon"]
    primary_ck = "best_encoder"      # the registration's primary: the early-stopped state
    bound_ck = "final_state"         # the maximal-destruction bound

    # The registered grid, plus the one declared extension if it was run. The extension is NOT assumed:
    # it appears only if its directories exist, and the JSON records whether it was taken.
    planned = [(r["lr"], r["condition"], r["seed"]) for r in g["runs"]]
    ext_lr = g["declared_extension_lr"]
    ext = [(ext_lr, c, s) for c in g["conditions"] for s in g["seeds"]]
    ext_present = [k for k in ext if cell_dir(*k).exists()]

    # A cell with no record is separated into "diverged" (the runner recorded a failure, and the reason
    # is read from its log) and "missing" (no record and no recorded failure -- never launched). Both are
    # reported, but only the second makes the outcome provisional: a recorded divergence IS a result.
    failed = diverged_cells()
    rows, missing, diverged = [], [], {}
    for lr, cond, seed in planned + ext_present:
        name = f"lr{lr}_{cond}_s{seed}"
        r = read_cell(lr, cond, seed, hz)
        if r is None:
            if name in failed:
                diverged[name] = dict(failed[name], registered=(lr, cond, seed) in planned,
                                      lr=lr, condition=cond, seed=seed)
            else:
                missing.append(name)
            continue
        r["registered"] = (lr, cond, seed) in planned
        rows.append(r)
    incomplete = [r["dir"] for r in rows if r[f"retention_A_{primary_ck}"] is None
                  or r[f"retention_A_{bound_ck}"] is None]
    # Provisionality is about the REGISTERED grid only. The extension is optional by the stopping rule,
    # so an extension cell that diverged does not make the registered ladder's verdict partial -- and
    # calling it partial would be its own error, since it would mark a complete 18/18 as unfinished.
    reg_missing = [f"lr{lr}_{c}_s{s}" for (lr, c, s) in planned
                   if f"lr{lr}_{c}_s{s}" in missing]
    reg_diverged = [k for k, v in diverged.items() if v["registered"]]
    reg_unscored = [r["dir"] for r in rows if r["registered"] and (
        r[f"retention_A_{primary_ck}"] is None or r[f"retention_A_{bound_ck}"] is None)]
    n_reg_rows = sum(1 for r in rows if r["registered"])

    if not rows:
        sys.exit("no positive-control runs found; run scripts/run_positive_control.sh first")

    # ---- validity condition 1: was task B actually learned?
    learned = [r["taskb_learned_pct"] for r in rows]
    n_learned = sum(1 for x in learned if x >= thr["taskb_learned_pct"])
    taskb_ok = n_learned == len(rows)

    # ---- the destruction rule, exactly as registered: retention_A >= 50% in a MAJORITY of seeds at
    # some rung under condition B. Applied to the PRIMARY checkpoint, and separately to the bound.
    def destruction(ck):
        hit = {}
        for lr in sorted({r["lr"] for r in rows}):
            vals = [r[f"retention_A_{ck}"] for r in rows
                    if r["lr"] == lr and r["condition"] == "B" and r[f"retention_A_{ck}"] is not None]
            if not vals:
                continue
            n_over = sum(1 for v in vals if v >= thr["destruction_pct"])
            hit[lr] = dict(n_seeds=len(vals), n_over=n_over,
                           majority=n_over * 2 > len(vals), max=max(vals))
        return dict(per_rung=hit, achieved=any(h["majority"] for h in hit.values()),
                    rungs_achieving=[lr for lr, h in hit.items() if h["majority"]])

    destr_primary, destr_bound = destruction(primary_ck), destruction(bound_ck)

    # ---- the CKA ordering test, as registered: Spearman rho between CKA on task A's inputs and
    # retention_A, across all runs. Negative means lower similarity goes with more damage.
    def ordering(ck):
        xs = [(r[f"cka_on_task_a_{ck}"], r[f"retention_A_{ck}"]) for r in rows
              if r.get(f"cka_on_task_a_{ck}") is not None and r[f"retention_A_{ck}"] is not None]
        if len(xs) < 3:
            return dict(n=len(xs), rho=None, p=None, orders=None)
        c, v = zip(*xs)
        s = stats.spearmanr(c, v)
        return dict(n=len(xs), rho=float(s.statistic), p=float(s.pvalue),
                    orders=bool(s.statistic <= thr["cka_ordering_rho"]))

    ord_primary, ord_bound = ordering(primary_ck), ordering(bound_ck)

    # ---- validity condition 2: does the frozen encoder preserve task A? Registered as falsifiable.
    by_rung = {}
    for lr in sorted({r["lr"] for r in rows}):
        cell = {}
        for cond in ("B", "D"):
            sel = [r for r in rows if r["lr"] == lr and r["condition"] == cond]
            cell[cond] = dict(
                taskb_learned=dispersion([r["taskb_learned_pct"] for r in sel]),
                retention_primary=dispersion([r[f"retention_A_{primary_ck}"] for r in sel]),
                retention_bound=dispersion([r[f"retention_A_{bound_ck}"] for r in sel]),
                cka_task_a=dispersion([r.get(f"cka_on_task_a_{primary_ck}") for r in sel]),
                cka_task_b=dispersion([r["cka_on_task_b_inputs"] for r in sel]),
                drift=dispersion([r["weight_drift"] for r in sel]),
                n_runs=len(sel))
        b, d = cell["B"]["retention_primary"]["mean"], cell["D"]["retention_primary"]["mean"]
        # The estimand in the same shape as Eq. 1's Delta_enc: B minus D, both on 100/L_ZS scaling.
        cell["delta_enc_like"] = None if (b is None or d is None) else b - d
        by_rung[str(lr)] = cell

    # ---- which registered outcome fired. The mapping is the registration's, not this file's.
    if not taskb_ok:
        outcome, key = "validity_violated", None
        text = reg["validity_conditions"][0]["if_violated"]
    elif destr_primary["achieved"] or destr_bound["achieved"]:
        orders = (ord_primary["orders"] if destr_primary["achieved"] else ord_bound["orders"])
        key = "1_destruction_and_cka_orders_it" if orders else "2_destruction_but_cka_does_not_order_it"
        outcome = "1" if orders else "2"
        text = reg["outcomes_fixed_in_advance"][key]
    else:
        outcome, key = "3", "3_destruction_not_achieved"
        text = reg["outcomes_fixed_in_advance"][key]

    # Identical states must give identical retention; they do not, because Moirai's forecast is sampled.
    # This is the evaluator's noise floor and every number above has to be read against it.
    same_state = [r for r in rows if not r["early_stopping_fired"]
                  and r[f"retention_A_{primary_ck}"] is not None
                  and r[f"retention_A_{bound_ck}"] is not None]
    noise = [abs(r[f"retention_A_{primary_ck}"] - r[f"retention_A_{bound_ck}"]) for r in same_state]

    out = dict(
        registration=str(REG.relative_to(ROOT)), registration_git_head=reg["git_head"],
        task_a=reg["task_a"], task_b=reg["task_b"],
        primary_checkpoint=primary_ck, bound_checkpoint=bound_ck,
        n_registered_runs=len(planned), n_runs_found=len(rows),
        n_registered_runs_found=n_reg_rows,
        extension_taken=bool(ext_present), extension_lr=ext_lr,
        extension_runs_found=len(rows) - n_reg_rows, extension_runs_planned=len(ext),
        missing_cells=missing, cells_without_retention=incomplete,
        diverged_cells=diverged,
        rows=rows, by_rung=by_rung,
        validity_taskb_learned=dict(rule=thr["taskb_learned_rule"], threshold=thr["taskb_learned_pct"],
                                    n_meeting=n_learned, n_runs=len(rows), all_meet=taskb_ok,
                                    min_pct=min(learned), max_pct=max(learned)),
        destruction=dict(rule=thr["destruction_rule"], threshold_pct=thr["destruction_pct"],
                         primary=destr_primary, bound=destr_bound),
        cka_ordering=dict(rule=reg["cka"]["ordering_test"], threshold=thr["cka_ordering_rho"],
                          primary=ord_primary, bound=ord_bound),
        evaluator_noise_pp=dict(
            n_identical_state_pairs=len(noise),
            max=max(noise) if noise else None,
            mean=float(np.mean(noise)) if noise else None,
            why="the two checkpoints are the same state whenever early stopping did not fire, so any "
                "difference between their retention numbers is the sampled forecast's own noise"),
        outcome=outcome, outcome_key=key, outcome_text_as_registered=text,
        # A one-cell grid can satisfy "destruction not achieved" trivially. The outcome is therefore
        # marked provisional until every registered cell has a retention score, so that a reader (or a
        # later script) cannot lift `outcome` out of this file and treat a partial ladder as a verdict.
        outcome_provisional=bool(reg_missing or reg_unscored or reg_diverged),
        what_this_cannot_show=reg["what_this_cannot_show"],
    )
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ------------------------------------------------------------------ LaTeX
    def f(x, p=1, sign=True):
        if x is None:
            return "--"
        s = f"{x:+.{p}f}" if sign else f"{x:.{p}f}"
        return s.replace("-", "$-$").replace("+", "$+$")

    def pm(dd, p=1, sign=True):
        if dd["mean"] is None:
            return "--"
        if dd["sem"] is None:
            return f(dd["mean"], p, sign)
        return f(dd['mean'], p, sign) + "{\\scriptsize$\\pm$" + f"{dd['sem']:.{p}f}" + "}"

    L = []
    A = L.append
    A("% GENERATED by scripts/emit_positive_control.py -- do not edit by hand.")
    A("% Every threshold, the grid and the outcome sentence come from")
    A("% results/positive_control/preregistration.json, committed before the first run.")
    # A loud, in-PDF marker rather than a comment: a partial ladder that LOOKS finished is the one
    # failure this table can cause, and it disappears by itself when the last cell lands.
    # Plain text, NOT \multicolumn: this file is \input inside a table environment but OUTSIDE the
    # tabular, and \multicolumn there is a LaTeX error rather than a marker anybody would see.
    # Registered and extension runs are counted against their OWN denominators. The first version printed
    # len(rows) over len(planned), so once the extension ran it read "21 of 18 registered runs present" --
    # a fraction over one, in bold, in the PDF, on the one table whose job is to say how much of the
    # ladder exists.
    if reg_missing or reg_unscored or reg_diverged:
        A(r"\textbf{PARTIAL GRID: " +
          f"{n_reg_rows} of {len(planned)} registered runs present" +
          (f", {len(reg_unscored)} unscored" if reg_unscored else "") +
          r". The outcome is provisional.}\\[3pt]")
    if diverged:
        # Stated in the table, not only in prose: a reader comparing the rows will notice a rung with one
        # condition and no other, and the reason has to be where the gap is.
        lrs = sorted({v["lr"] for v in diverged.values()})
        conds = sorted({COND_NAME[v["condition"]] for v in diverged.values()})
        A(r"\textbf{" + f"Diverged, not omitted: {len(diverged)} run(s)" +
          f" at lr~$={'$, $='.join(f'{x:g}' for x in lrs)}$, {', '.join(conds)}" +
          r"} did not complete; training diverged and the run is reported as a divergence "
          r"rather than a number (see text).\\[3pt]")
    # Seven columns overran \linewidth by 6.5pt at the default 6pt tabcolsep, and an Overfull \hbox is
    # a gate this paper checks. Narrowing the inter-column padding rather than shrinking the font keeps
    # the dispersions legible; the group scopes it to this table so no later table inherits it.
    A(r"{\setlength{\tabcolsep}{4pt}")
    A(r"\begin{tabular}{@{}llrrrrr@{}}")
    A(r"\toprule")
    A(r"& & task B & \multicolumn{2}{c}{retention on task A (\%)} & CKA on & $\ell_2$ \\")
    A(r"\cmidrule(lr){4-5}")
    A(r"lr & encoder & learned (\%) & early-stopped & final epoch & task A & drift \\")
    A(r"\midrule")
    for lr in sorted({r["lr"] for r in rows}):
        for cond in ("B", "D"):
            c = by_rung[str(lr)][cond]
            if c["n_runs"] == 0:
                continue
            tag = "" if any(r["lr"] == lr and r["registered"] for r in rows) else r"$^{\dagger}$"
            A(f"${lr:g}${tag} & {COND_NAME[cond]} & {pm(c['taskb_learned'], 1)} & "
              f"{pm(c['retention_primary'], 1)} & {pm(c['retention_bound'], 1)} & "
              f"{pm(c['cka_task_a'], 3, sign=False)} & {pm(c['drift'], 2, sign=False)} \\\\")
        A(r"\addlinespace")
    if L[-1] == r"\addlinespace":
        L.pop()
    A(r"\bottomrule")
    A(r"\end{tabular}}")
    OUT_TEX.write_text("\n".join(L) + "\n")

    # ------------------------------------------------------------------ stdout
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_TEX.relative_to(ROOT)}")
    print(f"  runs found {n_reg_rows}/{len(planned)} registered"
          f"{f' + {len(rows) - n_reg_rows}/{len(ext)} extension (lr={ext_lr:g})' if ext_present else ''}")
    if missing:
        print(f"  MISSING {len(missing)}: {', '.join(missing[:6])}"
              f"{' ...' if len(missing) > 6 else ''}")
    for name, v in sorted(diverged.items()):
        print(f"  DIVERGED {name} ({'registered' if v['registered'] else 'extension'}, {v['stage']}): "
              f"{v['reason'][:110]}")
    if incomplete:
        print(f"  NO RETENTION SCORE for {len(incomplete)}: {', '.join(incomplete[:4])}")
    print(f"  task B learned in {n_learned}/{len(rows)} runs "
          f"(threshold {thr['taskb_learned_pct']}%, min {min(learned):+.1f}%)")
    print(f"  destruction (primary): {destr_primary['achieved']}  "
          f"max retention_A under B = "
          f"{max((h['max'] for h in destr_primary['per_rung'].values()), default=float('nan')):+.1f}%")
    print(f"  destruction (bound):   {destr_bound['achieved']}")
    print(f"  CKA ordering (primary): rho={ord_primary['rho']}  n={ord_primary['n']}  "
          f"orders={ord_primary['orders']}")
    if noise:
        print(f"  evaluator noise floor: max {max(noise):.2f} pp over "
              f"{len(noise)} identical-state pairs")
    print(f"  REGISTERED OUTCOME {outcome}: {key}"
          + ("   [PROVISIONAL -- registered grid incomplete]"
             if (reg_missing or reg_unscored or reg_diverged) else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
