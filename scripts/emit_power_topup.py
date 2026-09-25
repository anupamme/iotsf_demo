#!/usr/bin/env python3
"""Phase F3: report the seed top-up as an AUGMENTATION, beside the published n, never instead of it.

WHAT THIS ANSWERS.  Five of the seven gate-passing cells are inconclusive on the paired encoder
contrast, and results/power_topup/preregistration_power.json fixed in advance which three get more
seeds, how many, and which seeds. The reader's question about any top-up is the same one: did the
extra seeds change the answer, or did they change the author's mind? The only way to settle it is to
put both columns in one table -- the published n and the augmented n, with the same estimator, the
same BH correction across the same 31 cells -- and to name the n at which any call changed.

WHY THE PUBLISHED COLUMN CANNOT MOVE.  cell_matrix.moirai_cells() globs results/**, so the added
records would have joined the matrix the moment they landed, shifting three cells' intervals and,
through BH across 31 cells, every other cell's q. cell_matrix.topup_paths() therefore excludes
exactly the 26 paths the registration lists from every published read, and the augmented read is an
explicit second computation. This file runs both, cross-checks the published one against the
committed results/paired_inference.json, and emits the comparison.

THE LADDER, AND WHAT "THE n AT WHICH IT CHANGED" MEANS.  For a topped-up cell we re-run the whole
analysis with that cell's added seeds admitted one at a time, in the order the registration lists
them, and report the smallest n whose call differs from the published one. The other two cells sit at
their FINAL seed count throughout, because BH couples the cells and a ladder that moved all three at
once would not isolate anything. That choice is stated in the table note rather than left implicit.
The ladder uses a small bootstrap because nothing on it is read off the bootstrap: the call comes
from the BH-adjusted paired t-test.

WHAT THE TWO UNRESOLVABLE CELLS GET.  Nothing, by registration: Moirai-Small/ETTh2 h=96 (needs 71
paired seeds) and Moirai-Base/ETTm2 h=192 (needs 83) were declared unresolvable before the top-up ran
and are not topped up however their p-values moved. They stay in the table with their required n as
the finding, because a cell that needs 71 seeds to resolve a 2.8 pp effect is telling the reader the
effect is small relative to the noise.

Run:  .venv12/bin/python scripts/emit_power_topup.py [--allow-incomplete] [--boot N] [--ladder-boot N]

Exits non-zero if any registered run is missing, so that once this is in rederive_all.sh a vanished
record is an error rather than a silently stale table.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import cell_matrix as cm                                          # noqa: E402
import paired_inference as pi                                     # noqa: E402

REG = ROOT / "results/power_topup/preregistration_power.json"
PUBLISHED = ROOT / "results/paired_inference.json"
OUT_JSON = ROOT / "results/power_topup/topup_comparison.json"
OUT_TEX = ROOT / "paper_8/tables/power_topup.tex"
TOL = 1e-9


def _by_ref(cells):
    return {c["ref"]: c for c in cells}


def _analyse(include_topup, boot, seed=0):
    rows = cm.build_rows(include_topup=include_topup)
    cells, _ = pi.analyse(rows, boot, seed)
    return _by_ref(cells)


def _paths_for(reg, ref):
    """The registered run paths of one cell, grouped by seed IN REGISTERED ORDER."""
    order, by_seed = [], {}
    for r in reg["planned_runs"]:
        if r["ref"] != ref:
            continue
        if r["seed"] not in by_seed:
            by_seed[r["seed"]] = []
            order.append(r["seed"])
        by_seed[r["seed"]].append(r["path"])
    return [(s, by_seed[s]) for s in order]


def _check_published(fresh, boot):
    """The published column must be the committed analysis, not a recomputation that drifted."""
    if not PUBLISHED.exists():
        sys.exit("results/paired_inference.json is missing; run scripts/paired_inference.py first")
    on_disk = _by_ref(json.loads(PUBLISHED.read_text())["cells"])
    if set(on_disk) != set(fresh):
        sys.exit(f"published cell set differs from the recomputation: "
                 f"{sorted(set(on_disk) ^ set(fresh))}")
    bad = []
    for ref, c in on_disk.items():
        for k in ("mean", "sd", "sem", "q", "n"):
            a, b = c["d_enc"][k], fresh[ref]["d_enc"][k]
            if a is None or b is None:
                if a is not b:
                    bad.append(f"{ref}.{k}: {a} vs {b}")
            elif abs(a - b) > TOL * max(1.0, abs(a)):
                bad.append(f"{ref}.{k}: {a} vs {b}")
        if c["call"] != fresh[ref]["call"]:
            bad.append(f"{ref}.call: {c['call']} vs {fresh[ref]['call']}")
    if bad:
        sys.exit("results/paired_inference.json is STALE against the current code/records "
                 f"({len(bad)} mismatches, e.g. {bad[:3]}); re-run scripts/paired_inference.py "
                 "and inspect what moved before trusting this table")
    # The bootstrap is reseeded here, so only the t-based quantities above are compared; say so
    # rather than implying the whole file was verified.
    return f"published column verified against results/paired_inference.json (n, mean, sd, sem, q, call)"


def ladder(reg, ref, published_call, all_topup, boot):
    """[(n, q, call)] as this cell's registered seeds are admitted one at a time.

    The other topped-up cells stay at their final seed count for every rung, so a change on this
    rung is attributable to this cell's seeds and not to BH moving underneath it.
    """
    mine = _paths_for(reg, ref)
    others = [p for p in all_topup
              if p not in {q for _, ps in mine for q in ps}]
    out, changed_at = [], None
    for j in range(len(mine) + 1):
        admitted = others + [p for _, ps in mine[:j] for p in ps]
        c = _analyse(admitted, boot)[ref]
        e = c["d_enc"]
        out.append(dict(added=j, n=e["n"], mean=e["mean"], sem=e["sem"], lo=e["lo"], hi=e["hi"],
                        q=e["q"], call=c["call"]))
        if changed_at is None and c["call"] != published_call:
            changed_at = e["n"]
    return out, changed_at


CALLS = ("freeze", "adapt", "inconclusive")
CALL_WORD = {"inconclusive": "inconclusive", "freeze": "freezing-decisive",
             "adapt": "adaptation-decisive"}


def caption(rows, changed, ladders, n_reg, tally):
    """The caption, generated. Every number in it is interpolated from the same rows the table prints.

    Written out rather than left to the appendix prose because the two claims a reader most needs
    beside this table -- that the published column did not move, and what changed and at which n --
    are claims about THESE numbers, and a hand-typed caption is how this project's tables drifted
    from their records before (see the comments in scripts/rederive_all.sh).
    """
    n_top = sum(r["augmented"] is not None for r in rows)
    L = [r"\caption{\textbf{The pre-registered seed top-up, beside the published seed counts.}",
         rf"The {len(rows)} cells clearing $\Vb{{\text{{ridge}}}}{{\geq}}0.20$.  {n_top} were topped"
         r" up under",
         r"\texttt{results/power\_topup/preregistration\_power.json}, which fixed which cells, which",
         rf"seeds and how many ({n_reg} runs) before any of those runs existed; all {n_reg} completed.",
         r"\textbf{No published number is replaced.}  Each cell's \emph{published} row is the",
         r"analysis of Table~\ref{tab:paired_inference}, re-verified against",
         r"\texttt{results/paired\_inference.json} at emit time; its \emph{augmented} row is a second",
         r"computation with the added seeds admitted, on the same estimator and the same",
         rf"Benjamini--Hochberg correction over the same {tally['m']} cells."]
    lab = {r["ref"]: r["label"] for r in rows}
    if changed:
        for ref, before, after, at in changed:
            rung = [g for g in ladders[ref] if g["n"] == at - 1]
            L += [rf"\textbf{{One call changes}}: {lab[ref]} moves from {CALL_WORD[before]} to",
                  rf"{CALL_WORD[after]}, first at $n{{=}}{at}$"
                  + (rf", with the rung below it at $q{{=}}{rung[0]['q']:.3f}$" if rung else "")
                  + r".",
                  rf"Across all {tally['m']} cells the augmented read changes that call and no other",
                  rf"(published {tally['pub']['freeze']} freezing-better,"
                  rf" {tally['pub']['adapt']} adaptation-better,"
                  rf" {tally['pub']['inconclusive']} inconclusive; augmented"
                  rf" {tally['aug']['freeze']}, {tally['aug']['adapt']},"
                  rf" {tally['aug']['inconclusive']}).",
                  ]
    else:
        L.append(rf"\textbf{{No cell's call changes}} under the augmented read.")
    exp = [r for r in rows if r.get("expected_to_stay_inconclusive") and r["augmented"]]
    if exp:
        held = [r for r in exp if r["augmented"]["call"] == "inconclusive"]
        fail = [r for r in exp if r not in held]
        L += [rf"The {len(exp)} cells on the \emph{{uncorrected}} tier were run to a budget below their",
              r"BH-adjusted requirement and were registered as \emph{expected to remain",
              rf"inconclusive}}, which held for {len(held)} of the {len(exp)}"
              + (rf" ({'; '.join(lab[r['ref']] for r in held)})" if held else "")
              + (rf" and failed for {', '.join(lab[r['ref']] for r in fail)}." if fail else "."),
              r"A failed registered prediction is reported as one; the registration is not edited."]
    L += [r"The ladder behind ``first at'' admits each cell's added seeds in registered order with the",
          r"other topped-up cells at their final $n$, so a change is attributable to that cell's own",
          r"seeds and not to BH moving underneath it.",
          r"The last column gives each cell's registered budget and the tier it came from",
          r"(adj.\ = the BH-adjusted requirement, unc.\ = the uncorrected one), or the reason it was",
          r"not topped up.}"]
    return L


def emit_tex(rows, note, changed=(), ladders=None, n_reg=0, tally=None, path=OUT_TEX):
    """TWO ROWS PER TOPPED-UP CELL, not two column groups.

    The side-by-side layout this started as ran 129pt past \\textwidth, and four of the seven rows
    would have been half full of dashes: only three cells have an augmented read at all. Stacking the
    two reads spends the width on the four cells that have something in both columns and gives the
    remaining three a single row, which is what they are.
    """
    def row(e, read, note_):
        call = {"inconclusive": "incon.", "freeze": "freeze", "adapt": "adapt"}.get(
            e["call"], e["call"])
        q = f"${e['q']:.3f}$" if e["q"] is not None else "--"
        return (f"{read} & {e['n']} & ${e['mean']:+.1f}{{\\pm}}{e['sem']:.1f}$ & {q} & "
                f"{call} & {note_} \\\\")
    lines = ["% GENERATED by scripts/emit_power_topup.py -- do not edit by hand.",
             "\\begin{table}[t]", "\\centering"]
    lines += caption(rows, changed, ladders or {}, n_reg, tally)
    lines += [
        "\\label{tab:power_topup}", "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}l r l r r r l l@{}}",
        "\\toprule",
        "Cell & $\\Vb{\\text{ridge}}$ & read & $n$ & $\\denc$ & $q$ & call & top-up \\\\",
        "\\midrule",
    ]
    for i, r in enumerate(rows):
        if i:
            lines.append("\\addlinespace")
        head = f"{r['label']} & ${r['gate']:+.3f}$ & "
        # The note sits on the row it describes: on the augmented row where there is one (it says what
        # budget produced that row), and on the only row there is where the cell was not topped up.
        lines.append(head + row(r["published"], "published",
                                "" if r["augmented"] else r["topup_note"]))
        if r["augmented"]:
            lines.append(" & & " + row(r["augmented"], "augmented", r["topup_note"]))
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", "", f"% {note}"]
    path.write_text("\n".join(lines) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--ladder-boot", type=int, default=200,
                    help="the ladder reads only the BH-adjusted t-test, so its bootstrap is small")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="report the state of the top-up without emitting the table; for use while "
                         "the runs are still going")
    a = ap.parse_args()

    if not REG.exists():
        sys.exit(f"{REG.relative_to(ROOT)} is missing; Phase F1 must be registered first")
    reg = json.loads(REG.read_text())
    all_topup = [r["path"] for r in reg["planned_runs"]]
    have = [p for p in all_topup if (ROOT / p).exists()]
    print(f"registered top-up runs: {len(have)} of {len(all_topup)} records exist")
    if len(have) < len(all_topup) and not a.allow_incomplete:
        missing = [p for p in all_topup if p not in have]
        sys.exit(f"{len(missing)} registered run(s) missing, e.g. {missing[:2]}; "
                 f"pass --allow-incomplete to report the state without emitting the table")

    pub = _analyse(False, a.boot)
    print(" ", _check_published(pub, a.boot))
    aug = _analyse(True, a.boot)

    gp = [e for e in reg["cells"]]
    rows, ladders = [], {}
    for e in gp:
        ref = e["ref"]
        p, q_ = pub[ref], aug[ref]
        topped = e["action"] == "top up"
        if topped:
            # The tier is abbreviated because it has to fit a column: "adj." is the BH-adjusted budget
            # and "unc." the uncorrected one, and the caption says which is which rather than leaving
            # the reader to guess from three letters.
            tier = {"adjusted": "adj.", "uncorrected": "unc."}.get(e["tier"], e["tier"])
            note = (f"${e['n']}{{\\to}}{e['target_n']}$ ({tier})"
                    if e["tier"] else f"${e['n']}{{\\to}}{e['target_n']}$")
        elif e["action"].startswith("none (UNRESOLVABLE"):
            note = f"not run: needs $n{{=}}{e['n_needed']}$"
        else:
            note = "not run: already decisive"
        rows.append(dict(
            # cm._display, not a local abbreviation: every row must name its n_train, because these
            # seven cells do not share one (the ETTh2 spectrum cells ran at 500, their neighbours at
            # 1,000) and a row that omits it reads as sharing the row above's.
            ref=ref, label=cm._display(p["cell"], p.get("n_train")), gate=p["gate"],
            published=dict(n=p["d_enc"]["n"], mean=p["d_enc"]["mean"], sem=p["d_enc"]["sem"],
                           sd=p["d_enc"]["sd"], lo=p["d_enc"]["lo"], hi=p["d_enc"]["hi"],
                           q=p["d_enc"]["q"], call=p["call"], mde=p["d_enc"]["mde"]),
            augmented=(dict(n=q_["d_enc"]["n"], mean=q_["d_enc"]["mean"], sem=q_["d_enc"]["sem"],
                            sd=q_["d_enc"]["sd"], lo=q_["d_enc"]["lo"], hi=q_["d_enc"]["hi"],
                            q=q_["d_enc"]["q"], call=q_["call"], mde=q_["d_enc"]["mde"])
                       if topped else None),
            tier=e["tier"], action=e["action"], topup_note=note,
            expected_to_stay_inconclusive=e["expected_to_stay_inconclusive"]))

    changed = []
    for r in rows:
        if r["augmented"] is None:
            continue
        rungs, at = ladder(reg, r["ref"], r["published"]["call"], all_topup, a.ladder_boot)
        ladders[r["ref"]] = rungs
        r["call_changed_at_n"] = at
        if at is not None:
            changed.append((r["ref"], r["published"]["call"], r["augmented"]["call"], at))

    # The call tally over EVERY cell, under both reads, and the refs whose call moves anywhere in the
    # matrix rather than only among the seven this table prints. BH couples all of the tests, so "one
    # call changed" is a claim about the whole matrix and counting the seven gate-passing rows cannot
    # establish it: an added seed shifts every other cell's q through the ranking.
    tally = dict(m=len(pub),
                 pub={k: sum(c["call"] == k for c in pub.values()) for k in CALLS},
                 aug={k: sum(c["call"] == k for c in aug.values()) for k in CALLS},
                 changed_refs=sorted(r for r in pub if pub[r]["call"] != aug[r]["call"]))
    for w, t in (("published", tally["pub"]), ("augmented", tally["aug"])):
        assert sum(t.values()) == tally["m"], f"{w}: {sum(t.values())} calls tallied of {tally['m']}"
    # Two more whole-matrix facts the appendix states about the augmented read, derived here rather
    # than asserted there. (a) The degradation definition's three forms: the augmented seeds cannot
    # manufacture a degradation cell by way of d_enc, but that is an argument and this is the count.
    # (b) How far an UNTOPPED cell's q moves, which is the size of the BH coupling the added seeds
    # induce -- the reason the ladder holds the other topped-up cells fixed in the first place.
    topped_refs = {r["ref"] for r in rows if r["augmented"]}
    tally["degrades"] = {w: {k: sum(bool(c.get(k)) for c in d.values())
                             for k in ("degrades_unanimous", "degrades_ci", "degrades_ci_ungated")}
                         for w, d in (("pub", pub), ("aug", aug))}
    _dq = [(abs(aug[r]["d_enc"]["q"] - pub[r]["d_enc"]["q"]), r) for r in pub
           if r not in topped_refs and pub[r].get("d_enc") and aug[r].get("d_enc")
           and pub[r]["d_enc"]["q"] is not None and aug[r]["d_enc"]["q"] is not None]
    tally["max_dq_untopped"], tally["max_dq_untopped_ref"] = max(_dq)
    # For every cell whose call moved, the BH arithmetic behind the move. The appendix has to explain
    # why a cell the registration expected to stay inconclusive crossed anyway, and the explanation is
    # arithmetic rather than rhetorical: the registered budget was computed at alpha/m, which is
    # Bonferroni, while BH's threshold for a cell at rank j is j*alpha/m. Both numbers are recorded
    # here so that the explanation in the prose is checkable instead of plausible.
    _order = sorted((c["d_enc"]["t_p"], r) for r, c in aug.items()
                    if c.get("d_enc") and c["d_enc"]["t_p"] == c["d_enc"]["t_p"])
    _rank = {r: i for i, (_, r) in enumerate(_order, 1)}
    tally["bh_diag"] = {
        ref: dict(t_p=aug[ref]["d_enc"]["t_p"], bh_rank=_rank[ref],
                  m_tests=aug[ref]["d_enc"]["m_tests"],
                  bh_threshold=_rank[ref] * pi.ALPHA / aug[ref]["d_enc"]["m_tests"],
                  alpha_adj=aug[ref]["d_enc"]["alpha_adj"],
                  n_needed_adj_published=pub[ref]["d_enc"]["n_needed_adj"],
                  n_needed_adj_augmented=aug[ref]["d_enc"]["n_needed_adj"])
        for ref, *_ in changed}
    assert tally["changed_refs"] == sorted(r for r, *_ in changed), (
        f"cells whose call moves across the whole matrix {tally['changed_refs']} differ from the "
        f"topped-up cells whose call moved {sorted(r for r, *_ in changed)}; a cell that was NOT "
        f"topped up changed call, which the table's 'one call changes' note would not report")

    print("\nPUBLISHED vs AUGMENTED on the paired encoder contrast (BH across 31 cells):")
    for r in rows:
        p, q_ = r["published"], r["augmented"]
        line = (f"  {r['ref']:20s} gate {r['gate']:+.3f}  n={p['n']:2d} "
                f"{p['mean']:+6.2f}+-{p['sem']:.2f} q={p['q']:.3f} {p['call']:28s}")
        if q_:
            line += (f"-> n={q_['n']:2d} {q_['mean']:+6.2f}+-{q_['sem']:.2f} q={q_['q']:.3f} "
                     f"{q_['call']}  MDE {p['mde']:.2f}->{q_['mde']:.2f}")
        else:
            line += f"-> {r['topup_note'].replace(chr(92), '')}"
        print(line)
    print(f"  whole-matrix tally  published {tally['pub']}  augmented {tally['aug']}")
    print(f"  degradation counts  published {tally['degrades']['pub']}  "
          f"augmented {tally['degrades']['aug']}")
    for ref, dg in tally["bh_diag"].items():
        print(f"  BH arithmetic for {ref}: t_p={dg['t_p']:.5f} at rank {dg['bh_rank']} of "
              f"{dg['m_tests']}, BH threshold {dg['bh_threshold']:.5f} vs alpha/m "
              f"{dg['alpha_adj']:.5f}; registered n_needed_adj {dg['n_needed_adj_published']} "
              f"-> {dg['n_needed_adj_augmented']}")
    print(f"  largest |dq| on an UNTOPPED cell: {tally['max_dq_untopped']:.4f} "
          f"({tally['max_dq_untopped_ref']})")
    if changed:
        for ref, before, after, at in changed:
            print(f"  CALL CHANGED  {ref}: {before} -> {after}, first at n={at}")
    else:
        print("  no cell's call changed under the top-up")
    for r in rows:
        if r.get("expected_to_stay_inconclusive") and r["augmented"]:
            held = r["augmented"]["call"] == "inconclusive"
            print(f"  registered expectation for {r['ref']} (uncorrected tier, expected to stay "
                  f"inconclusive): {'held' if held else 'DID NOT HOLD'}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(
        n_registered=len(all_topup), n_present=len(have), boot=a.boot, ladder_boot=a.ladder_boot,
        rows=rows, ladders=ladders, tally=tally,
        changed=[dict(ref=r, published_call=b, augmented_call=c, first_n=n)
                 for r, b, c, n in changed]), indent=1) + "\n")
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")

    if len(have) < len(all_topup):
        print(f"NOT writing {OUT_TEX.relative_to(ROOT)}: the top-up is incomplete "
              f"({len(have)}/{len(all_topup)} records)")
        return
    note = ("ladder: each cell's added seeds admitted in registered order with the other topped-up "
            "cells at their final n, so a change is attributable to that cell's seeds")
    n = emit_tex(rows, note, changed=changed, ladders=ladders, n_reg=len(all_topup), tally=tally)
    print(f"wrote {OUT_TEX.relative_to(ROOT)}  ({n} gate-passing cells, "
          f"{sum(r['augmented'] is not None for r in rows)} topped up)")


if __name__ == "__main__":
    main()
