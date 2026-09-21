#!/usr/bin/env python3
"""
Label every directory under results/ as CANONICAL, SUPERSEDED or ORPHAN, and write results/MANIFEST.md.

WHY. There are 139 directories under results/ and, until now, nothing marking which of them the paper
actually stands on. That objection is practical rather than stylistic: handed this repository a reviewer
cannot tell a canonical run from a superseded one, and two directories (frozen_diffts_crosseval,
unfrozen_diffts_crosseval) are from an abandoned IoT anomaly arm that reads like an experiment and is
referenced by nothing. An artifact a reviewer cannot navigate is not a reproducibility asset.

HOW THE LABELS ARE DERIVED: BY OBSERVATION, NOT BY PARSING. Each registered emitter is executed -- with
the exact flags scripts/rederive_all.sh gives it, parsed out of that file rather than restated -- under
an audit hook that records every file it actually OPENS under results/ (scripts/_trace_opens.py). A
directory is CANONICAL when a registered emitter opened a file inside it.

The first version of this script read path literals by AST instead, and it was wrong in both
directions. It missed score_prospective.py, which builds its paths from the pre-registration at
runtime, and it mis-read cell_matrix.py, which globs results/**/*.json recursively and then opens a
small subset -- so a static reader either sees no dependency or sees all 139. A glob is not a
dependency; an open is. Tracing costs a few minutes and removes the whole class of error.

WHAT 'SUPERSEDED' MEANS. A directory whose numbers were published and then corrected. Kept on purpose
-- deleting it would destroy the audit trail the corrections appendix rests on -- but not to be read as
current. This is the one hand-maintained list in the file, marked as such, with an assert that every
path in it still exists.

WHAT 'ORPHAN' MEANS. No registered emitter opened anything in it. That is not an accusation of
wrongness: most are exploratory or pre-protocol runs. It means no number in the paper depends on them,
which is precisely what a reader needs to be able to see.

Usage:  .venv12/bin/python scripts/emit_results_manifest.py
        .venv12/bin/python scripts/emit_results_manifest.py --check       # fail if MANIFEST.md is stale
        .venv12/bin/python scripts/emit_results_manifest.py --from-trace  # relabel without re-running
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "MANIFEST.md"
OUT_JSON = RESULTS / "manifest.json"
TRACE = RESULTS / "manifest_trace.json"
SWEEP = ROOT / "scripts/rederive_all.sh"
TRACER = ROOT / "scripts/_trace_opens.py"

PY = ROOT / ".venv12/bin/python"
PYFIG = Path("/opt/homebrew/bin/python3")

# The one hand-maintained list: directories whose published numbers were later corrected. Kept as the
# audit trail behind the corrections appendix. Asserted to exist, so a rename fails loudly.
SUPERSEDED = {
    "results/v5_etth2_sweep": (
        "the original ETTh2 sweep, scored against an UNFITTED linear baseline; every gate value it "
        "reports is superseded by the fitted-baseline correction"),
    "results/v47_prospective": (
        "prospective batch 1 (n=8), superseded as the primary prospective arm by v48_prospective2; "
        "retained because its pre-registration is the one batch 2 reuses"),
}

ORPHAN_NOTES = {
    "results/frozen_diffts_crosseval": "abandoned IoT anomaly arm (F1/recall on stealth attacks)",
    "results/unfrozen_diffts_crosseval": "abandoned IoT anomaly arm (F1/recall on stealth attacks)",
}


def sweep_invocations():
    """Every `run <label> <argv...>` line in rederive_all.sh, with $PY/$PYFIG resolved.

    Parsed out of the sweep so the manifest traces what the sweep actually runs, flags included. A
    restated list here would drift from the sweep, which is the failure mode this whole file exists to
    prevent.
    """
    out = []
    for raw in SWEEP.read_text().splitlines():
        s = raw.strip()
        if not s.startswith("run ") or s.startswith("#"):
            continue
        try:
            toks = shlex.split(s)
        except ValueError:
            continue
        toks = toks[1:]                      # drop 'run'
        if not toks:
            continue
        label, argv = toks[0], toks[1:]
        if not argv:
            continue
        # This script is itself a `run` line in the sweep (as --check, which reads the stored trace
        # and re-labels without re-running anything). Tracing it would be self-referential: it would
        # add an entry whose only opens are the trace file it just read, and it would make the
        # "N emitters traced" line in MANIFEST.md a function of itself. Skipped on purpose, and the
        # skip is stated in the generated file rather than left for a reader to notice.
        if any(Path(t).name == Path(__file__).name for t in argv):
            continue
        interp = argv[0]
        if interp in ("$PY", "${PY}"):
            interp = str(PY)
        elif interp in ("$PYFIG", "${PYFIG}"):
            interp = str(PYFIG)
        script = argv[1] if len(argv) > 1 else None
        if not script or not script.endswith(".py"):
            continue
        out.append(dict(label=label, interp=interp, script=script, args=argv[2:]))
    # check_paper_numbers.py is invoked by the sweep outside a `run` line (captured, not run), and it
    # reads more records than any single emitter. Omitting it would mislabel directories as orphans.
    out.append(dict(label="check_paper_numbers", interp=str(PY),
                    script="scripts/check_paper_numbers.py", args=[]))
    return out


def trace_all(invocations):
    """Run each emitter under the audit hook and collect what it opened."""
    env = dict(os.environ, SOURCE_DATE_EPOCH="1600000000")
    traces = []
    with tempfile.TemporaryDirectory() as td:
        for i, inv in enumerate(invocations):
            if not Path(inv["interp"]).exists():
                traces.append(dict(inv, status="interpreter missing", opened=[]))
                print(f"  {inv['label']:34s} SKIPPED (no {inv['interp']})")
                continue
            tf = Path(td) / f"t{i}.json"
            proc = subprocess.run(
                [inv["interp"], str(TRACER), str(tf), inv["script"], *inv["args"]],
                cwd=ROOT, env=env, capture_output=True, text=True)
            if not tf.exists():
                traces.append(dict(inv, status=f"tracer produced nothing (rc={proc.returncode})",
                                   opened=[]))
                print(f"  {inv['label']:34s} TRACER FAILED rc={proc.returncode}")
                if proc.stderr:
                    print("      " + proc.stderr.strip().splitlines()[-1][:160])
                continue
            t = json.loads(tf.read_text())
            traces.append(dict(inv, status=t["status"], error=t.get("error"),
                               opened=t["opened"]))
            n_dirs = len({p.split("/")[1] for p in t["opened"]
                          if p.startswith("results/") and len(p.split("/")) > 2})
            flag = "" if t["status"] == "ok" else f"  [{t['status']}]"
            print(f"  {inv['label']:34s} {len(t['opened']):5d} opens, {n_dirs:3d} dirs{flag}")
    return traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit nonzero if MANIFEST.md is out of date")
    ap.add_argument("--from-trace", action="store_true",
                    help="relabel from results/manifest_trace.json without re-running the emitters")
    a = ap.parse_args()

    for p in SUPERSEDED:
        assert (ROOT / p).exists(), (
            f"{p} is listed as superseded but does not exist; the hand-maintained list has drifted")

    if a.from_trace or a.check:
        if not TRACE.exists():
            sys.exit(f"{TRACE.relative_to(ROOT)} missing; run without --from-trace first")
        traces = json.load(open(TRACE))["traces"]
    else:
        invocations = sweep_invocations()
        print(f"tracing {len(invocations)} registered emitters for the files they open under results/")
        traces = trace_all(invocations)
        TRACE.write_text(json.dumps(dict(traces=traces), indent=2))
        print(f"wrote {TRACE.relative_to(ROOT)}")

    failed = [t["label"] for t in traces if t["status"] != "ok"]

    # dir -> the emitters that opened something inside it
    readers = defaultdict(set)
    top_readers = defaultdict(set)
    for t in traces:
        for p in t["opened"]:
            parts = p.split("/")
            if len(parts) < 2 or parts[0] != "results":
                continue
            if len(parts) == 2:
                top_readers[parts[1]].add(t["script"])
            else:
                readers[f"results/{parts[1]}"].add(t["script"])

    dirs = sorted(f"results/{d.name}" for d in RESULTS.iterdir() if d.is_dir())
    rows, counts = [], defaultdict(int)
    for d in dirs:
        who = sorted(readers.get(d, ()))
        if d in SUPERSEDED:
            label = "SUPERSEDED"
        elif who:
            label = "CANONICAL"
        else:
            label = "ORPHAN"
        counts[label] += 1
        rows.append(dict(path=d, label=label, readers=who,
                         n_json=len(list((ROOT / d).rglob("*.json"))),
                         note=ORPHAN_NOTES.get(d)))

    top = sorted(p.name for p in RESULTS.glob("*.json"))
    top_rows = [dict(name=n, readers=sorted(top_readers.get(n, ()))) for n in top]

    L = []
    A = L.append
    A("# What is in `results/`, and which of it the paper stands on")
    A("")
    A("GENERATED by `scripts/emit_results_manifest.py` -- do not edit by hand.")
    A("")
    A("Labels are derived by **observation**: each emitter `scripts/rederive_all.sh` runs is executed")
    A("with the flags that file gives it, under an audit hook recording every file it opens under")
    A("`results/` (`scripts/_trace_opens.py`). A directory is CANONICAL when a registered emitter")
    A("opened a file inside it. A recursive glob is not a dependency; an open is.")
    A("")
    A("`emit_results_manifest.py` is itself one of the sweep's `run` lines and is the one invocation")
    A("NOT traced: tracing it would be self-referential. Every other `run` line is traced, including")
    A("the figure scripts and the two TIER B recompute variants of the baseline ladder.")
    A("")
    A(f"- **CANONICAL** ({counts['CANONICAL']}): a registered emitter reads it. A number in the paper")
    A("  depends on it.")
    A(f"- **SUPERSEDED** ({counts['SUPERSEDED']}): published, then corrected. Kept as the audit trail")
    A("  behind the corrections appendix. Do not read as current.")
    A(f"- **ORPHAN** ({counts['ORPHAN']}): no registered emitter opens anything in it. Exploratory or")
    A("  pre-protocol runs; no number in the paper depends on them.")
    A("")
    A(f"{len(dirs)} directories, {len(top)} top-level JSON records, {len(traces)} registered emitters"
      f" traced.")
    if failed:
        A("")
        A(f"**{len(failed)} emitter(s) did not complete during tracing** "
          f"({', '.join(failed)}). Directories reached only by those may be under-labelled here; the")
        A("failure is printed rather than hidden because a silent trace failure turns a canonical")
        A("directory into an orphan.")
    A("")
    A("## Superseded, and why")
    A("")
    for p, why in sorted(SUPERSEDED.items()):
        A(f"- `{p}` -- {why}")
    A("")
    A("## The TIER A inputs: `results/*.json`")
    A("")
    A("These are what a clean clone re-derives every table and figure from.")
    A("")
    A("| record | read by |")
    A("|---|---|")
    for r in top_rows:
        who = ", ".join(f"`{Path(w).name}`" for w in r["readers"]) or "_not read during tracing_"
        A(f"| `{r['name']}` | {who} |")
    A("")
    A("## Every directory")
    A("")
    A("| directory | label | .json | read by |")
    A("|---|---|---|---|")
    order = {"CANONICAL": 0, "SUPERSEDED": 1, "ORPHAN": 2}
    for r in sorted(rows, key=lambda r: (order[r["label"]], r["path"])):
        who = ", ".join(f"`{Path(w).name}`" for w in r["readers"]) or "--"
        note = f" _{r['note']}_" if r["note"] else ""
        A(f"| `{r['path']}` | {r['label']} | {r['n_json']} | {who}{note} |")
    A("")
    A("## The two `crosseval` directories")
    A("")
    A("`results/frozen_diffts_crosseval` and `results/unfrozen_diffts_crosseval` are from an abandoned")
    A("IoT anomaly-detection arm (F1 and recall on stealth attacks), not from the forecasting work this")
    A("paper reports. Nothing references them. They are named here because they are the two directories")
    A("most likely to be mistaken for a forecasting experiment.")
    A("")
    body = "\n".join(L) + "\n"

    if a.check:
        if not OUT.exists():
            sys.exit(f"{OUT.relative_to(ROOT)} missing; run without --check")
        if OUT.read_text() != body:
            sys.exit(f"{OUT.relative_to(ROOT)} is STALE -- re-run scripts/emit_results_manifest.py")
        print(f"ok  {OUT.relative_to(ROOT)} matches the tree")
        return 0

    OUT.write_text(body)
    OUT_JSON.write_text(json.dumps(dict(dirs=rows, top_level=top_rows, counts=dict(counts),
                                        failed_emitters=failed), indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")
    print(f"  {counts['CANONICAL']} canonical, {counts['SUPERSEDED']} superseded, "
          f"{counts['ORPHAN']} orphan  ({len(dirs)} dirs)")
    if failed:
        print(f"  WARNING: {len(failed)} emitter(s) failed while tracing: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
