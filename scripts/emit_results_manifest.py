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
    # results/v47_prospective was listed here until 21 Sep 2026, as "superseded as the primary
    # prospective arm by v48_prospective2". That was never true, and the trace is what showed it: v47
    # is opened by seven targeted call sites including score_prospective.py:load, it holds a record the
    # matrix loader keeps, and all EIGHT cells the prospective arm scores are inside it. v48 holds
    # condition-A references for six DIFFERENT cells and nothing else. Labelling the directory the
    # pre-registered prospective test is scored from "do not read as current" was the single most
    # misleading line in a file whose whole purpose is to stop a reader being misled.
}

# Hand-written notes for directories a reader would otherwise have to guess about, whatever label
# they end up with. Named DIR_NOTES rather than ORPHAN_NOTES because two of the three entries are now
# SCANNED, not ORPHAN, and a dict whose name disagrees with its contents is how the labels drifted.
DIR_NOTES = {
    "results/frozen_diffts_crosseval": "abandoned IoT anomaly arm (F1/recall on stealth attacks)",
    "results/unfrozen_diffts_crosseval": "abandoned IoT anomaly arm (F1/recall on stealth attacks)",
    "results/v48_prospective2": (
        "a SECOND prospective batch that was never run: six condition-A zero-shot references and "
        "nothing else -- no condition B or D run, and not even the registration file "
        "scripts/preregister_prospective2.py would write. The prospective arm the paper reports is "
        "batch 1, in results/v47_prospective, and none of these six cells is among its eight"),
}


def rel_to_root(p) -> str:
    """A path as recorded in the trace: repo-relative if it is inside the repo, absolute otherwise.

    Not cosmetic. results/manifest_trace.json is committed and SHIPS IN THE SUPPLEMENTARY BUNDLE, and
    the sweep's interpreter is $PY = <repo>/.venv12/bin/python -- whose absolute form contains the home
    directory, i.e. the author's name. The first version stored str(PY) and the bundle's anonymity gate
    (check_anonymity.py --tree, allowlist disabled) refused to write the zip, naming 21 occurrences.
    That is the gate working, but a de-identification that depends on a gate catching it is one release
    away from being an author-identifying artifact, so the identifier is not written in the first place.
    /opt/homebrew/bin/python3 is outside the repo and identifies nobody, so it is left absolute.
    """
    # NOT p.resolve() first. .venv12/bin/python is a SYMLINK to an interpreter outside the repository,
    # so resolving it lands in /opt/homebrew, relative_to(ROOT) raises, and the fallback returns the
    # absolute path -- which is how the first version of this function left every one of the 21
    # identifiers in place while looking like it had removed them. The question here is where the path
    # POINTS FROM, not where it ends up.
    p = Path(p)
    for cand in (p, Path(os.path.abspath(p))):
        try:
            return str(cand.relative_to(ROOT))
        except ValueError:
            continue
    return str(p)


def abs_interp(interp: str) -> Path:
    """The inverse of rel_to_root, for actually invoking the thing."""
    p = Path(interp)
    return p if p.is_absolute() else ROOT / p


def sweep_invocations():
    """Every TIER A `run <label> <argv...>` line in rederive_all.sh, with $PY/$PYFIG resolved.

    Parsed out of the sweep so the manifest traces what the sweep actually runs, flags included. A
    restated list here would drift from the sweep, which is the failure mode this whole file exists to
    prevent.

    TIER A ONLY, and the exclusion is load-bearing twice over. The sweep's `run` lines inside
    `if [ "$WITH_DATA" = 1 ]` refit the whole baseline ladder from the benchmark CSVs -- hours, not
    minutes, and the CSVs are gitignored, so tracing them makes this script unrunnable on the clean
    clone whose directory layout it exists to document. It would also buy nothing: the two recompute
    lines differ from their --from-json counterparts (traced above) only in reading data/*.csv and
    WRITING the ladder JSON that --from-json READS, so every results/ directory they touch is already
    labelled by the cheap path. The first version of this parser was line-based and blind to the
    guard: it spent 65 minutes on one recompute line before anyone noticed it was in the list at all.
    """
    out = []
    # `if` depth, and the depth at which an optional-tier guard opened. Depth is tracked rather than a
    # bare in/out flag because the WITH_DATA guard CONTAINS a nested `if` (the data_manifest hash
    # check) whose `fi` would otherwise be read as closing the guard -- leaving the rest of the guard
    # looking like top level. The matplotlib guard must NOT be skipped, which is why only `if [ "$WITH_`
    # arms the skip, and `skipped` is asserted non-empty below so a renamed guard fails loudly instead
    # of quietly restoring the hours-long recompute to the trace.
    depth, guard_depth, skipped = 0, None, []
    for raw in SWEEP.read_text().splitlines():
        s = raw.strip()
        if s.startswith("if ") or s == "if":
            depth += 1
            if s.startswith('if [ "$WITH_') and guard_depth is None:
                guard_depth = depth
            continue
        if s == "fi":
            if guard_depth == depth:
                guard_depth = None
            depth -= 1
            continue
        if not s.startswith("run ") or s.startswith("#"):
            continue
        if guard_depth is not None:
            skipped.append(shlex.split(s)[1] if len(shlex.split(s)) > 1 else s)
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
            interp = rel_to_root(PY)
        elif interp in ("$PYFIG", "${PYFIG}"):
            interp = rel_to_root(PYFIG)
        script = argv[1] if len(argv) > 1 else None
        if not script or not script.endswith(".py"):
            continue
        out.append(dict(label=label, interp=interp, script=script, args=argv[2:]))
    # Both asserts are about the PARSE, not the sweep: they fire if this function stopped understanding
    # the file it reads. An unbalanced depth means an `if`/`fi` form it does not recognise appeared; an
    # empty `skipped` means the optional-tier guards stopped matching, which would silently put the
    # hours-long CSV recompute back in the trace and make this script need the gitignored benchmark data.
    assert depth == 0, f"unbalanced if/fi while parsing {SWEEP.name}: depth {depth} at EOF"
    assert skipped, (f"no optional-tier `run` lines found in {SWEEP.name}. Either the TIER B/D guards "
                     "were renamed, or their run lines moved; re-read the guard before trusting this.")
    # check_paper_numbers.py is invoked by the sweep outside a `run` line (captured, not run), and it
    # reads more records than any single emitter. Omitting it would mislabel directories as orphans.
    out.append(dict(label="check_paper_numbers", interp=rel_to_root(PY),
                    script="scripts/check_paper_numbers.py", args=[]))
    return out, skipped


def classify_sites(site_dirs):
    """Split call sites into whole-tree SCANS and targeted reads, by how much of the tree each opened.

    The distinction is the whole basis of the CANONICAL label, so the threshold must not be a guess
    that happens to work. It isn't: the distribution is sharply bimodal -- the recursive scan in
    cell_matrix.moirai_cells() opens into every directory that exists, and every other call site opens
    into single digits, because it was given a path. The assert below states that as a requirement.
    If some future call site opens into, say, 40% of the tree, it is neither a scan nor a targeted
    read and no threshold can classify it honestly, so this fails and asks for a human rather than
    bucketing it and quietly changing what CANONICAL means.
    """
    n = len({d for ds in site_dirs.values() for d in ds})
    scans = {s for s, ds in site_dirs.items() if len(ds) >= 0.5 * n}
    ambiguous = {s: len(ds) for s, ds in site_dirs.items() if 0.1 * n < len(ds) < 0.5 * n}
    assert not ambiguous, (
        f"call site(s) opened into a middling share of the {n} directories: {ambiguous}. The "
        "scan/targeted split assumes a bimodal distribution and this is neither; read the call site "
        "and decide what it is before trusting any label in this file.")
    return scans, {s for s in site_dirs if s not in scans}


def survived_the_filter():
    """Directories holding at least one record the main matrix's loader ACCEPTED.

    The second of the two observations behind CANONICAL, and it is needed because the first cannot see
    the paper's core. Every cell in the main matrix arrives through cell_matrix.moirai_cells(), which
    is a whole-tree scan -- so on call-site evidence alone the 31 scored cells would look no different
    from the abandoned IoT records sitting in the same tree, which the same scan also opens and then
    throws away. What separates them is the filter: a paired B/D record with both final_val_mse and
    test_mse survives, anything else does not. So the loader is asked what it KEPT rather than what it
    touched -- its return value is keyed by directory, so this reads its verdict rather than
    reimplementing its predicate, which would be a second copy free to drift from the first.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import io                                     # noqa: PLC0415 - local to keep the import cost here
    import contextlib                             # noqa: PLC0415
    with contextlib.redirect_stdout(io.StringIO()):
        import cell_matrix                        # noqa: PLC0415
        kept = {k[0] for k in cell_matrix.moirai_cells()}
    return {f"results/{d}" for d in kept}


def trace_all(invocations):
    """Run each emitter under the audit hook and collect what it opened."""
    env = dict(os.environ, SOURCE_DATE_EPOCH="1600000000")
    traces = []
    with tempfile.TemporaryDirectory() as td:
        for i, inv in enumerate(invocations):
            interp = abs_interp(inv["interp"])
            if not interp.exists():
                traces.append(dict(inv, status="interpreter missing", opened=[]))
                print(f"  {inv['label']:34s} SKIPPED (no {inv['interp']})")
                continue
            tf = Path(td) / f"t{i}.json"
            proc = subprocess.run(
                [str(interp), str(TRACER), str(tf), inv["script"], *inv["args"]],
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
                               opened=t["opened"], sites=t.get("sites", {})))
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
        stored = json.load(open(TRACE))
        traces = stored["traces"]
        untraced = stored.get("untraced_optional_tier", [])
    else:
        invocations, untraced = sweep_invocations()
        print(f"tracing {len(invocations)} registered emitters for the files they open under results/")
        print(f"  not traced ({len(untraced)}): the optional-tier lines {untraced} -- they need the "
              "gitignored benchmark CSVs, and their results/ opens are covered by the --from-json path")
        traces = trace_all(invocations)
        blob = json.dumps(dict(traces=traces, untraced_optional_tier=untraced), indent=2)
        # This file is committed and ships in the supplementary bundle, so it is checked for the one
        # identifier it is structurally prone to carrying -- the home-directory name, which arrives
        # inside any absolute path above the repo. Checked here rather than left to the bundle's
        # anonymity gate: that gate runs once, at release, and a de-identification that only holds
        # because someone ran the last step is not one.
        home = Path.home().name
        assert home not in blob and str(ROOT) not in blob, (
            f"the trace records an absolute path above the repository ({home!r}); every path in it must "
            "be repo-relative (see rel_to_root) or this file carries the author's home directory into "
            "the supplementary bundle")
        TRACE.write_text(blob)
        print(f"wrote {TRACE.relative_to(ROOT)}")

    failed = [t["label"] for t in traces if t["status"] != "ok"]

    # dir -> the emitters that opened something inside it
    readers = defaultdict(set)
    top_readers = defaultdict(set)
    site_dirs = defaultdict(set)            # "file.py:function" -> directories it opened into
    dir_sites = defaultdict(set)            # directory -> the call sites that opened into it
    for t in traces:
        for p in t["opened"]:
            parts = p.split("/")
            if len(parts) < 2 or parts[0] != "results":
                continue
            if len(parts) == 2:
                top_readers[parts[1]].add(t["script"])
            else:
                readers[f"results/{parts[1]}"].add(t["script"])
        for p, ss in t.get("sites", {}).items():
            parts = p.split("/")
            if len(parts) < 3 or parts[0] != "results":
                continue
            for s in ss:
                site_dirs[s].add(parts[1])
                dir_sites[f"results/{parts[1]}"].add(s)

    # Without call-site attribution every directory falls through to SCANNED and the counts still look
    # entirely reasonable -- which is how this shipped once already, after trace_all() aggregated the
    # per-invocation traces and silently dropped the `sites` key. A label that degrades quietly into a
    # different, plausible label is worse than one that breaks, so the degradation is made loud here.
    assert site_dirs, (
        "the trace carries no call-site attribution, so every directory would be labelled SCANNED and "
        "the CANONICAL count would silently collapse to the matrix loader's filter alone. Re-run "
        "without --from-trace to regenerate results/manifest_trace.json with scripts/_trace_opens.py.")
    scans, targeted_sites = classify_sites(site_dirs)
    targeted = {d for d, ss in dir_sites.items() if ss - scans}
    survived = survived_the_filter()
    canonical = targeted | survived

    dirs = sorted(f"results/{d.name}" for d in RESULTS.iterdir() if d.is_dir())
    rows, counts = [], defaultdict(int)
    for d in dirs:
        who = sorted(readers.get(d, ()))
        # Order matters: SUPERSEDED wins over everything (it is a statement about the numbers having
        # been corrected, not about who reads the directory), and SCANNED must be checked before
        # ORPHAN or every scanned directory would read as untouched.
        if d in SUPERSEDED:
            label = "SUPERSEDED"
        elif d in canonical:
            label = "CANONICAL"
        elif who:
            label = "SCANNED"
        else:
            label = "ORPHAN"
        counts[label] += 1
        rows.append(dict(path=d, label=label, readers=who,
                         sites=sorted(dir_sites.get(d, set()) - scans),
                         in_matrix=d in survived,
                         n_json=len(list((ROOT / d).rglob("*.json"))),
                         note=DIR_NOTES.get(d)))

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
    A("`results/` **and the call site that opened it** (`scripts/_trace_opens.py`).")
    A("")
    A("Two observations decide CANONICAL, and it takes both. Almost every emitter imports")
    A("`cell_matrix`, whose `moirai_cells()` globs `results/**/*.json` and opens all of them in order")
    A("to *reject* the ones that are not paired B/D records -- so \"an emitter opened it\" is true of")
    A("nearly every directory here, including ones nothing uses. A directory is CANONICAL when either")
    A("a **targeted read** names it (a call site that was not that whole-tree scan) or a record inside")
    A("it **survived the scan's filter** (the loader is asked what it kept, not what it touched). The")
    A("second is what the main matrix arrives by; the first is what every other arm arrives by.")
    A("")
    A("Two kinds of `run` line are deliberately NOT traced, and the count above reflects that.")
    A("`emit_results_manifest.py` is itself one of the sweep's `run` lines: tracing it would be")
    A(f"self-referential. And the {len(untraced)} line(s) inside the sweep's optional-tier guards")
    A(f"({', '.join(f'`{u}`' for u in untraced) or 'none'}) refit the baseline ladder from the")
    A("benchmark CSVs, which are gitignored -- so tracing them would make this file impossible to")
    A("regenerate from a clean clone, which is the situation it exists to document. Nothing is lost:")
    A("those lines differ from their `--from-json` counterparts (traced) only in reading `data/*.csv`")
    A("and writing the ladder JSON that `--from-json` reads, so they reach no `results/` directory the")
    A("traced path does not. Every other `run` line is traced, figure scripts included.")
    A("")
    A(f"- **CANONICAL** ({counts['CANONICAL']}): a targeted read names it, or a record in it survived")
    A("  the matrix loader's filter. These are the directories the paper's numbers come from.")
    A(f"- **SUPERSEDED** ({counts['SUPERSEDED']}): published, then corrected. Kept as the audit trail")
    A("  behind the corrections appendix. Do not read as current.")
    A(f"- **SCANNED** ({counts['SCANNED']}): opened only by the whole-tree scan, and nothing inside it")
    A("  survived that scan's filter. Being read and rejected is not being used. This is the honest")
    A("  limit of what tracing can show: it rules out the main matrix, and a targeted arm would have")
    A("  shown up as a call site, so nothing here is known to be load-bearing -- but the evidence is")
    A("  absence of a positive signal, not proof of irrelevance.")
    A(f"- **ORPHAN** ({counts['ORPHAN']}): never opened at all, by any emitter, at any call site.")
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
    # "read by" used to list every emitter whose process opened anything inside the directory, which
    # was sixteen identical names on almost every row -- a column that is the same everywhere carries
    # no information. What a reader needs is the two things that decide the label: whether the matrix
    # loader kept a record from here, and which targeted call sites named it.
    A("| directory | label | .json | in the matrix | named by (targeted call sites) |")
    A("|---|---|---|---|---|")
    order = {"CANONICAL": 0, "SUPERSEDED": 1, "SCANNED": 2, "ORPHAN": 3}
    for r in sorted(rows, key=lambda r: (order[r["label"]], r["path"])):
        who = ", ".join(f"`{s}`" for s in r["sites"]) or "--"
        note = f" _{r['note']}_" if r["note"] else ""
        A(f"| `{r['path']}` | {r['label']} | {r['n_json']} | {'yes' if r['in_matrix'] else '--'} "
          f"| {who}{note} |")
    A("")
    A("## The two `crosseval` directories")
    A("")
    # The previous version of this file said "nothing references them" in prose while the table two
    # sections above labelled both CANONICAL, read by sixteen emitters -- the document contradicted
    # itself, and the prose was the part that was right. The claim is now read back out of the derived
    # labels, and asserted, so the two cannot disagree again.
    cross = {r["path"]: r["label"] for r in rows if "crosseval" in r["path"]}
    assert cross and all(v in ("SCANNED", "ORPHAN") for v in cross.values()), (
        f"the crosseval directories came out {cross}. This section says no number depends on them; "
        "either that is no longer true, or the labelling changed. Do not ship the two disagreeing.")
    A(", ".join(f"`{p}` ({v})" for p, v in sorted(cross.items())) + " are from an abandoned")
    A("IoT anomaly-detection arm (F1 and recall on stealth attacks), not from the forecasting work this")
    A("paper reports. The whole-tree scan opens the record each one holds and rejects it, which is why")
    A("they are listed as scanned rather than untouched. They are named here because they are the two")
    A("directories most likely to be mistaken for a forecasting experiment.")
    A("")
    A("## The whole-tree scans")
    A("")
    A("These call sites open into half the tree or more; an open from one of them is not evidence that")
    A("a directory is used. Every other call site was given its path and is listed per directory above.")
    A("")
    for s in sorted(scans):
        A(f"- `{s}` -- opened into {len(site_dirs[s])} directories")
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
          f"{counts['SCANNED']} scanned, {counts['ORPHAN']} orphan  ({len(dirs)} dirs)")
    print(f"  {len(scans)} whole-tree scan site(s), {len(targeted_sites)} targeted; "
          f"{len(survived)} dir(s) hold a record the matrix loader kept")
    if failed:
        print(f"  WARNING: {len(failed)} emitter(s) failed while tracing: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
