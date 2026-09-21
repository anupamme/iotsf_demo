#!/usr/bin/env python3
"""
Build the supplementary bundle the paper promises, from `git archive` and never from the worktree.

WHY git archive. The worktree contains peer-review correspondence (reviewer comments, our responses,
the review maps), which is gitignored precisely so it cannot be committed to a public repository. A
bundle assembled by copying directories would pick all of it up, and the failure would be silent and
irreversible -- the material would be in the venue's hands. `git archive HEAD` can only emit tracked
paths, so the correspondence is structurally unable to appear. The correspondence patterns are ALSO
checked for by name afterwards, belt and braces, because "structurally impossible" is a claim about a
tool's behaviour and this one is worth verifying rather than trusting.

WHAT GOES IN, AND WHY IT IS AN INCLUDE LIST. The repository still contains an abandoned IoT
anomaly-detection project: an app/, a src/, 147 synthetic .npy files under data/, a superseded paper/
directory, and an unrelated PDF at the top level. An exclude list against that tree would be long and
would fail open -- anything new lands in the bundle by default. So the bundle is an INCLUDE list,
checked against what \\S~"What we release" actually promises:

    paper_8/        the LaTeX source, the style files, every table and figure (minus workshop/)
    scripts/        the analysis and experiment scripts, every table emitter, the re-derivation sweep
    results/        the per-run JSON records, the data manifest, the pre-registration files

Excluded and stated: model checkpoints (gitignored, too large -- the appendices say which runs kept
one); the benchmark CSVs (not ours to redistribute -- results/data_manifest.json pins their hashes);
paper_8/workshop/ (a different, shorter document -- shipping it beside a conference submission invites
a dual-submission reading that is not ours to invite); and the seven tracked files that deliberately
name the author (the public preprint page, the pip metadata, the README, the AWS notes, the demo app
and the superseded paper/ directory), which are also allowlisted in scripts/check_anonymity.py.

THE GATE. scripts/check_anonymity.py --tree runs on the staged tree with the allowlist DISABLED, so any
author, email, repo-owner or home-directory string anywhere in the bundle is a hard failure. That is
deliberately a stricter check than the one the repository itself passes.

Usage:  .venv12/bin/python scripts/make_supplementary.py
        .venv12/bin/python scripts/make_supplementary.py --out /tmp/supplementary.zip --keep
"""
import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INCLUDE = ["paper_8", "scripts", "results"]

# Excluded at the `git archive` level, as pathspecs, rather than deleted after extraction. Necessary
# rather than tidy: .gitignore lists *.pt, but .gitignore does not untrack what was already added, and
# 45 checkpoints totalling 2.5 GB are still in HEAD. Extracting those and then deleting them would
# write 2.5 GB to a temporary directory to throw it away, and it would make the bundle's size depend
# on a repository-hygiene problem that is not the bundle's to solve. FORBIDDEN_SUFFIXES below still
# asserts they are absent afterwards -- the exclusion is the mechanism, the assertion is the check.
EXCLUDE_PATHSPECS = [":(exclude)*.pt", ":(exclude)*.pth", ":(exclude)*.ckpt",
                     ":(exclude)*.safetensors", ":(exclude)*.csv", ":(exclude)*.npz"]

# Dropped from the staged tree after extraction. Every entry is a prefix match on the bundle-relative
# path, and every entry carries its reason -- an unexplained exclusion in a reproducibility bundle is
# indistinguishable from something being hidden.
DROP = {
    "paper_8/workshop": "a different, shorter document; not part of this submission",
    "paper_8/response_letter.md": "correspondence with the venue (tracked by mistake; see below)",
    "paper_8/main.fdb_latexmk": "latexmk's build database, machine-local",
    "paper_8/main.fls": "latexmk's file list, machine-local",
}

# Checked for by name in the staged tree, in addition to the structural guarantee git archive gives.
CORRESPONDENCE = [
    r"official_comment", r"response_reviewer", r"rebuttal_round", r"review_map",
    r"response_letter", r"meta_?review", r"reviewer\d",
]

# What the paper's "What we release" paragraph promises, as paths that must exist in the bundle. Not a
# spot check: each of these is a sentence in the submitted PDF, and a bundle missing one makes that
# sentence false.
PROMISED = {
    "the one-command re-derivation script": "scripts/rederive_all.sh",
    "the prose-number checker": "scripts/check_paper_numbers.py",
    "the benchmark-data manifest": "results/data_manifest.json",
    "the results index": "results/MANIFEST.md",
    "the LaTeX source": "paper_8/main.tex",
    "the conference style file": "paper_8/iclr2027_conference.sty",
    "the bibliography": "paper_8/main.bbl",
    "the positive-control pre-registration": "results/positive_control/preregistration.json",
    "the prospective-arm pre-registration": "results/v47_prospective/preregistration.json",
    "the LOCO pre-registration": "scripts/preregister_loco.py",
    "the task-B generator": "scripts/make_conflicting_series.py",
}

# Patterns that must NOT appear in the bundle, with the reason. Checkpoints are the size problem the
# paper states; the .csv rule is the redistribution one.
FORBIDDEN_SUFFIXES = {
    ".pt": "model checkpoint (size; the appendices say which runs kept one)",
    ".pth": "model checkpoint",
    ".ckpt": "model checkpoint",
    ".safetensors": "model checkpoint",
    ".csv": "benchmark data is not ours to redistribute; results/data_manifest.json pins its hashes",
}


def stage(dest: Path):
    """Extract the INCLUDE prefixes of HEAD into dest. Tracked paths only, by construction."""
    with tempfile.NamedTemporaryFile(suffix=".tar") as tf:
        p = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "HEAD",
                            *INCLUDE, *EXCLUDE_PATHSPECS], stdout=tf, stderr=subprocess.PIPE)
        if p.returncode != 0:
            sys.exit(f"git archive failed: {p.stderr.decode()[:400]}")
        tf.flush()
        with tarfile.open(tf.name) as t:
            # filter="data" explicitly: the default changes in Python 3.14 and warns until then, and
            # "data" is the correct choice here -- it refuses absolute paths, parent traversal, links
            # and device files, none of which a source bundle has any business containing.
            t.extractall(dest, filter="data")
    return sorted(q for q in dest.rglob("*") if q.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="supplementary.zip")
    ap.add_argument("--keep", action="store_true", help="leave the staged tree in place for inspection")
    a = ap.parse_args()
    out = (ROOT / a.out).resolve() if not Path(a.out).is_absolute() else Path(a.out)

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", *INCLUDE],
                           capture_output=True, text=True).stdout.strip().splitlines()
    if dirty:
        print(f"WARNING: {len(dirty)} uncommitted change(s) under {', '.join(INCLUDE)}. The bundle is "
              f"built from HEAD ({head[:8]}), so those changes are NOT in it:")
        for line in dirty[:12]:
            print(f"    {line}")
        if len(dirty) > 12:
            print(f"    ... and {len(dirty) - 12} more")

    tmp = Path(tempfile.mkdtemp(prefix="supp_"))
    staged = tmp / "supplementary"
    files = stage(staged)
    print(f"staged {len(files)} tracked files from HEAD {head[:8]}")

    # ---- drop, with a count per reason so nothing leaves silently
    for prefix, why in DROP.items():
        hit = [q for q in staged.rglob("*") if str(q.relative_to(staged)).startswith(prefix)]
        # Counted BEFORE anything is unlinked. The first version counted after, so is_file() was false
        # for everything it had just deleted and every line read "dropped 0" -- a drop report that
        # always says nothing was dropped is the silent exclusion the report exists to prevent.
        n = sum(1 for q in hit if q.is_file())
        for q in sorted((q for q in hit if q.is_file()), reverse=True):
            q.unlink()
        for q in sorted((d for d in hit if d.is_dir()), reverse=True):
            shutil.rmtree(q, ignore_errors=True)
        print(f"  dropped {n:4d}  {prefix}  -- {why}")

    files = sorted(q for q in staged.rglob("*") if q.is_file())

    fail = []

    # ---- correspondence, by name, on top of what git archive already guarantees
    corr = [str(q.relative_to(staged)) for q in files
            if any(re.search(p, q.name, re.I) for p in CORRESPONDENCE)]
    if corr:
        fail.append(f"CORRESPONDENCE IN THE BUNDLE: {corr}")
    print(f"  correspondence check: {len(corr)} file(s) matching {len(CORRESPONDENCE)} patterns")

    # ---- forbidden suffixes
    for suf, why in FORBIDDEN_SUFFIXES.items():
        hit = [str(q.relative_to(staged)) for q in files if q.suffix == suf]
        if hit:
            fail.append(f"{len(hit)} {suf} file(s) in the bundle ({why}): {hit[:4]}")

    # ---- everything the paper says we release
    missing = {k: v for k, v in PROMISED.items() if not (staged / v).exists()}
    for k, v in missing.items():
        fail.append(f"PROMISED BUT ABSENT -- {k}: {v}"
                    + ("  (generated: run scripts/emit_results_manifest.py, then commit it -- the "
                       "bundle is built from HEAD, not the worktree)" if v.endswith("MANIFEST.md")
                       else ""))
    print(f"  promised-inventory check: {len(PROMISED) - len(missing)}/{len(PROMISED)} present")

    # ---- the anonymity gate, allowlist disabled
    print("  anonymity check on the staged tree (allowlist NOT applied):")
    rc = subprocess.run([str(ROOT / ".venv12/bin/python"), str(ROOT / "scripts/check_anonymity.py"),
                         "--tree", str(staged)], capture_output=True, text=True)
    for line in rc.stdout.strip().splitlines():
        print(f"    {line}")
    if rc.returncode != 0:
        fail.append("the staged tree carries an author/host identifier (see above)")

    if fail:
        print("\nBUNDLE NOT WRITTEN:")
        for f in fail:
            print(f"  - {f}")
        print(f"  staged tree left at {staged} for inspection")
        return 1

    # ---- a README for the bundle, generated so it cannot drift from what is actually in it
    n_json = len([q for q in files if q.suffix == ".json"])
    (staged / "README_SUPPLEMENTARY.md").write_text(
        "# Supplementary material\n\n"
        f"Built by `scripts/make_supplementary.py` from commit `{head[:12]}`; "
        f"{len(files)} files, {n_json} JSON run records.\n\n"
        "## Where to start\n\n"
        "- `results/MANIFEST.md` labels every directory under `results/` **canonical**, "
        "**superseded** or **orphan**, derived by tracing which files each registered emitter "
        "actually opens. Read it before reading any run record.\n"
        "- `bash scripts/rederive_all.sh` re-derives every table and figure the paper `\\input`s from "
        "the JSON records alone, and re-checks every number in the prose against them.\n"
        "- `scripts/check_paper_numbers.py` is that prose check on its own.\n\n"
        "## What is deliberately not here\n\n"
        "- **Model checkpoints** (`*.pt`): too large. The appendices state which runs retained one; "
        "two re-measurements were blocked by runs whose checkpoints were not kept.\n"
        "- **Benchmark CSVs**: not ours to redistribute. `results/data_manifest.json` pins the SHA-256 "
        "of every file we used, and `scripts/data_manifest.py --check` verifies yours match before any "
        "step that needs them.\n"
        "- **The synthetic task-B series** used by the positive control: regenerate it with "
        "`scripts/make_conflicting_series.py` and check the SHA-256 recorded in "
        "`results/positive_control/preregistration.json`.\n"
        "- **Correspondence with the venue**, and a workshop-length version of part of this work.\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(q for q in staged.rglob("*") if q.is_file())
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for q in files:
            z.write(q, q.relative_to(staged.parent))
    mb = out.stat().st_size / 1e6
    print(f"\nwrote {out} -- {len(files)} files, {mb:.1f} MB")
    if a.keep:
        print(f"staged tree kept at {staged}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
