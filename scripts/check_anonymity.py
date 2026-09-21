#!/usr/bin/env python3
"""
Fail if anything that would be submitted carries an author, host or home-directory identifier.

WHY THIS FILE CONTAINS NO NAMES. A checker that hard-codes the strings it looks for is itself the leak:
it would be the first hit of its own search, and it would go into the supplementary bundle carrying the
identifiers the bundle is supposed to be free of. So the terms are DERIVED at runtime -- from git's
configured user, the origin remote, and the name of the home directory -- and extra terms can be added
in a gitignored file (.anonymity_terms, one per line) for anything those three do not reach, such as a
secondary email or an affiliation. On a machine with no git identity the derived set is empty and the
script says so loudly rather than passing vacuously.

WHAT IT CHECKS
  --tracked      every tracked text file, with an allowlist (below). Reports; does not gate.
  --tree DIR     every text file under DIR, no allowlist. HARD FAIL on any hit. This is the mode
                 scripts/make_supplementary.py runs against the staged bundle.
  --pdf FILE     the built PDF: extracted text AND document metadata (pdftotext, then the /Info
                 dictionary read straight out of the file, because a leaked Author field is invisible
                 in the rendered text).
  --latex        submission hygiene in paper_8/: \\iclrfinalcopy must be commented out, and there must
                 be no literal em dash in the source (the style file's fonts render one as a wrong
                 glyph, and it has shipped that way before).
With no flags, all four run.

THE ALLOWLIST, AND WHY IT IS NOT A WAIVER. Seven tracked files name the author on purpose: the public
preprint page, the pip metadata, the top-level README, the AWS notes, the abandoned IoT demo app and
the superseded paper/ directory. None of them is part of an ICLR submission. They are allowlisted HERE
and EXCLUDED from the bundle by scripts/make_supplementary.py -- two separate places, deliberately: if
a future bundle starts including one of them, --tree fails even though --tracked passes.

Usage:  .venv12/bin/python scripts/check_anonymity.py
        .venv12/bin/python scripts/check_anonymity.py --tree /tmp/bundle
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Tracked files that name the author deliberately and are not part of any submission. Excluded from
# the bundle by make_supplementary.py as well; this list only stops --tracked from crying wolf.
ALLOWLIST = {
    "index.html",                      # the public preprint page
    "README.md",                       # names the public repo in a clone command
    "setup.py",                        # pip author metadata for the abandoned demo package
    "aws_setup_instructions.md",       # operational notes, home-directory paths
    "app/pages/05_detection.py",       # abandoned IoT demo app, contact line
    "paper/main.tex",                   # the SUPERSEDED paper directory (paper_8 is current)
    "paper/sections/01_introduction.tex",
}

BINARY = {".pdf", ".png", ".jpg", ".jpeg", ".npz", ".pt", ".pth", ".ckpt", ".safetensors",
          ".zip", ".gz", ".pyc", ".pkl", ".bst", ".sty", ".woff", ".woff2", ".ttf"}


def git(*args):
    p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def derive_terms():
    """The identifiers to search for, derived rather than written down. Returns (terms, provenance)."""
    terms, prov = {}, []

    def add(t, why):
        t = t.strip()
        if len(t) >= 4 and t.lower() not in terms:
            terms[t.lower()] = t
            prov.append(f"{t!r} from {why}")

    name = git("config", "user.name")
    if name:
        add(name, "git config user.name")
        for part in name.split():
            add(part, "a component of git config user.name")
    email = git("config", "user.email")
    if email:
        add(email, "git config user.email")
        add(email.split("@")[0], "the local part of git config user.email")
    remote = git("remote", "get-url", "origin")
    if remote:
        # The handle and the repo name, not the whole URL: the URL appears in the allowlisted README
        # while the handle is what identifies a person.
        # rstrip("/") first: the configured URL here ends in a slash, and without the strip the
        # anchored match silently found nothing -- a checker that derives no term for the repo owner
        # and still prints "clean" is the exact failure this file is meant not to have.
        m = re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?$", remote.rstrip("/"))
        if m:
            add(m.group(1), "the owner of the origin remote")
    add(Path.home().name, "the name of the home directory")

    extra = ROOT / ".anonymity_terms"
    if extra.exists():
        for line in extra.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                add(line, ".anonymity_terms")
    return terms, prov


def scan(paths, terms, root):
    hits = []
    for p in paths:
        if p.suffix.lower() in BINARY or not p.is_file():
            continue
        try:
            text = p.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        low = text.lower()
        for key, shown in terms.items():
            if key not in low:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if key in line.lower():
                    hits.append((str(p.relative_to(root)), i, shown, line.strip()[:110]))
    return hits


def check_tracked(terms):
    files = [ROOT / f for f in git("ls-files").splitlines()]
    hits = scan(files, terms, ROOT)
    unexpected = [h for h in hits if h[0] not in ALLOWLIST]
    print(f"-- tracked tree: {len(files)} files, {len(hits)} identifier hits "
          f"({len(hits) - len(unexpected)} in allowlisted files)")
    for f, i, t, line in unexpected[:40]:
        print(f"   NOT ALLOWLISTED  {f}:{i}  [{t}]  {line}")
    if unexpected:
        print(f"   {len(unexpected)} hit(s) outside the allowlist. Either fix the file or, if it is "
              "genuinely not part of any submission, add it to ALLOWLIST with the reason.")
    # Gates on the UNEXPECTED hits only. The first version of this function returned 0 unconditionally
    # and printed the offenders, so a newly leaking file would have been listed above a line reading
    # "ANONYMITY CHECK CLEAN" -- a checker that reports a failure and exits 0 is worse than no checker,
    # because it is read as a pass.
    return 1 if unexpected else 0


def check_tree(root, terms):
    root = Path(root).resolve()
    hits = scan(sorted(root.rglob("*")), terms, root)
    print(f"-- staged tree {root}: {len(hits)} identifier hits (allowlist NOT applied)")
    for f, i, t, line in hits[:40]:
        print(f"   LEAK  {f}:{i}  [{t}]  {line}")
    return 1 if hits else 0


def check_pdf(pdf, terms):
    pdf = Path(pdf)
    if not pdf.exists():
        print(f"-- pdf: {pdf} does not exist; build it first")
        return 1
    fail = 0
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                         capture_output=True, text=True).stdout
    for key, shown in terms.items():
        if key in txt.lower():
            print(f"   LEAK in rendered text: [{shown}]")
            fail = 1
    # The /Info dictionary, read from the bytes: pdftotext does not show it, and it is where an author
    # name arrives without anybody typing it -- put there by the PDF producer, not by the LaTeX source.
    raw = pdf.read_bytes()
    meta = []
    for field in (b"/Author", b"/Title", b"/Subject", b"/Keywords", b"/Creator"):
        for m in re.finditer(re.escape(field) + rb"\s*\((.*?)\)", raw, re.S):
            v = m.group(1).decode("latin-1", "replace")
            meta.append(f"{field.decode()}={v!r}")
            for key, shown in terms.items():
                if key in v.lower():
                    print(f"   LEAK in PDF metadata {field.decode()}: [{shown}]")
                    fail = 1
    print(f"-- pdf {pdf.name}: {len(txt)} chars of text, metadata fields: "
          + (", ".join(meta) if meta else "none"))
    return fail


def check_latex():
    fail = 0
    src = sorted((ROOT / "paper_8").glob("*.tex")) + sorted((ROOT / "paper_8/sections").glob("*.tex"))
    for p in src:
        for i, line in enumerate(p.read_text().splitlines(), 1):
            bare = line.split("%")[0]
            if r"\iclrfinalcopy" in bare:
                print(f"   \\iclrfinalcopy IS ACTIVE at {p.relative_to(ROOT)}:{i} -- that prints the "
                      "author block")
                fail = 1
            if "—" in line:
                print(f"   literal em dash at {p.relative_to(ROOT)}:{i}")
                fail = 1
    print(f"-- latex: {len(src)} source files checked for \\iclrfinalcopy and literal em dashes")
    return fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracked", action="store_true")
    ap.add_argument("--tree")
    ap.add_argument("--pdf", nargs="?", const="paper_8/main.pdf")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    everything = not (a.tracked or a.tree or a.pdf or a.latex)

    terms, prov = derive_terms()
    if not terms:
        sys.exit("NO IDENTIFIERS DERIVED: no git user.name, no user.email, no origin remote and no "
                 ".anonymity_terms. This check would pass vacuously, which is worse than not running "
                 "it. Set a git identity or write .anonymity_terms.")
    print(f"{len(terms)} identifier(s) derived, none written down in this file:")
    for line in prov:
        print(f"   {line}")

    fail = 0
    if a.tracked or everything:
        fail += check_tracked(terms)
    if a.tree:
        fail += check_tree(a.tree, terms)
    if a.pdf or everything:
        fail += check_pdf(ROOT / (a.pdf or "paper_8/main.pdf"), terms)
    if a.latex or everything:
        fail += check_latex()

    print("ANONYMITY CHECK CLEAN" if fail == 0 else f"ANONYMITY CHECK FAILED: {fail} category(ies)")
    return fail


if __name__ == "__main__":
    raise SystemExit(main())
