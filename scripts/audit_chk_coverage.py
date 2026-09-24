#!/usr/bin/env python3
"""Find numerals in the paper that NO chk() pattern in check_paper_numbers.py reaches.

WHY THIS EXISTS.  check_paper_numbers.py answers "is every registered claim still true?".  It cannot
answer "is every claim registered?", and those are different questions.  A number can sit in the prose
for ten rounds with nothing checking it, and the checker will still report a clean run -- it only knows
about the patterns someone wrote.  Two failures of exactly that shape are already recorded: ~140 lines
of untraceable numbers in app:chronos_detail, and the gate's operating point, which had five checked
sites under two phrasings while TWELVE further sites saying the gate "clears $0.20$" had none.

WHY GREPPING THE CHECKER FOR THE NUMERAL DOES NOT WORK.  The expectations in check_paper_numbers.py are
DERIVED from run records (R["gate_threshold"], not 0.20), so the digits printed in the paper do not
appear in the checker's source at all.  Grepping for them finds nothing and reads as "uncovered" for
every number in the paper, including the well-covered ones.  Coverage has to be measured the way the
checker itself measures agreement: compile each pattern, run it over the same whitespace-collapsed
text, and ask which characters it matched.  A numeral is covered iff it lies inside some pattern's
match span.

HOW IT WORKS.  It calls check_paper_numbers.build_checks(rederive()) and reads the registry the checker
itself runs, so the pattern set is the real one by construction.  An earlier version of this script
lifted the patterns statically with `ast` and could not read EIGHT of them -- the ones built at runtime
by string concatenation over a variable (`... + RETRO`), by joining a group per ladder rung, or by an
f-string over a per-backbone label.  Those eight included the gate-rho and ladder-count checks, so the
static version was quietly not measuring some of the densest numeric prose in the paper.  Using the
registry costs one rederive() (a few seconds of run-record reading) and removes the blind spot.

WHAT IT IS NOT.  An uncovered numeral is not automatically a defect.  Three legitimate categories:
  * numbers the prose exists to DISOWN, which by construction have no record (app:chronos_detail's
    superseded 84.5% is the whole example);
  * numbers verified against SOURCE rather than against a record -- file:line citations and code
    literals, which check_paper_numbers.py anchors with asserts instead of chk() patterns;
  * structural numerals -- equation numbers, layer indices, years in prose.
So this prints a list to be read, not a gate to be passed.  It exits 0 on a clean read of the paper
and 1 only if a pattern could not be compiled, which is a defect in the checker itself.

Run:  .venv12/bin/python scripts/audit_chk_coverage.py [--section LABEL] [--file NAME] [--min-digits N]
      .venv12/bin/python scripts/audit_chk_coverage.py --section app:chronos_detail
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from check_paper_numbers import BODY, build_checks, collapse, rederive   # noqa: E402

CHECKER = ROOT / "scripts/check_paper_numbers.py"

# LaTeX arguments whose digits are never claims: cross-reference keys, bibliography keys, file paths.
# Blanked to spaces rather than deleted so the character offsets stay aligned with the line map.
NOT_PROSE = re.compile(r"\\(?:label|ref|eqref|cite[a-z]*|input|includegraphics|url|href)\{[^}]*\}")


def chk_patterns():
    """(name, pattern) for every check the checker actually runs, from its own registry."""
    return [(c["name"], c["pattern"]) for c in build_checks(rederive())]


def covered_mask(text, pats):
    """A byte per character of `text`, non-zero where some chk pattern matched. Also returns errors."""
    mask, bad = bytearray(len(text)), []
    for name, pat in pats:
        try:
            rx = re.compile(pat, re.S)
        except re.error as e:
            bad.append((name, f"does not compile: {e}"))
            continue
        for m in rx.finditer(text):
            mask[m.start():m.end()] = b"\x01" * (m.end() - m.start())
    return mask, bad


def span_of(text, label):
    """Character range of the sectioning unit carrying \\label{label}, up to the next \\section."""
    key = "\\label{%s}" % label
    i0 = text.find(key)
    if i0 < 0:
        return None
    nxt = re.compile(r"\\section\{").search(text, i0 + len(key))
    return i0, (nxt.start() if nxt else len(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", help="restrict to the unit carrying this \\label")
    ap.add_argument("--file", help="restrict to one .tex file, by name")
    ap.add_argument("--min-digits", type=int, default=2,
                    help="ignore shorter numerals; 1-digit ones are mostly structural (default 2)")
    a = ap.parse_args()

    pats = chk_patterns()
    print(f"-- read {len(pats)} chk() patterns from {CHECKER.name}'s own registry")

    files = [p for p in BODY if not a.file or p.name == a.file]
    if a.file and not files:
        sys.exit(f"no body file named {a.file}; have {[p.name for p in BODY]}")

    total_uncovered, errors = 0, []
    for path in files:
        text, lmap = collapse(path)
        lo, hi = 0, len(text)
        if a.section:
            got = span_of(text, a.section)
            if got is None:
                continue
            lo, hi = got
        mask, bad = covered_mask(text, pats)
        errors += bad
        prose = NOT_PROSE.sub(lambda m: " " * len(m.group(0)), text)
        rows = []
        for m in re.finditer(r"\d+(?:\.\d+)?", prose[lo:hi]):
            s, e = lo + m.start(), lo + m.end()
            if len(m.group(0)) < a.min_digits or any(mask[s:e]):
                continue
            ctx = re.sub(r"\s+", " ", text[max(lo, s - 55):min(hi, e + 25)])
            rows.append((m.group(0), lmap[s], ctx))
        if rows:
            print(f"\n== {path.name}: {len(rows)} numeral(s) no chk pattern reaches")
            for v, ln, ctx in rows:
                print(f"   {v:>10}  {path.name}:{ln:<5d}  ...{ctx}...")
        total_uncovered += len(rows)

    print(f"\n{total_uncovered} uncovered numeral(s)"
          + (f" in {a.section}" if a.section else "")
          + ". Read them: a number the prose DISOWNS, a file:line citation anchored by an assert, or a"
            " structural index is legitimately uncovered. Anything else is an unchecked claim.")
    for name, why in errors:
        print(f"   PATTERN ERROR  {name!r}  {why}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
