#!/usr/bin/env python3
"""SHA-256 manifest for the benchmark CSVs, which are used but deliberately not committed.

WHY. data/forecasting/*.csv are gitignored (`.gitignore:48`, `*.csv`) and are not ours to
redistribute: they are the standard long-term-forecasting release files. A reader can therefore run
every table emitter from a clean clone -- the run records are committed -- but cannot re-run the gate
or a fine-tuning cell without obtaining the CSVs themselves. The gap that actually breaks
reproduction is not the download; it is not knowing whether the file you downloaded is the file we
used. ETT and Weather circulate in several variants (column order, an extra index column, a
different date parse), and a variant with the same name silently changes every window and hence
every MSE.

So this manifest records, per file: SHA-256 of the bytes, size, row and column counts, and the first
and last timestamp. The hash settles identity; the shape and date range let someone diagnose WHICH
variant they have when the hash differs, which a hash alone cannot.

WHAT IT IS NOT. Not a claim that the CSVs are ours, not a redistribution, and not a download script:
it verifies, it does not fetch.

Run:  .venv12/bin/python scripts/data_manifest.py            # write the manifest
      .venv12/bin/python scripts/data_manifest.py --check    # verify local files against it
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/data_manifest.json"

# Explicit paths, not a glob of one directory: ILI lives at data/national_illness.csv rather than
# under data/forecasting/, and a glob of the latter would have silently omitted the one dataset in
# the paper that is an external case study -- the file a reader is most likely to fetch wrong.
GLOBS = ["data/forecasting/*.csv", "data/national_illness.csv"]


def describe(p):
    """Identity plus enough shape to diagnose a mismatch.

    `symlink_to` matters here: Electricity7.csv is a symlink to Electricity.csv, so the two share a
    hash, and without this field an identical digest for two differently-named datasets reads as a
    manifest bug. The 7-series restriction that makes Electricity7 a distinct cell lives in
    Electricity7Loader.FEATURE_COLUMNS (MT_001..MT_006 plus OT), not in the file -- which is exactly
    the kind of thing a reader cannot infer from a hash and needs told.
    """
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    return dict(sha256=h.hexdigest(), bytes=p.stat().st_size,
                rows=len(body), cols=len(header), header=header,
                first=body[0][0] if body else None, last=body[-1][0] if body else None,
                symlink_to=(p.readlink().name if p.is_symlink() else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare local files against the stored manifest and exit nonzero on drift")
    a = ap.parse_args()

    local = {str(p.relative_to(ROOT)): describe(p)
             for g in GLOBS for p in sorted(ROOT.glob(g))}

    if not a.check:
        OUT.write_text(json.dumps(dict(globs=GLOBS, files=local), indent=2))
        for n, d in local.items():
            note = f"  -> {d['symlink_to']}" if d["symlink_to"] else ""
            note += "  EMPTY PLACEHOLDER (no cell in the paper uses it)" if not d["rows"] else ""
            print(f"  {n:32s} {d['sha256'][:16]}  {d['rows']:>7d} rows x {d['cols']:>3d} cols  "
                  f"{d['first']} .. {d['last']}{note}")
        print(f"wrote {OUT.relative_to(ROOT)}  ({len(local)} files)")
        return 0

    if not OUT.exists():
        sys.exit(f"{OUT.relative_to(ROOT)} not found; run without --check first")
    stored = json.load(open(OUT))["files"]
    bad = 0
    # Only files the manifest knows about are checked. An EXTRA local CSV is not a failure -- a
    # reader may keep other series in the same directory -- but a MISSING or DIFFERING one is, and
    # the difference is reported on the fields that identify which variant it is.
    for n, d in stored.items():
        if n not in local:
            print(f"  MISSING  {n}")
            bad += 1
        elif local[n]["sha256"] != d["sha256"]:
            l = local[n]
            print(f"  DIFFERS  {n}\n"
                  f"           expected {d['sha256'][:16]}  {d['rows']} x {d['cols']}  "
                  f"{d['first']} .. {d['last']}\n"
                  f"           found    {l['sha256'][:16]}  {l['rows']} x {l['cols']}  "
                  f"{l['first']} .. {l['last']}")
            bad += 1
        else:
            print(f"  ok       {n}")
    print(f"{len(stored) - bad}/{len(stored)} benchmark CSVs match the manifest")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
