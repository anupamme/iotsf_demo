#!/usr/bin/env python3
"""
Run one script and record every file under results/ that it actually OPENS.

Used only by scripts/emit_results_manifest.py. The point is to answer "which run records does the
paper depend on?" by observation rather than by parsing: a static read of path literals misses
score_prospective.py, which builds its paths from the registration at runtime, and mis-reads
cell_matrix.py, which globs results/**/*.json recursively and then opens a small subset. A glob is
not a dependency; an open is.

sys.addaudithook's "open" event fires before any file is opened, including by pandas, json and
numpy, so the record is complete regardless of which library did the reading. glob/scandir do not
raise it, which is exactly the behaviour wanted here: directories that are merely enumerated do not
count as read.

EVERY OPEN IS ATTRIBUTED TO ITS CALL SITE, as `file.py:function`, and that is what makes the trace
able to say anything. Per-PROCESS opens cannot: almost every emitter imports cell_matrix, whose
moirai_cells() globs results/**/*.json and opens all 1020 of them to reject the ones that are not
paired B/D records. Judged per process, 132 of 139 directories come out "read by 16 emitters" --
including the two abandoned IoT anomaly directories, which is the exact opposite of the answer the
manifest exists to give. Per call site, that same sweep is one scan inside one function, so a
directory reached ONLY by a whole-tree scan is distinguishable from one a targeted read names.

Usage:  python scripts/_trace_opens.py <out.json> <script.py> [args...]
"""
import json
import os
import runpy
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF = os.path.realpath(__file__)


def main():
    out_path, script = Path(sys.argv[1]), sys.argv[2]
    args = sys.argv[3:]

    opened = set()
    sites = defaultdict(set)            # results-relative path -> {"file.py:function", ...}

    def call_site():
        """The innermost frame belonging to this repository, as `file.py:function`.

        Innermost, not outermost: the question is which of OUR functions issued this open, and the
        outermost repo frame is always the script's module body, which would collapse every open in
        the process onto one site and lose exactly the distinction this exists to draw. Frames from
        the stdlib and site-packages are skipped so `json.load` is never the answer, and this file is
        skipped so the hook does not attribute opens to itself.
        """
        f = sys._getframe(2)            # 0 = here, 1 = hook, 2 = whoever opened
        while f is not None:
            fn = f.f_code.co_filename
            # realpath, not a startswith on the raw co_filename: the sweep invokes every emitter by a
            # RELATIVE path (`scripts/foo.py`, cwd=ROOT), so the traced script's own frames carry a
            # relative co_filename and an unresolved prefix test misses every one of them. It matched
            # only imported modules, which sys.path makes absolute -- so heldout_decomposition.py's 146
            # targeted reads were attributed to "<outside the repo>" and the directories they name fell
            # through to SCANNED. A mis-attribution here does not fail anything; it silently moves
            # directories into the label that means "nothing depends on this".
            real = os.path.realpath(fn)
            if real.startswith(str(ROOT)) and real != SELF:
                return f"{Path(real).name}:{f.f_code.co_name}"
            f = f.f_back
        return "<outside the repo>"

    def hook(event, event_args):
        if event != "open":
            return
        p = event_args[0]
        if isinstance(p, bytes):
            try:
                p = p.decode()
            except UnicodeDecodeError:
                return
        if not isinstance(p, str):      # an int fd, e.g. from os.fdopen
            return
        # This substring test is the cheap guard that keeps the stack walk below off the hot path:
        # the hook fires on every open in the process, imports included, and only results/ paths pay.
        if "results/" not in p and "results\\" not in p:
            return
        try:
            rel = Path(p).resolve().relative_to(ROOT)
        except (ValueError, OSError):
            return
        opened.add(str(rel))
        sites[str(rel)].add(call_site())

    sys.argv = [script] + args
    sys.addaudithook(hook)      # cannot be removed once added; this process exits straight after

    status, err = "ok", None
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if code:
            status, err = f"exit {code}", str(e.code)
    except BaseException as e:                                  # noqa: BLE001 - report, never mask
        status, err = "exception", f"{type(e).__name__}: {e}"

    # Written even on failure: a partial trace plus a recorded failure is honest, a missing trace
    # silently relabels directories as orphans.
    out_path.write_text(json.dumps(
        dict(script=script, args=args, status=status, error=err, opened=sorted(opened),
             sites={k: sorted(v) for k, v in sorted(sites.items())}), indent=2))


if __name__ == "__main__":
    main()
