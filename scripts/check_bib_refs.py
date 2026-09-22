#!/usr/bin/env python3
"""Verify every bibliography entry against an external record, via clibib (github.com/delip/clibib).

WHY, AND WHY IDENTIFIERS ONLY.  A bibliography is the one part of a paper where a plausible-looking
error survives every check the repo runs: check_paper_numbers.py verifies numbers against run
records, latexmk verifies that a key resolves, and neither can tell whether the paper being cited
exists, or whether the venue and year we attribute to it are the venue and year it has.

clibib fetches a BibTeX record for a DOI, an arXiv ID, a URL, an ISBN, a PMID -- or a TITLE. The
title path is the trap. Asked for the title of a real paper in this bibliography, "Exploring
Representations and Interventions in Time Series Foundation Models", it returned
`@book{valjakka_visual_2018}`, "Visual arts, representations and interventions in contemporary
China: urbanized interface" -- a different work, in a different decade, in a different field, with
no warning and no confidence score. Asked for that paper's arXiv ID, 2409.12915, it returned the
paper. So this script queries by IDENTIFIER and treats a title lookup as evidence of nothing:
a title hit is reported as UNVERIFIED, exactly like a miss, because a hit cannot be distinguished
from the China-book case without reading the result.

WHAT EACH VERDICT MEANS.
  CONFIRMED   an identifier in our entry resolved, and the returned title matches ours. The work
              exists and our title for it is right.
  MISMATCH    the identifier resolved to a DIFFERENT work. Our DOI/arXiv link is wrong: either the
              identifier or the title/authors around it. This is the failure worth finding.
  AUTHOR      same work, but an author name disagrees with the record. Compared position by position
              on surname and given name, not just the first author, because a wrong given name on the
              second author is both the likeliest error and the least likely to be noticed.
  YEAR/VENUE  the identifier resolved to the same work, but the year differs. Usually legitimate --
              an arXiv preprint dated 2024 published at a 2025 conference -- and never resolvable by
              this script alone, which sees only the record clibib returns. Reported, not judged.
  NO_RECORD   the identifier did not resolve. Inconclusive: a lookup service gap, not evidence.
  UNVERIFIED  the entry carries no identifier, so there is nothing to round-trip. These need a human
              at the venue's own site; the script says which ones and stops.

Run:  .venv12/bin/python scripts/check_bib_refs.py [--bib paper_8/bibliography.bib] [--all]
      --all also queries entries with no identifier, by title, purely to show what comes back; the
      verdict stays UNVERIFIED whatever it returns.

Results cache to results/bib_check.json so a re-run does not re-query; delete it to refetch.
"""
import argparse
import difflib
import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results/bib_check.json"
VCACHE = ROOT / "results/bib_check_venues.json"
TITLE_MATCH = 0.90      # normalized-title similarity above which two records are the same work


def parse_bib(path):
    src = path.read_text()
    out = []
    for typ, key, body in re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", src, re.S):
        def field(name):
            m = re.search(r"\b" + name + r"\s*=\s*[{\"](.+?)[}\"]\s*,?\s*(?=\n|$)", body, re.S | re.I)
            return re.sub(r"\s+", " ", m.group(1)).strip() if m else None
        url = field("url") or ""
        # An arXiv ID hides in more places than the `eprint` field. Several entries here carry it in
        # the URL, and luo2023forgetting carries it only inside `journal = {arXiv preprint
        # arXiv:2308.08747}` -- so scan the whole entry body, or that entry looks identifier-less and
        # gets reported as unverifiable when it is in fact checkable.
        arx = (re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", body, re.I)
               or re.search(r"arXiv[:\s]\s*([0-9]{4}\.[0-9]{4,5})", body, re.I))
        eprint = field("eprint")
        out.append(dict(key=key.strip(), type=typ, title=field("title"), author=field("author"),
                        year=field("year"), doi=field("doi"), url=url or None,
                        arxiv=(arx.group(1) if arx else None) or eprint,
                        venue=field("booktitle") or field("journal") or field("institution")))
    return out


def cited_keys(bbl):
    if not bbl.exists():
        return None
    return set(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]+)\}", bbl.read_text()))


def fold(s):
    r"""Strip accents to bare ASCII letters, from BOTH LaTeX and Unicode spellings of the same name.

    This has to happen before any comparison or it manufactures errors: our entry writes
    Wili{\'n}ski and the fetched record writes Wilinski or Wiliński, and a naive strip of
    non-alphanumerics turns the first into "wili nski" and reports a surname disagreement against a
    record that says the same thing. Handles \'{n} and \'n and {\.Z} and combining marks alike.
    """
    s = s or ""
    s = re.sub(r"\\([a-zA-Z]{1,2}|['`^\"~=.])\s*\{?(\w)\}?", r"\2", s)    # \v{s} \'n {\.Z}
    s = re.sub(r"\\[a-zA-Z]+", " ", s)                                    # any remaining command
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


def norm(s):
    s = re.sub(r"\{|\}", "", fold(s))
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def title_sim(a, b):
    return difflib.SequenceMatcher(None, " ".join(norm(a)), " ".join(norm(b))).ratio()


def one_name(s):
    """(surname, first initial) for one author, tolerating "Last, First" and "First Last"."""
    if "," in s:
        last, _, rest = s.partition(",")
        last, given = norm(last), norm(rest)
    else:
        toks = norm(s)
        last, given = (toks[-1:], toks[:1]) if toks else ([], [])
    return (last[-1] if last else "", given[0] if given else "")


def given_clash(x, y):
    """Do two given names disagree? Initials are compared as initials; full names in full.

    "Zhen" vs "Zhengxuan" share an initial and are different people, so an initial-only comparison
    would pass it -- which is how a wrong given name reaches a submitted bibliography. But half the
    world's BibTeX writes "P. Khosla", so a one-letter given name may only be held to its letter.
    """
    if not x or not y:
        return False
    if len(x) == 1 or len(y) == 1:
        return x[0] != y[0]
    return x != y


def authors(author):
    """[(surname, initial)] and whether the list was truncated with "and others"."""
    if not author:
        return [], False
    parts = [p.strip() for p in re.split(r"\s+and\s+", author) if p.strip()]
    trunc = bool(parts) and norm(parts[-1]) == ["others"]
    if trunc:
        parts = parts[:-1]
    return [one_name(p) for p in parts], trunc


def author_diff(ours, theirs):
    """Author disagreements between our entry and the fetched record, as readable strings.

    Compared position by position on (surname, first initial): a fabricated or mistyped given name is
    exactly the error a first-author-only check misses, and it is the one most likely to survive into a
    submitted bibliography. "and others" truncates our list, so only the prefix is compared. An
    ordering difference or an extra author is reported as a count difference rather than silently
    aligned away, because guessing the intended alignment would invent agreement.
    """
    a, trunc = authors(ours)
    b, _ = authors(theirs)
    if not a or not b:
        return []
    out = []
    n = min(len(a), len(b))
    for i in range(n):
        if a[i][0] != b[i][0]:
            out.append(f"author {i+1} surname: ours {a[i][0]!r} vs record {b[i][0]!r}")
        elif given_clash(a[i][1], b[i][1]):
            out.append(f"author {i+1} ({a[i][0]}) given name: ours {a[i][1]!r} vs "
                       f"record {b[i][1]!r}")
    if not trunc and len(a) != len(b):
        out.append(f"author count: ours {len(a)} vs record {len(b)}")
    return out


def clibib(query):
    try:
        p = subprocess.run(["clibib", "--first", query], capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return None, f"{type(e).__name__}"
    txt = p.stdout.strip()
    if not txt.startswith("@"):
        return None, (p.stderr.strip().splitlines() or ["no BibTeX in stdout"])[-1][:120]
    got = {}
    for name in ("title", "author", "year", "doi", "url", "note", "booktitle", "journal",
                 "publisher"):
        m = re.search(r"\b" + name + r"\s*=\s*[{\"](.+?)[}\"],?\s*\n", txt, re.S | re.I)
        if m:
            got[name] = re.sub(r"\s+", " ", m.group(1)).strip()
    m = re.match(r"@(\w+)\{([^,]+),", txt)
    if m:
        got["_type"], got["_key"] = m.group(1), m.group(2)
    return got, None


def dblp(title):
    """[(venue, year, title)] from OpenReview for a title, preferring the real venue over the preprint.

    (The function keeps its name because it is the venue-lookup slot; the source changed. DBLP itself
    is unusable from here -- it sits behind a bot challenge that answers HTTP 200 with an HTML "Making
    sure you're not a bot!" page, so every JSON parse fails and, cached naively, would have looked like
    "venue not found" for all 28 entries. OpenReview mirrors DBLP's CoRR records and is authoritative
    for ICLR, ICML 2023+, NeurIPS 2023+ and TMLR, which is most of this bibliography.)

    WHY A SECOND SOURCE IS NEEDED.  clibib resolved 16 of the 20 cited entries through arXiv, and an
    arXiv record carries `publisher = {arXiv}` and the LATEST REVISION's year -- it cannot say whether
    a paper appeared at ICML 2024 or NeurIPS 2023, and its year routinely disagrees with the venue's
    for a reason that is not an error. But `booktitle` and `year` are exactly where a wrong
    attribution hides: it looks right, it compiles, and no identifier check touches it. One cited entry
    here claims TMLR 2023 for a paper TMLR published in 2022.

    Using DBLP by TITLE is safe in a way that using clibib by title is not, because this runs only on
    entries whose identifier ALREADY round-tripped: the work's existence and title are established, and
    DBLP is asked one narrow question about a title we have confirmed. A title hit that disagrees is
    reported for a human to read, never auto-trusted.
    """
    import datetime
    import urllib.request
    # A wide limit, then filter by title: OpenReview's relevance ranking puts comment threads and
    # review replies (which have no title at all) above the paper itself, so a small limit silently
    # returns nothing for papers it definitely has.
    url = "https://api2.openreview.net/notes/search?limit=25&term=" + urllib.parse.quote(title)
    req = urllib.request.Request(url, headers={"User-Agent": "bib-check (one-off, paced)"})

    def val(c, k):
        v = c.get(k)
        return v.get("value") if isinstance(v, dict) else v

    def parse(data):
        out = []
        for n in data.get("notes", []):
            c = n.get("content", {})
            t = val(c, "title")
            if not t or title_sim(title, t) < TITLE_MATCH:
                continue
            v = val(c, "venue") or val(c, "venueid") or ""
            # Prefer the year the venue string itself states ("ICML 2024 Poster"); a note's pdate is
            # when the camera-ready was posted, which can land in the next calendar year.
            m = re.search(r"\b(19|20)\d{2}\b", v)
            pd = n.get("pdate") or n.get("cdate")
            yr = (int(m.group(0)) if m else
                  (datetime.datetime.fromtimestamp(pd / 1000, datetime.UTC).year if pd else None))
            out.append((v, yr, t))
        out.sort(key=lambda v: ("corr" in str(v[0]).lower(),))   # the real venue before the preprint
        return out

    # AN EMPTY RESULT IS NOT AN ANSWER.  Under load this endpoint returns HTTP 200 with a note list
    # that does not contain the paper -- the Diffusion-TS ICLR 2024 record came back empty during a
    # paced sweep and returned two hits on the very next call. Cached as "no hit", that soft failure
    # would have inflated the coverage of this whole report, so zero title matches is retried and only
    # believed after it repeats.
    last = None
    for attempt in range(4):
        if attempt:
            time.sleep(5 * attempt)
        try:
            with urllib.request.urlopen(req, timeout=60) as fh:
                hits = parse(json.loads(fh.read().decode()))
            if hits:
                return hits, None
            last = None
        except Exception as e:                                   # noqa: BLE001 -- inconclusive, not fatal
            last = f"{type(e).__name__}: {e}"
    return ([] if last is None else None), last


def arxiv_meta(arxiv_id):
    """Primary arXiv metadata for an ID: title, authors, v1 date, journal_ref and comment.

    THE VENUE IS USUALLY IN HERE, KEYED BY IDENTIFIER.  clibib's arXiv record drops the two fields
    that state where a preprint was published -- `journal_ref` and the author-supplied `comment`,
    which for Kumar et al. reads "ICLR (Oral) 2022" -- leaving `publisher = {arXiv}` and the latest
    revision's year. Those fields make the venue check an identifier round-trip like the rest of this
    script, instead of a title search against an index whose ranking buries the paper (OpenReview's
    own search does not return the ICLR 2022 paper for its exact title).

    Also returns the v1 date, which is the right year for an entry cited AS a preprint: the API's
    `updated`/clibib's `year` is the latest revision and drifts years later.
    """
    import urllib.request
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    req = urllib.request.Request(url, headers={"User-Agent": "bib-check (one-off, paced)"})
    last = None
    for attempt in range(4):
        if attempt:
            time.sleep(4 * 2 ** attempt)
        try:
            with urllib.request.urlopen(req, timeout=60) as fh:
                xml = fh.read().decode()
            break
        except Exception as e:                                   # noqa: BLE001 -- inconclusive
            last = f"{type(e).__name__}: {e}"
    else:
        return None, last
    m = re.search(r"<entry>(.*?)</entry>", xml, re.S)
    if not m:
        return None, "no entry in the arXiv response"
    e = m.group(1)

    def one(tag):
        g = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", e, re.S)
        return " ".join(g.group(1).split()) if g else None

    return dict(title=one("title"), authors=" and ".join(re.findall(r"<name>(.*?)</name>", e)),
                v1=(one("published") or "")[:4], journal_ref=one("arxiv:journal_ref"),
                comment=one("arxiv:comment")), None


VENUE_ALIASES = {
    "ICML": ["icml", "international conference on machine learning"],
    "NeurIPS": ["neurips", "nips", "neural information processing"],
    "ICLR": ["iclr", "international conference on learning representations"],
    "TMLR": ["tmlr", "transactions on machine learning research"],
    "AAAI": ["aaai"],
    "ACL": ["acl", "association for computational linguistics"],
    "NAACL": ["naacl"],
    # Journals arrive both spelled out and in DBLP's abbreviations; an alias gap here reads as a venue
    # disagreement, which is a fabricated finding about a correct entry.
    "PNAS": ["pnas", "national academy of sciences", "natl acad"],
    "TPAMI": ["tpami", "pattern analysis and machine intelligence", "pattern anal"],
    "IEEE Pervasive": ["pervasive computing"],
    "CoRR": ["corr", "arxiv"],
}


def venue_key(s):
    """The venue's canonical short name, or None if it is not one we can compare."""
    low = " ".join(norm(s))
    for k, pats in VENUE_ALIASES.items():
        if any(p in low for p in pats):
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", default="paper_8/bibliography.bib")
    ap.add_argument("--bbl", default="paper_8/main.bbl")
    ap.add_argument("--all", action="store_true",
                    help="also query identifier-less entries by title (verdict stays UNVERIFIED)")
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()

    entries = parse_bib(ROOT / a.bib)
    cited = cited_keys(ROOT / a.bbl)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    rows = []
    for e in entries:
        ident = ("doi", e["doi"]) if e["doi"] else (("arxiv", e["arxiv"]) if e["arxiv"] else None)
        r = dict(e, is_cited=(cited is None or e["key"] in cited), query=None, query_kind=None,
                 verdict=None, got=None, note=None, venue_confirmed_by_doi=False)
        if ident is None:
            r["verdict"] = "UNVERIFIED"
            r["note"] = "no DOI or arXiv ID in the entry; a title lookup cannot confirm it"
            if a.all:
                r["query_kind"], r["query"] = "title", e["title"]
        else:
            r["query_kind"], r["query"] = ident
        if r["query"]:
            ck = f"{r['query_kind']}:{r['query']}"
            if ck in cache:
                got, err = cache[ck]["got"], cache[ck]["err"]
            else:
                got, err = clibib(r["query"])
                cache[ck] = dict(got=got, err=err)
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(cache, indent=1) + "\n")
                time.sleep(a.sleep)
            r["got"] = got
            if r["verdict"] != "UNVERIFIED":
                if got is None:
                    r["verdict"], r["note"] = "NO_RECORD", err
                else:
                    sim = title_sim(e["title"], got.get("title"))
                    r["title_sim"] = round(sim, 3)
                    adiff = author_diff(e["author"], got.get("author"))
                    r["author_diff"] = adiff
                    # A DOI resolves through the publisher's own metadata, so it carries the venue and
                    # the publication year -- the two fields an arXiv record cannot speak to. Where
                    # that agrees with our entry, the attribution is settled at the source and no
                    # title-based index lookup can improve on it.
                    rec_v = got.get("booktitle") or got.get("journal")
                    r["venue_confirmed_by_doi"] = bool(
                        r["query_kind"] == "doi" and rec_v
                        and venue_key(rec_v) is not None
                        and venue_key(rec_v) == venue_key(e["venue"])
                        and (not (got.get("year") and e["year"])
                             or str(got["year"]) == str(e["year"])))
                    if sim < TITLE_MATCH:
                        r["verdict"] = "MISMATCH"
                        r["note"] = (f"the identifier resolves to \"{got.get('title')}\" "
                                     f"(similarity {sim:.2f}; "
                                     f"{len(adiff)} author disagreement(s))")
                    elif adiff:
                        r["verdict"] = "AUTHOR"
                        r["note"] = "; ".join(adiff)
                    elif got.get("year") and e["year"] and got["year"] != e["year"]:
                        r["verdict"] = "YEAR"
                        r["note"] = (f"same work, year {e['year']} in our entry vs {got['year']} in "
                                     f"the record ({got.get('publisher') or got.get('journal') or ''}"
                                     f"{' ' + got.get('note', '') if got.get('note') else ''})"
                                     .strip())
                    else:
                        r["verdict"] = "CONFIRMED"
            elif got:
                r["title_sim"] = round(title_sim(e["title"], got.get("title")), 3)
                r["note"] += f"; title query returned \"{got.get('title')}\" (similarity {r['title_sim']:.2f})"
        rows.append(r)

    # --- stage 2: venue and year, for entries whose identifier already round-tripped ----------------
    vcache = json.loads(VCACHE.read_text()) if VCACHE.exists() else {}
    for r in rows:
        r["venue_check"] = None
        r["arxiv_meta"] = None
        if not r["title"]:
            continue
        if r["title"] not in vcache:
            hits, err = dblp(r["title"])
            if err is None:                  # never cache a transport failure as if it were an answer
                vcache[r["title"]] = dict(hits=hits, err=err)
                VCACHE.parent.mkdir(parents=True, exist_ok=True)
                VCACHE.write_text(json.dumps(vcache, indent=1) + "\n")
            else:
                vcache[r["title"]] = dict(hits=None, err=err)
            time.sleep(max(a.sleep, 3.0))    # DBLP is a courtesy service; pace the sweep
        hits, err = vcache[r["title"]]["hits"], vcache[r["title"]]["err"]

        # Preferred source for an arXiv-resolved entry: arXiv's own journal_ref/comment, fetched by ID.
        if r["arxiv"] and not r["venue_confirmed_by_doi"]:
            ak = f"arxiv-meta:{r['arxiv']}"
            if ak not in vcache or vcache[ak].get("err"):
                meta, aerr = arxiv_meta(r["arxiv"])
                if aerr is None:
                    vcache[ak] = dict(meta=meta, err=None)
                    VCACHE.write_text(json.dumps(vcache, indent=1) + "\n")
                else:
                    vcache[ak] = dict(meta=None, err=aerr)
                time.sleep(max(a.sleep, 3.0))
            meta = vcache[ak].get("meta")
            r["arxiv_meta"] = meta
            if meta:
                stated = " ".join(x for x in (meta.get("journal_ref"), meta.get("comment")) if x)
                vk, ym = venue_key(stated), re.search(r"\b(19|20)\d{2}\b", stated)
                ours_v = venue_key(r["venue"])
                if vk and ours_v and vk == ours_v:
                    st = "ok (arXiv)"
                    if ym and r["year"] and ym.group(0) != str(r["year"]):
                        st = "YEAR_DISAGREES"
                    r["venue_check"] = dict(status=st, ours=f"{ours_v} {r['year']}",
                                            detail=f"arXiv states: {stated[:110]!r}")
                    continue
                if vk and ours_v and vk != ours_v and ours_v != "CoRR":
                    r["venue_check"] = dict(status="VENUE_DISAGREES", ours=f"{ours_v} {r['year']}",
                                            detail=f"arXiv states: {stated[:110]!r}")
                    continue
                if ours_v == "CoRR" and r["year"] and meta.get("v1") and meta["v1"] != str(r["year"]):
                    # Cited as a preprint, so the year should be the posting year, not a later revision.
                    r["venue_check"] = dict(
                        status="YEAR_DISAGREES", ours=f"preprint {r['year']}",
                        detail=f"cited as an arXiv preprint but v1 was posted in {meta['v1']}"
                               + (f"; arXiv states {stated[:60]!r}" if stated else ""))
                    continue

        if r["venue_confirmed_by_doi"]:
            # Checked first: a publisher's own DOI metadata already settled venue and year, and an
            # index miss must not be reported as "unverified" over the top of it.
            r["venue_check"] = dict(status="ok (DOI)", ours=f"{venue_key(r['venue'])} {r['year']}",
                                    detail="venue and year match the DOI's publisher metadata")
            continue
        if err or not hits:
            r["venue_check"] = dict(status="NO_DBLP_HIT", detail=err or "no title match on DBLP")
            continue
        ours_v, ours_y = venue_key(r["venue"]), r["year"]
        real = [(v, y, t) for v, y, t in hits if venue_key(v) != "CoRR"]
        cands = real or hits
        vs = {venue_key(v) or v: y for v, y, _ in cands}
        st, detail = "ok", f"index: {'; '.join(f'{v} {y}' for v, y, _ in cands[:4])}"
        if r["venue_confirmed_by_doi"]:
            st = "ok (DOI)"                         # stage 1 already matched venue AND year at source
        elif ours_v is None:
            st = "UNCOMPARABLE"                     # e.g. we cite it as a bare arXiv preprint
        elif not real and ours_v != "CoRR":
            # The index holds only the preprint mirror. That says nothing about whether the paper also
            # appeared at the venue we claim -- madry2018pgd really is ICLR 2018 and really has a 2017
            # CoRR record -- so this is inconclusive, not a disagreement.
            st = "PREPRINT_ONLY"
        elif ours_v not in vs:
            st = "VENUE_DISAGREES"
        elif ours_y and vs[ours_v] and str(vs[ours_v]) != str(ours_y):
            st = "YEAR_DISAGREES"
            detail = f"ours {ours_v} {ours_y}, DBLP has {ours_v} {vs[ours_v]}"
        r["venue_check"] = dict(status=st, detail=detail, ours=f"{ours_v} {ours_y}")

    order = ["MISMATCH", "AUTHOR", "NO_RECORD", "YEAR", "UNVERIFIED", "CONFIRMED"]
    print(f"{len(rows)} entries in {a.bib}"
          + (f", {sum(r['is_cited'] for r in rows)} of them cited in {a.bbl}" if cited else ""))
    for v in order:
        sel = [r for r in rows if r["verdict"] == v]
        if not sel:
            continue
        print(f"\n{v}: {len(sel)} ({sum(r['is_cited'] for r in sel)} cited)")
        for r in sorted(sel, key=lambda r: (not r["is_cited"], r["key"])):
            mark = "*" if r["is_cited"] else " "
            print(f" {mark} {r['key']:28s} {r['year'] or '????'}  {(r['title'] or '')[:64]}")
            if r["note"]:
                print(f"     -> {r['note'][:200]}")
    vbad = [r for r in rows if (r.get("venue_check") or {}).get("status")
            in ("VENUE_DISAGREES", "YEAR_DISAGREES")]
    print(f"\nVENUE/YEAR cross-check: {len(vbad)} disagreement(s) "
          f"({sum(r['is_cited'] for r in vbad)} cited)")
    for r in sorted(vbad, key=lambda r: (not r["is_cited"], r["key"])):
        v = r["venue_check"]
        print(f" {'*' if r['is_cited'] else ' '} {r['key']:28s} {v['status']:16s} ours: {v['ours']}")
        print(f"     -> {v['detail'][:170]}")
    for st in ("ok (DOI)", "ok (arXiv)", "ok", "PREPRINT_ONLY", "NO_DBLP_HIT", "UNCOMPARABLE"):
        sel = [r["key"] for r in rows if (r.get("venue_check") or {}).get("status") == st]
        if sel:
            print(f"  {st}: {len(sel)} ({', '.join(sel[:8])}{' ...' if len(sel) > 8 else ''})")

    out = ROOT / "results/bib_check_report.json"
    out.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}   (* = cited in the built PDF)")
    print("UNVERIFIED is not a finding against the reference; it means this tool cannot speak to it. "
          "A title lookup returned a 2018 book on Chinese urban art for one real paper here, so "
          "title hits are not treated as confirmation.")


if __name__ == "__main__":
    main()
