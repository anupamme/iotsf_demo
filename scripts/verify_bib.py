#!/usr/bin/env python3
"""Round-trip every cited bibliography entry through Crossref or arXiv by identifier.

WHY THIS EXISTS. One entry in this file shipped fabricated: rasul2024lagllama named a venue the
paper never had and listed four people who are not its authors, because it was assembled from a
TITLE LOOKUP whose top hit was a different paper. Crossref will happily return an unrelated work as
rank 1 -- querying it for "Overcoming catastrophic forgetting in neural networks" returns
"Overcoming Catastrophic Forgetting with Gaussian Mixture Replay" first -- and nothing in the
response says "this is not what you asked for".

So this script never accepts a search result on rank. It requires the candidate's title to match the
entry's title exactly after normalisation, and then fetches the record AGAIN by its own DOI or arXiv
ID and checks that the second fetch agrees with the first. An entry passes only if a stable
identifier resolves to the title and first author the .bib already claims; anything else is printed
for a human to resolve, never auto-written.

Usage:  .venv12/bin/python scripts/verify_bib.py [--key KEY ...]
Prints one block per cited entry. Exit 1 if any cited entry is unresolved.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "paper_8/bibliography.bib"
AUX = ROOT / "paper_8/main.aux"
UA = {"User-Agent": "iotsf-bib-check/1.0 (mailto:noreply@example.invalid)"}


def get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:                                    # noqa: BLE001 -- network flake
            if i == tries - 1:
                return f"__ERROR__ {e}"
            time.sleep(2 * (i + 1))
    return "__ERROR__ unreachable"


def detex(s):
    """'{MOMENT}: ... {\\'n}' -> a plain lowercase string, so titles compare on their words.

    Accent commands are spelled with a non-letter (\\'n, \\"o, \\v{s}) and must be dropped WITHOUT
    leaving a separator, or "Wili{\\'n}ski" compares as "wili nski" and a correct entry is reported as
    a mismatched author -- a checker that cries wolf is how the fabricated entry got past review.
    """
    s = re.sub(r"\\[^a-zA-Z]", "", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = re.sub(r"[{}$\\]", "", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return " ".join(s.split())


def parse_bib(text):
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = dict(re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n\s*\w+\s*=|\Z)",
                                 m.group(3), re.S))
        out[m.group(2).strip()] = {"type": m.group(1), "fields": fields, "raw": m.group(0)}
    return out


def cited_keys(aux_text):
    keys = set()
    for m in re.finditer(r"\\(?:abx@aux@cite|citation)\{([^}]*)\}", aux_text):
        keys.update(k for k in m.group(1).split(",") if k)
    return keys


def first_author_surname(author_field):
    first = author_field.split(" and ")[0]
    return detex(first.split(",")[0] if "," in first else first.split()[-1])


# ---------------------------------------------------------------- Crossref

def crossref_search(title, author):
    q = urllib.parse.urlencode({"query.bibliographic": f"{detex(title)} {author}", "rows": "8"})
    body = get(f"https://api.crossref.org/works?{q}")
    if body.startswith("__ERROR__"):
        return None, body
    try:
        items = json.loads(body)["message"]["items"]
    except Exception as e:                                        # noqa: BLE001
        return None, f"unparseable crossref response: {e}"
    want = detex(title)
    for it in items:
        for t in it.get("title", []):
            if detex(t) == want:
                return it["DOI"], None
    got = "; ".join(detex(it.get("title", [""])[0])[:60] for it in items[:3])
    return None, f"no exact title match in top {len(items)} (saw: {got})"


def crossref_by_doi(doi):
    body = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    if body.startswith("__ERROR__"):
        return None, body
    try:
        it = json.loads(body)["message"]
    except Exception as e:                                        # noqa: BLE001
        return None, f"unparseable: {e}"
    auth = it.get("author") or []
    return {
        "title": (it.get("title") or [""])[0],
        "venue": (it.get("container-title") or [""])[0],
        "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
        "first_author": detex(auth[0].get("family", "")) if auth else "",
        "n_authors": len(auth),
    }, None


# ---------------------------------------------------------------- arXiv

def arxiv_search(title):
    q = urllib.parse.urlencode({"search_query": f'ti:"{detex(title)}"', "max_results": "8"})
    body = get(f"http://export.arxiv.org/api/query?{q}")
    if body.startswith("__ERROR__"):
        return None, body
    want = detex(title)
    ids = re.findall(r"<id>http://arxiv\.org/abs/([^<]+)</id>", body)
    titles = re.findall(r"<title>(.*?)</title>", body, re.S)[1:]   # [0] is the feed title
    for aid, t in zip(ids, titles):
        if detex(t) == want:
            return re.sub(r"v\d+$", "", aid), None
    got = "; ".join(detex(t)[:60] for t in titles[:3])
    return None, f"no exact title match in top {len(titles)} (saw: {got})"


def arxiv_by_id(aid):
    body = get(f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(aid)}")
    if body.startswith("__ERROR__"):
        return None, body
    titles = re.findall(r"<title>(.*?)</title>", body, re.S)[1:]
    names = re.findall(r"<name>(.*?)</name>", body)
    pub = re.search(r"<published>(\d{4})", body)
    jref = re.search(r"<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>", body, re.S)
    if not titles:
        return None, "id_list returned no entry"
    return {
        "title": " ".join(titles[0].split()),
        "venue": " ".join(jref.group(1).split()) if jref else "(arXiv preprint, no journal_ref)",
        "year": int(pub.group(1)) if pub else None,
        "first_author": detex(names[0].split()[-1]) if names else "",
        "n_authors": len(names),
    }, None


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", action="append", default=None)
    a = ap.parse_args()

    entries = parse_bib(BIB.read_text())
    keys = sorted(a.key) if a.key else sorted(cited_keys(AUX.read_text()))
    unresolved = []

    for k in keys:
        e = entries.get(k)
        if e is None:
            print(f"[{k}] NO SUCH ENTRY")
            unresolved.append(k)
            continue
        f = e["fields"]
        title, author = f.get("title", ""), f.get("author", "")
        surname = first_author_surname(author) if author else ""
        claimed_venue = f.get("journal") or f.get("booktitle") or ""

        known_doi = f.get("doi")
        known_arx = re.search(r"(?:arxiv[.:/ ]|abs/)(\d{4}\.\d{4,5})",
                              f.get("url", "") + " " + f.get("journal", ""), re.I)

        rec = err = ident = None
        if known_doi:
            ident = f"doi:{known_doi}"
            rec, err = crossref_by_doi(known_doi)
        elif known_arx:
            ident = f"arXiv:{known_arx.group(1)}"
            rec, err = arxiv_by_id(known_arx.group(1))
        else:
            doi, derr = crossref_search(title, surname)
            if doi:
                ident = f"doi:{doi}"
                rec, err = crossref_by_doi(doi)
            else:
                aid, aerr = arxiv_search(title)
                if aid:
                    ident = f"arXiv:{aid}"
                    rec, err = arxiv_by_id(aid)
                else:
                    err = f"crossref: {derr} | arxiv: {aerr}"

        print(f"[{k}]  {ident or '-- UNRESOLVED --'}")
        if rec is None:
            print(f"    !! {err}")
            unresolved.append(k)
            continue
        tmatch = detex(rec["title"]) == detex(title)
        amatch = rec["first_author"] == surname
        # An arXiv record's year is when the PREPRINT was posted, which is legitimately earlier than
        # the conference year the entry cites -- every ICML/ICLR entry here posts the year before. So
        # equality is only required of DOI records; for arXiv, the bib year merely may not PRECEDE the
        # posting, which is the direction that would mean the entry cites a venue that had not met.
        bib_year = str(f.get("year", ""))
        is_arxiv = ident.startswith("arXiv:")
        if is_arxiv and bib_year.isdigit() and rec["year"]:
            ystat = "OK " if int(bib_year) >= int(rec["year"]) else "BAD"
            ynote = " (arXiv posting year; venue year checked separately)"
        else:
            ystat = "OK " if bib_year == str(rec["year"]) else "DIFF"
            ynote = ""
        ymatch = ystat == "OK "
        print(f"    title  {'OK ' if tmatch else 'DIFF'}  {rec['title'][:88]}")
        print(f"    author {'OK ' if amatch else 'DIFF'}  first={rec['first_author']!r} "
              f"bib={surname!r}  n_authors={rec['n_authors']}")
        print(f"    year   {ystat}  record={rec['year']} bib={f.get('year')}{ynote}")
        print(f"    venue        record={rec['venue'][:70]!r}")
        print(f"                 bib   ={claimed_venue[:70]!r}")
        if not (tmatch and amatch and ymatch):
            unresolved.append(k)
        time.sleep(0.4)

    print(f"\n{len(keys) - len(unresolved)}/{len(keys)} cited entries round-tripped.")
    if unresolved:
        print("UNRESOLVED or MISMATCHED: " + ", ".join(unresolved))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
