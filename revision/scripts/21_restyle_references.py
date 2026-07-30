#!/usr/bin/env python3
"""Restyle the manuscript reference list into Plant Communications (Molecular Plant) format.

The submitted list is in Vancouver style with author lists truncated at six names
("et al."). This script resolves each entry against CrossRef — by DOI when the
entry carries one, otherwise by a bibliographic query — and rebuilds the entry as

    Surname, A.B., Surname, C.D., and Surname, E.F. (Year). Title. J. Abbrev. Vol:pages.

A CrossRef record is only accepted when the title, the first author's surname, and
the year all agree with the existing entry, so a wrong-paper substitution cannot
pass silently. Entries that cannot be verified are copied through unchanged and
listed in the report for manual handling; nothing is invented.

Usage:
    python 21_restyle_references.py --manuscript ../../manuscript_v2.md \
        --out ../../references_plant_communications.md
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

MAILTO = "woncyim@gmail.com"
MAX_AUTHORS = 10  # Cell Press lists up to ten authors before "et al."


def norm_title(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return " ".join(s.split())


def crossref(url: str) -> dict | None:
    try:
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", "25", "-A", f"mailto:{MAILTO}", url],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(proc.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None
    if payload.get("status") != "ok":
        return None
    return payload.get("message")


def by_doi(doi: str) -> dict | None:
    return crossref(f"https://api.crossref.org/works/{doi}")


def by_title(title: str, author: str) -> dict | None:
    from urllib.parse import quote

    url = (
        "https://api.crossref.org/works"
        f"?query.bibliographic={quote(title)}"
        f"&query.author={quote(author)}&rows=3&select=DOI,title,author,"
        "container-title,short-container-title,volume,page,issued,type"
    )
    msg = crossref(url)
    if not msg:
        return None
    items = msg.get("items") or []
    return items[0] if items else None


def year_of(rec: dict) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        parts = rec.get(key, {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def format_authors(rec: dict) -> str | None:
    authors = rec.get("author") or []
    names = []
    for a in authors:
        family = (a.get("family") or "").strip()
        given = (a.get("given") or "").strip()
        if not family:
            if a.get("name"):
                names.append(a["name"].strip())
                continue
            return None
        initials = "".join(
            f"{p[0].upper()}."
            for p in re.split(r"[\s.\-]+", given)
            if p
        )
        names.append(f"{family}, {initials}" if initials else family)
    if not names:
        return None
    truncated = len(names) > MAX_AUTHORS
    shown = names[:MAX_AUTHORS]
    if truncated:
        return ", ".join(shown) + ", et al."
    if len(shown) == 1:
        return shown[0]
    return ", ".join(shown[:-1]) + ", and " + shown[-1]


def journal_abbrev(rec: dict) -> str:
    short = rec.get("short-container-title") or []
    full = rec.get("container-title") or []
    name = (short[0] if short else (full[0] if full else "")).strip()
    return name


def format_entry(rec: dict) -> str | None:
    authors = format_authors(rec)
    year = year_of(rec)
    titles = rec.get("title") or []
    if not authors or not year or not titles:
        return None
    title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", titles[0])).strip().rstrip(".")
    journal = journal_abbrev(rec)
    vol = (rec.get("volume") or "").strip()
    pages = (rec.get("page") or "").strip().replace("-", "–")
    doi = (rec.get("DOI") or "").strip()

    out = f"{authors} ({year}). {title}."
    if journal:
        out += f" {journal}"
        if vol:
            out += f" {vol}"
            if pages:
                out += f":{pages}"
        elif pages:
            out += f" {pages}"
        out += "."
    if doi:
        out += f" doi:{doi}."
    return re.sub(r"\s+", " ", out).strip()


REF_LINE = re.compile(r"^[A-Z][^|#]*\.\s")


def parse_refs(text: str) -> list[str]:
    block = text.split("## References", 1)[1]
    return [ln.strip() for ln in block.splitlines() if ln.strip() and REF_LINE.match(ln.strip())]


def extract_doi(entry: str) -> str | None:
    m = re.search(r"doi:\s*(10\.\S+?)\.?$", entry, re.IGNORECASE)
    return m.group(1).rstrip(".") if m else None


def extract_title(entry: str) -> str:
    """The title is the sentence after the author list and before the journal name."""
    body = re.sub(r"^(.*?)\.\s+", "", entry, count=1)  # drop the author block
    parts = re.split(r"\.\s+", body)
    return parts[0] if parts else body


def first_surname(entry: str) -> str:
    return re.split(r"[\s,]+", entry.strip())[0]


def main() -> int:
    here = Path(__file__).resolve()
    root = here.parents[3]
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", type=Path, default=root / "manuscript_v2.md")
    ap.add_argument("--out", type=Path, default=root / "references_plant_communications.md")
    ap.add_argument("--report", type=Path, default=root / "supplementary" / "reference_restyle_report.txt")
    args = ap.parse_args()

    refs = parse_refs(args.manuscript.read_text())
    print(f"parsed {len(refs)} reference entries", file=sys.stderr)

    out_entries: list[tuple[str, str, str]] = []  # (sort key, formatted, status)
    report: list[str] = []

    for i, entry in enumerate(refs, 1):
        doi = extract_doi(entry)
        title = extract_title(entry)
        surname = first_surname(entry)
        rec = by_doi(doi) if doi else by_title(title, surname)
        time.sleep(0.3)

        status = "unresolved"
        formatted = entry
        if rec:
            cr_title = (rec.get("title") or [""])[0]
            sim = difflib.SequenceMatcher(
                None, norm_title(title), norm_title(cr_title)
            ).ratio()
            authors = rec.get("author") or []
            cr_surname = (authors[0].get("family") or "") if authors else ""
            yr = year_of(rec)
            m = re.search(r"\b(19|20)\d{2}\b", entry)
            claimed_year = int(m.group(0)) if m else None
            ok_title = sim >= 0.85
            ok_author = (not cr_surname) or cr_surname.lower().startswith(surname.lower()[:4])
            ok_year = claimed_year is None or yr is None or abs(yr - claimed_year) <= 1
            if ok_title and ok_author and ok_year:
                f = format_entry(rec)
                if f:
                    formatted, status = f, "crossref"
            else:
                status = (
                    f"rejected (title {sim:.2f}, author {surname}/{cr_surname}, "
                    f"year {claimed_year}/{yr})"
                )

        out_entries.append((surname.lower() + f"{i:03d}", formatted, status))
        report.append(f"[{i:02d}] {status}\n     OLD: {entry}\n     NEW: {formatted}\n")
        print(f"  [{i:02d}/{len(refs)}] {status:<12} {surname}", file=sys.stderr)

    out_entries.sort(key=lambda t: t[0])
    body = ["# References — Plant Communications (Molecular Plant) style\n",
            "Author lists, journal abbreviations, volumes, and page ranges were resolved "
            "against CrossRef; entries marked in the accompanying report as unresolved are "
            "carried over from the submitted manuscript unchanged and need a manual check.\n"]
    body += [e[1] + "\n" for e in out_entries]
    args.out.write_text("\n".join(body))

    n_ok = sum(1 for e in out_entries if e[2] == "crossref")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        f"{n_ok}/{len(out_entries)} entries resolved against CrossRef\n\n" + "\n".join(report)
    )
    print(f"\n{n_ok}/{len(out_entries)} resolved; wrote {args.out}", file=sys.stderr)
    print(f"report: {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
