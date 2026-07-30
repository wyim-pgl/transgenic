#!/usr/bin/env python3
"""Cross-check in-text citations against the reference list.

Reports citations with no matching reference entry and reference entries that are
never cited. Matching is on (first-author surname, year); "and"/"et al." forms and
a/b year suffixes are handled.

Usage:
    python 22_check_citations.py [--manuscript PATH] [--references PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (Surname[ and Surname| et al.], 2020a) — also handles multiple refs in one paren.
CITE = re.compile(
    r"\(([^()]*?\b(?:19|20)\d{2}[a-z]?(?:\s*;[^()]*?\b(?:19|20)\d{2}[a-z]?)*)\)"
)
# Corporate authors ("The Tomato Genome Consortium") must be matched before the
# personal-name branch, and greedily, or only their first word is captured.
ONE = re.compile(
    r"(The\s+[A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+)*"
    r"|(?:de|van|von|der|del|di|la|le)\s+[A-Z][\w'’\-]+"
    r"|[A-Z][\w'’\-]+"
    r")((?:\s+(?:and|&)\s+[A-Z][\w'’\-]+)|(?:\s+et\s+al\.))?,\s*((?:19|20)\d{2}[a-z]?)"
)

# Nobiliary particles are dropped so that the in-text "(de Almeida et al., 2024)" and
# the list entry "de Almeida, B.P., ... (2024)." reduce to the same key.
PARTICLE = re.compile(r"^(?:de|van|von|der|del|di|la|le)\s+", re.I)


def sort_key(name: str) -> str:
    """Reduce an author string to the token the two sides can be compared on."""
    name = name.strip().rstrip(",")
    if name.lower().startswith("the "):
        return name.lower()
    name = PARTICLE.sub("", name)
    return re.split(r"\s+", name)[0].lower()


def parse_intext(text: str) -> set[tuple[str, str]]:
    body = text.split("## References", 1)[0]
    out: set[tuple[str, str]] = set()
    for m in CITE.finditer(body):
        for chunk in m.group(1).split(";"):
            for mm in ONE.finditer(chunk.strip()):
                out.add((sort_key(mm.group(1)), mm.group(3)))
    # Table cells carry citations bare rather than parenthesised, e.g. the Reference
    # column of Supplemental Table S1.
    for line in body.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        for cell in line.split("|"):
            for mm in ONE.finditer(cell.strip()):
                out.add((sort_key(mm.group(1)), mm.group(3)))
    return out


def parse_reflist(text: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "|", "-", "*", ">")):
            continue
        m = re.match(
            r"^(The\s+[A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+)*"
            r"|(?:de|van|von|der|del|di|la|le)\s+[A-Z][\w'’\-]+"
            r"|[A-Za-z][\w'’\-]*"
            r"),?.*?\(((?:19|20)\d{2}[a-z]?)\)",
            line,
        )
        if m:
            out.add((sort_key(m.group(1)), m.group(2)))
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", type=Path, default=root / "manuscript_v2.md")
    ap.add_argument("--references", type=Path,
                    default=root / "references_plant_communications.md")
    # Supplemental Table S1 cites the genome papers and lives outside the manuscript,
    # so scan it too or those twelve entries look uncited.
    ap.add_argument("--also", type=Path, nargs="*",
                    default=[root / "supplementary" / "supplemental_information.md"],
                    help="further files whose in-text citations count as cited")
    args = ap.parse_args()

    cited = parse_intext(args.manuscript.read_text())
    for extra in args.also:
        if extra.exists():
            cited |= parse_intext(extra.read_text())
        else:
            print(f"note: {extra} not found, skipping")
    listed = parse_reflist(args.references.read_text())

    missing = sorted(cited - listed)
    unused = sorted(listed - cited)

    print(f"in-text citations (unique):  {len(cited)}")
    print(f"reference-list entries:      {len(listed)}")
    print()
    if missing:
        print(f"CITED BUT NOT IN THE LIST ({len(missing)}):")
        for a, y in missing:
            print(f"  {a} {y}")
    else:
        print("every in-text citation has a reference entry")
    print()
    if unused:
        print(f"IN THE LIST BUT NEVER CITED ({len(unused)}):")
        for a, y in unused:
            print(f"  {a} {y}")
    else:
        print("every reference entry is cited")
    return len(missing)


if __name__ == "__main__":
    sys.exit(main())
