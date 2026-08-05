#!/usr/bin/env python3
"""Assemble the Plant Communications submission version of the manuscript.

Takes `manuscript_v2.md` as the source of the scientific text and applies the
journal-format changes that do not touch the science:

  * abstract shortened to the journal's 250-word limit
  * key words added
  * three in-text citations updated to the print versions of their references
    (Holst 2025 -> 2026, Guo 2021 -> 2022, Vogel 2010 -> the consortium)
  * back matter reordered into Plant Communications order and the placeholder
    sections replaced with drafts the authors can confirm
  * the reference list replaced with the CrossRef-resolved, Cell Press-styled list
  * figure legends and supplemental item lists kept together at the end

The body text of Introduction, Results, Discussion, and Methods is copied through
verbatim so that no wording change can slip in unnoticed. Every substitution is
asserted, so the script fails loudly if the source text has moved on.

Usage:
    python 23_build_pc_manuscript.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "manuscript_v2.md"
REFS = ROOT / "references_plant_communications.md"
DST = ROOT / "manuscript_v3_plant_communications.md"

KEYWORDS = (
    "**Key words:** genome annotation; alternative splicing; transcript isoform; "
    "transformer; deep learning; gene structure prediction"
)

CITATION_FIXES = [
    ("Holst et al., 2025", "Holst et al., 2026"),
    ("Guo et al., 2021", "Guo et al., 2022"),
    ("(Vogel et al., 2010)", "(The International Brachypodium Initiative, 2010)"),
]

BACK_MATTER = """## Supplemental information

Supplemental information is available at *Plant Communications* Online.

**Supplemental figures.** Figure S1, functional completeness (BUSCO) across 13 species;
Figure S2, TSS and TES positional accuracy; Figure S3, GSF vocabulary coverage;
Figure S4, six representative loci at which a prompted isoform matches AtRTD3 but not TAIR10.

**Supplemental tables.** Table S1, genome assemblies and annotations; Table S2, GFFCompare
benchmark across 13 species; Table S3, BUSCO completeness; Table S4, alternative-transcript-only
evaluation, transcript-level accuracy, and splice-event recovery, including the AUGUSTUS
posterior-sampling baseline (Tables S4a–c); Table S5, structural self-consistency; Table S6,
TSS/TES positional accuracy; Table S7, per-gene feature counts and GSF vocabulary coverage;
Table S8, alternative splicing content of the reference annotations; Table S9, whole-chromosome
pilot on A. thaliana chromosome 4; Table S10, software versions and command lines.

{availability}

{author_sections}
"""


def main() -> int:
    text = SRC.read_text()

    # ---- abstract -------------------------------------------------------
    # The abstract comes from manuscript_v2.md. It used to be a constant in this file,
    # which meant every edit to v2's abstract was silently discarded — found 2026-08-04,
    # after a tone pass and a back-matter rewrite had both gone missing from the built
    # manuscript without a single check failing.
    m = re.search(r"## Abstract\n\n(.+?)\n\n## Introduction", text, re.S)
    assert m, "could not locate the abstract block in manuscript_v2.md"
    abstract = m.group(1).strip()
    n_words = len(abstract.split())
    assert n_words <= 250, (
        f"the abstract in manuscript_v2.md is {n_words} words; Plant Communications allows "
        f"250. Shorten it in v2 - this script no longer carries a separate copy."
    )
    text = text.replace(
        f"## Abstract\n\n{m.group(1)}\n\n## Introduction",
        f"## Abstract\n\n{abstract}\n\n{KEYWORDS}\n\n## Introduction",
        1,
    )

    # ---- in-text citations ---------------------------------------------
    for old, new in CITATION_FIXES:
        n = text.count(old)
        assert n > 0, f"citation {old!r} not found — has the text changed?"
        text = text.replace(old, new)
        print(f"  citation: {old} -> {new}  ({n} occurrence{'s' if n != 1 else ''})")

    # ---- move the availability statement out of Methods ----------------
    m = re.search(r"### Data and code availability\n\n(.*?)(?=\n## Funding)", text, re.S)
    assert m, "could not locate the Methods availability subsection"
    availability = "## Data and code availability\n\n" + m.group(1).strip()
    text = text[: m.start()] + text[m.end():]

    # ---- replace the placeholder back matter ---------------------------
    # Everything the authors write - funding, contributions, acknowledgments, competing
    # interests - is carried through from v2. Only the journal-specific blocks with no v2
    # counterpart (supplemental-item lists, the relocated availability statement) live in
    # BACK_MATTER.
    m = re.search(r"(## Funding\n\n.*?)(?=\n## Figures & Tables)", text, re.S)
    assert m, ("could not locate the author-written back matter in manuscript_v2.md; "
               "it must run from '## Funding' to '## Figures & Tables'")
    author_sections = m.group(1).rstrip()
    for heading in ("## Funding", "## Author contributions", "## Acknowledgments"):
        assert heading in author_sections, f"{heading} missing from manuscript_v2.md"
    author_sections = author_sections.replace(
        "## Declaration of competing interests", "## Declaration of interests")
    text = (text[: m.start()]
            + BACK_MATTER.format(availability=availability, author_sections=author_sections)
            + "\n" + text[m.end():])

    # ---- swap in the curated reference list -----------------------------
    ref_text = REFS.read_text()
    body = ref_text.split("---\n", 1)[1].split("\n---\n", 1)[0].strip()
    m = re.search(r"## References\n\n.*\Z", text, re.S)
    assert m, "could not locate the reference list"
    text = text[: m.start()] + "## References\n\n" + body + "\n"

    # ---- split the display-item block -----------------------------------
    # Main figure legends stay with the manuscript; the supplemental figure
    # legends and the supplemental table stubs belong to the supplemental file,
    # where the tables now carry their full data.
    m = re.search(r"## Figures & Tables\n\n(.*?)(?=\n## References)", text, re.S)
    assert m, "could not locate the display-item block"
    display = m.group(1).rstrip()
    text = text[: m.start()] + text[m.end():]

    split_at = display.index("**Figure S1.")
    main_figs = display[:split_at].rstrip()
    supp_items = display[split_at:].rstrip()
    assert main_figs.count("**Figure ") == 6, "expected six main figure legends"

    # The self-prompted configurations appear in the benchmark, BUSCO, and
    # self-consistency tables, not only in Table S5.
    old = "they are characterized separately in Table S5."
    assert old in main_figs
    main_figs = main_figs.replace(
        old, "they are characterized separately in Tables S2, S3, and S5."
    )

    text = text.rstrip() + "\n\n## Figure legends\n\n" + main_figs + "\n"

    supp_split = supp_items.index("**Table S1.")
    supp_figs = supp_items[:supp_split].rstrip()
    supp_tables_stub = supp_items[supp_split:].rstrip()

    # The supplemental figure legends travel with the supplemental document, which
    # `20_build_supplementary_tables.py` assembles; only the superseded table
    # legends are kept here, as a check that no legend text was lost.
    n_supp_figs = supp_figs.count("**Figure S")
    assert n_supp_figs == 4, f"expected four supplemental figure legends, found {n_supp_figs}"
    supp_dir = ROOT / "supplementary"
    supp_dir.mkdir(exist_ok=True)
    (supp_dir / "supplemental_table_legends_superseded.md").write_text(
        "# Supplemental table legends as submitted (superseded)\n\n"
        "These are the placeholder legends from `manuscript_v2.md`. The submission "
        "versions, with all data populated, are in `supplemental_information.md` and the "
        "accompanying `TableS*.csv` files; keep this file only to check that no legend "
        "text was lost in the transfer.\n\n" + supp_tables_stub + "\n"
    )

    DST.write_text(text)

    abstract_words = n_words
    main_text = re.search(
        r"## Introduction\n(.*?)\n## Supplemental information", text, re.S
    )
    print(f"\nwrote {DST}")
    print(f"  abstract:  {abstract_words} words (limit 250)")
    if main_text:
        print(f"  main text: {len(main_text.group(1).split()):,} words "
              f"(Introduction through Methods)")
    print(f"  references: {sum(1 for ln in body.splitlines() if ln.strip())} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
