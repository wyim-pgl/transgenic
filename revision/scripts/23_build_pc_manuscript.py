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

ABSTRACT = (
    "Accurate genome annotation is essential for deciphering biological systems, yet "
    "existing approaches frequently fail to detect the mRNA transcript isoforms generated "
    "by alternative splicing. Here we introduce TransGenic, a transformer-based framework "
    "that translates DNA sequence directly into standard gene annotation, including "
    "alternatively spliced isoforms. TransGenic was trained on nine diverse plant genomes "
    "and achieves a base-level F1 of 92.2% on held-out Arabidopsis thaliana gene models. In "
    "an extended benchmark across 13 plant species, de novo TransGenic predictions reached "
    "base-level accuracy comparable to recent deep-learning annotators (ANNEVO, Helixer, and "
    "Tiberius) when scored on single-gene loci, a per-locus setting that does not require "
    "gene-boundary resolution and therefore favors TransGenic; BUSCO analysis confirmed that "
    "prompted predictions recover 78.1–100% of conserved plant genes per species. When "
    "prompted with a primary transcript, TransGenic completed alternative isoforms with a "
    "transcript-level (exact intron-chain) recall of 61.9% and precision of 74.4% against "
    "TAIR10; against the AtRTD3 long-read transcriptome, base-level recall and precision for "
    "alternative transcripts were 59.4% and 77.3%. Structural validation showed that 85–98% "
    "of prompted transcripts across 13 species encode complete open reading frames, whereas "
    "unconstrained de novo output remains structurally inconsistent for most transcripts "
    "(8.9–27.4% valid ORFs), delimiting current use to locus annotation and prompt-based "
    "isoform completion. TransGenic thus combines the sequence representation of language "
    "models with generative prediction, providing a practical tool for isoform-aware gene "
    "model refinement in species for which transcript evidence is sparse."
)

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
Figure S4, all loci at which a prompted isoform matches AtRTD3 but not TAIR10.

**Supplemental tables.** Table S1, genome assemblies and annotations; Table S2, GFFCompare
benchmark across 13 species; Table S3, BUSCO completeness; Table S4, alternative-transcript-only
evaluation, transcript-level accuracy, and splice-event recovery, including the AUGUSTUS
posterior-sampling baseline (Tables S4a–c); Table S5, structural self-consistency; Table S6,
TSS/TES positional accuracy; Table S7, per-gene feature counts and GSF vocabulary coverage;
Table S8, alternative splicing content of the reference annotations; Table S9, whole-chromosome
pilot on A. thaliana chromosome 4; Table S10, software versions and command lines.

## Data and code availability

TransGenic source code, trained model weights (160M and 400M), example inputs and outputs,
and the complete evaluation pipeline used for every analysis reported here are publicly
available at https://github.com/wyim-pgl/transgenic. Pretrained models are additionally
hosted at https://huggingface.co/jlomas (HyenaTransgenic-768L12A6-400M and
HyenaTransgenic-512L9A4-160M). All genomes and annotations analysed in this study are
publicly available from the sources listed in Table S1; the AtRTD3 transcriptome
(atRTD3_TS_21Feb22_transfix) is available at https://ics.hutton.ac.uk/atRTD/RTD3/. The
summary tables underlying Figures 5, 6, and S1–S3 are provided as Supplemental Tables and
as machine-readable CSV files in the repository. A minimal command-line example is:

```bash
python src/run_genome_annotation.py genome.fa genes.gff3 -o output.gff3 --device cuda
```

Exact versions and command lines for every tool used are given in Table S10.

## Funding

*[TO BE COMPLETED BY THE AUTHORS — the submitted manuscript carried no funding statement.
Plant Communications requires the funding bodies and grant numbers that supported this work,
including any support for the GPU hardware and HPC allocation used for training and
inference.]*

## Author contributions

*[DRAFT — to be confirmed by all authors before submission.]* J.S.L. designed the model
architecture, implemented the training and inference code, and trained the models. F.R.
performed the cross-species benchmark and all evaluation analyses, including BUSCO,
GFFCompare, transcript- and splice-event-level isoform evaluation, structural
self-consistency, TSS/TES accuracy, and the AUGUSTUS posterior-sampling baseline. J.C.C. and
H.T. contributed to the interpretation of the results and revised the manuscript. W.C.Y.
conceived and supervised the study, acquired funding, and wrote the manuscript. All authors
read and approved the final manuscript.

## Acknowledgments

*[DRAFT — to be confirmed by the authors.]* We thank the three anonymous reviewers, whose
comments prompted the transcript- and event-level evaluation, the structural
self-consistency audit, and the AUGUSTUS sampling baseline reported here. Model training and
inference were performed on GPU workstations at the University of Nevada, Reno; genome
annotation benchmarks were run on the Pronghorn High-Performance Computing cluster at the
University of Nevada, Reno.

## Declaration of interests

The authors declare no competing interests.
"""


def main() -> int:
    text = SRC.read_text()

    # ---- abstract -------------------------------------------------------
    m = re.search(r"## Abstract\n\n(.+?)\n\n## Introduction", text, re.S)
    assert m, "could not locate the abstract block in manuscript_v2.md"
    old_abstract = m.group(1)
    assert len(old_abstract.split()) > 250, (
        "the source abstract is already within the limit; check whether this step is needed"
    )
    text = text.replace(
        f"## Abstract\n\n{old_abstract}\n\n## Introduction",
        f"## Abstract\n\n{ABSTRACT}\n\n{KEYWORDS}\n\n## Introduction",
        1,
    )

    # ---- in-text citations ---------------------------------------------
    for old, new in CITATION_FIXES:
        n = text.count(old)
        assert n > 0, f"citation {old!r} not found — has the text changed?"
        text = text.replace(old, new)
        print(f"  citation: {old} -> {new}  ({n} occurrence{'s' if n != 1 else ''})")

    # ---- move the availability statement out of Methods ----------------
    m = re.search(
        r"### Data and code availability\n\n.*?(?=\n## Author contributions)", text, re.S
    )
    assert m, "could not locate the Methods availability subsection"
    text = text[: m.start()] + text[m.end():]

    # ---- replace the placeholder back matter ---------------------------
    m = re.search(r"## Author contributions\n\n.*?(?=\n## Figures & Tables)", text, re.S)
    assert m, "could not locate the placeholder back-matter block"
    text = text[: m.start()] + BACK_MATTER + "\n" + text[m.end():]

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

    abstract_words = len(ABSTRACT.split())
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
