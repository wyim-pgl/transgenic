#!/usr/bin/env python3
"""Keep only the added structures that are structurally sound, using no reference at all.

WHY THIS EXISTS

Completion mode over-generates when the prompt's gene boundary differs from the one the
model was trained on. Measured on the 2026-08-06/07 frame experiment: prompted with TAIR10's
own CDS+UTR inside TAIR10's gene boundary it added 1,014 structures at 18.2% precision, and
prompted with the identical CDS+UTR inside Helixer's boundary it added 12,628 at 2.2% — the
same model, the same biology, twelve times the output. The extra structures are not a
different kind of mistake; they dilute a fixed number of correct ones.

Two properties separate them, and both are decidable from the genome and the predicted
coordinates alone:

    complete ORF        starts ATG, ends in a stop codon, length divisible by three, no
                        internal stop
    canonical introns   every intron begins GT and ends AG on the transcribed strand

Applied to that 12,628-structure run they remove 87.8% of the additions while keeping 262
of 275 correct ones — precision 2.2% -> 16.9%, which is where the manuscript's own
TAIR10-prompted figure sits (18.1%). Neither test consults TAIR10, AtRTD3 or any other
annotation, so this is a filter rather than a measurement of the answer: it can be applied
to a genome with no reference to score against, which is the only situation where a tool
like this is needed.

WHAT IT DOES NOT FIX

Precision recovers only where correct structures exist to be kept. On Helixer's own
annotation the same filter takes 2,462 additions down to 178 and the correct count stays at
1 (1/2,462 = 0.0% before, 1/178 = 0.6% after; helixer_filter.json,
helixer_filtered_as_additions.json) — the additions were not diluted, they were wrong to
begin with, because the prompt's UTR was wrong (at the 17,830 loci where Helixer's CDS
equals TAIR10's primary, 0 supplied UTRs match TAIR10 exactly). Filtering is worth applying when
the prompt is trustworthy and the coordinate frame is not; it is not a substitute for a
trustworthy prompt.

Usage:
    python 36_filter_additions_structurally.py --input PROMPT.gff3 --output COMPLETED.gff3 \
        --genome GENOME.fa --filtered OUT.gff3 [--json report.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENOME = ROOT / "revision" / "data" / "TAIR10" / "TAIR10_genome.fa"

_GM = re.compile(r"GM=([^;\s]+)")
_PARENT = re.compile(r"Parent=([^;]+)")
COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})

# TAIR distributes this genome under Ensembl sequence names while every annotation in this
# project uses TAIR names. A silent miss here is not hypothetical: it made every structure
# unscoreable and reported "0 complete ORFs", which reads as a finding rather than a failed
# lookup. Both spellings are registered so the join cannot quietly produce zero.
SEQID_ALIASES = {"1": "Chr1", "2": "Chr2", "3": "Chr3", "4": "Chr4", "5": "Chr5",
                 "Mt": "ChrM", "Pt": "ChrC"}


def load_genome(path: Path) -> dict[str, str]:
    """FASTA -> {name: sequence}, keyed by both the file's name and its annotation alias."""
    seqs: dict[str, str] = {}
    name, buf = None, []

    def store(seq_name: str, seq: str) -> None:
        seqs[seq_name] = seq
        alias = SEQID_ALIASES.get(seq_name)
        if alias:
            seqs[alias] = seq

    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    store(name, "".join(buf))
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name is not None:
        store(name, "".join(buf))
    if not seqs:
        raise ValueError(f"no sequences read from {path}")
    return seqs


def spliced_cds(chromosome: str, segments: tuple, strand: str) -> str:
    joined = "".join(chromosome[start - 1:end] for start, end in segments)
    if strand == "-":
        joined = joined.translate(COMPLEMENT)[::-1]
    return joined.upper()


def has_complete_orf(cds: str) -> bool:
    if len(cds) < 6 or len(cds) % 3:
        return False
    if not cds.startswith("ATG") or cds[-3:] not in STOP_CODONS:
        return False
    return not any(cds[i:i + 3] in STOP_CODONS for i in range(3, len(cds) - 3, 3))


def has_canonical_introns(chromosome: str, segments: tuple, strand: str) -> bool:
    """Every intron is GT..AG read on the transcribed strand. Single-exon passes."""
    for (_, exon_end), (next_start, _) in zip(segments, segments[1:]):
        intron = chromosome[exon_end:next_start - 1].upper()
        if len(intron) < 4:
            return False
        donor, acceptor = intron[:2], intron[-2:]
        if strand == "-":
            donor, acceptor = (acceptor.translate(COMPLEMENT)[::-1],
                               donor.translate(COMPLEMENT)[::-1])
        if donor != "GT" or acceptor != "AG":
            return False
    return True


def read_cds_by_transcript(path: Path) -> tuple[dict, dict, dict]:
    """-> {tx: sorted CDS tuple}, {tx: GM= or None}, {tx: (seqid, strand)}"""
    segments: dict[str, list] = defaultdict(list)
    gm: dict[str, str | None] = {}
    meta: dict[str, tuple] = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            parent = _PARENT.search(fields[8])
            if not parent:
                continue
            tx = parent.group(1)
            segments[tx].append((int(fields[3]), int(fields[4])))
            found = _GM.search(fields[8])
            gm[tx] = found.group(1) if found else None
            meta[tx] = (fields[0], fields[6])
    return ({tx: tuple(sorted(v)) for tx, v in segments.items()}, gm, meta)


def supplied_structures(prompt_gff: Path) -> dict:
    """gene -> the FIRST transcript's CDS tuple, which is what the decoder was prompted with."""
    first_of_gene: dict[str, str] = {}
    order: list = []
    segments: dict[str, list] = defaultdict(list)
    parent_of: dict[str, str] = {}
    with prompt_gff.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[2] == "gene":
                first_of_gene.setdefault(fields[8].split(";")[0].split("=")[1], "")
            elif fields[2] in ("mRNA", "transcript"):
                tid = re.search(r"ID=([^;]+)", fields[8])
                par = _PARENT.search(fields[8])
                if tid and par:
                    parent_of[tid.group(1)] = par.group(1)
                    if not first_of_gene.get(par.group(1)):
                        first_of_gene[par.group(1)] = tid.group(1)
                    order.append(tid.group(1))
            elif fields[2] == "CDS":
                par = _PARENT.search(fields[8])
                if par:
                    segments[par.group(1)].append((int(fields[3]), int(fields[4])))
    return {gene: tuple(sorted(segments[tx]))
            for gene, tx in first_of_gene.items() if tx and segments.get(tx)}


def filter_predictions(prompt_gff: Path, completed_gff: Path, genome_path: Path,
                       out_gff: Path | None) -> dict:
    genome = load_genome(genome_path)
    supplied = supplied_structures(prompt_gff)
    structures, gm_of, meta_of = read_cds_by_transcript(completed_gff)
    if not structures:
        raise ValueError(f"no CDS-bearing transcripts in {completed_gff} — refusing to "
                         "report a filter result on an empty file")

    keep: set[str] = set()
    stats = {"transcripts": 0, "prompt_re_emissions": 0, "additions": 0,
             "additions_kept": 0, "failed_orf": 0, "failed_splice": 0,
             "unknown_seqid": 0}
    seen_addition: set[tuple] = set()

    for tx, struct in structures.items():
        stats["transcripts"] += 1
        gene = gm_of.get(tx)
        if gene is not None and supplied.get(gene) == struct:
            stats["prompt_re_emissions"] += 1
            keep.add(tx)  # the supplied model is retained by design, never filtered
            continue

        # Counters are per DISTINCT structure so they line up with the precision figures,
        # which collapse identical emissions; `keep` is per transcript, because the file
        # being rewritten has one row set per transcript.
        key = (gene, struct)
        first_sighting = key not in seen_addition
        if first_sighting:
            seen_addition.add(key)
            stats["additions"] += 1

        seqid, strand = meta_of[tx]
        chromosome = genome.get(seqid)
        if chromosome is None:
            if first_sighting:
                stats["unknown_seqid"] += 1
            continue
        if not has_complete_orf(spliced_cds(chromosome, struct, strand)):
            if first_sighting:
                stats["failed_orf"] += 1
            continue
        if not has_canonical_introns(chromosome, struct, strand):
            if first_sighting:
                stats["failed_splice"] += 1
            continue

        keep.add(tx)
        if first_sighting:
            stats["additions_kept"] += 1

    if out_gff is not None:
        write_filtered(completed_gff, keep, out_gff)
    stats["additions_discarded_pct"] = (
        round(100 * (1 - stats["additions_kept"] / stats["additions"]), 1)
        if stats["additions"] else None)
    return stats


def write_filtered(completed_gff: Path, keep: set, out_gff: Path) -> None:
    """Emit gene/mRNA/child rows for kept transcripts only, dropping emptied genes."""
    kept_genes: set[str] = set()
    with completed_gff.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] not in ("mRNA", "transcript"):
                continue
            tid = re.search(r"ID=([^;]+)", fields[8])
            par = _PARENT.search(fields[8])
            if tid and par and tid.group(1) in keep:
                kept_genes.add(par.group(1))

    with completed_gff.open() as src, out_gff.open("w") as dst:
        dst.write("##gff-version 3\n")
        for line in src:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            kind, attrs = fields[2], fields[8]
            if kind == "gene":
                if attrs.split(";")[0].split("=")[1] in kept_genes:
                    dst.write(line)
            elif kind in ("mRNA", "transcript"):
                tid = re.search(r"ID=([^;]+)", attrs)
                if tid and tid.group(1) in keep:
                    dst.write(line)
            else:
                par = _PARENT.search(attrs)
                if par and par.group(1) in keep:
                    dst.write(line)


def orf_audit(annotation_gff: Path, genome_path: Path) -> dict:
    """Fraction of an annotation's transcripts that pass the two filter criteria.

    The filter's own control. Both predicates are cheap to get wrong in a way that returns
    zero for everything rather than raising: the first run of this filter reported no
    complete ORFs at all, because the genome FASTA used Ensembl sequence names (`1`, `2`,
    `Mt`) while the annotation used TAIR names (`Chr1`, `ChrM`), so every lookup missed.
    Nothing in the output distinguished that from a genuinely terrible prediction set.

    Running the same predicates over a curated annotation is what separates the two: on
    TAIR10 the complete-ORF fraction must land near the 92.9-100% the manuscript reports
    for reference annotations (Table S5). A number far below that means the checker is
    broken, not the annotation. Report a filtered set's numbers only alongside this.
    """
    genome = load_genome(genome_path)
    segments, _, meta = read_cds_by_transcript(annotation_gff)

    total = orf_ok = introns_ok = both_ok = missing_seqid = 0
    for tx, segs in segments.items():
        seqid, strand = meta[tx]
        chromosome = genome.get(seqid)
        if chromosome is None:
            missing_seqid += 1
            continue
        total += 1
        orf = has_complete_orf(spliced_cds(chromosome, segs, strand))
        introns = has_canonical_introns(chromosome, segs, strand)
        orf_ok += orf
        introns_ok += introns
        both_ok += orf and introns

    pct = lambda n: round(100 * n / total, 1) if total else 0.0
    return {
        "annotation": str(annotation_gff),
        "transcripts_scored": total,
        "transcripts_skipped_unknown_seqid": missing_seqid,
        "complete_orf": orf_ok,
        "complete_orf_pct": pct(orf_ok),
        "canonical_introns": introns_ok,
        "canonical_introns_pct": pct(introns_ok),
        "both": both_ok,
        "both_pct": pct(both_ok),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, help="the prompt annotation")
    ap.add_argument("--output", type=Path, help="completion-mode output")
    ap.add_argument("--genome", type=Path, default=DEFAULT_GENOME)
    ap.add_argument("--filtered", type=Path, default=None, help="where to write the kept GFF3")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--orf-audit", type=Path, default=None, metavar="ANNOTATION.gff3",
                    help="score one annotation against both criteria and stop; the control "
                         "that shows the checker works before any filtered result is quoted")
    args = ap.parse_args(argv)

    if args.orf_audit:
        stats = orf_audit(args.orf_audit, args.genome)
    else:
        if not args.input or not args.output:
            ap.error("--input and --output are required unless --orf-audit is given")
        stats = filter_predictions(args.input, args.output, args.genome, args.filtered)

    width = max(len(k) for k in stats)
    for key, value in stats.items():
        print(f"  {key:<{width}}  {value}")
    if args.json:
        args.json.write_text(json.dumps(stats, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
