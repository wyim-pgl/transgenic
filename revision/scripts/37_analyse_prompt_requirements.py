#!/usr/bin/env python3
"""Measure what a completion-mode prompt has to get right, holding the rest fixed.

The nine-arm frame experiment (2026-08-07) established that prompting with TAIR10 scores
18.2% against TAIR10's alternative transcripts while prompting with Helixer or ANNEVO scores
0.0%. This script produces the three measurements that located the cause, each restricted to
loci where the supplied CDS ALREADY equals a real TAIR10 transcript, so CDS quality is held
fixed and only the quantity under test varies.

    utr-accuracy    Does the tool's UTR match the reference's, and does that predict whether
                    the model adds anything correct?
                    Result on Helixer: 0 of 17,830 loci have an exactly correct UTR; those
                    loci score 0.1%.

    utr-closeness   Is the UTR effect graded or all-or-nothing? Buckets loci by how many
                    bases separate the supplied UTR from the reference's.
                    Result: 1-10 bp scores 0.0%, the same as >1000 bp. All-or-nothing, so
                    improving UTR prediction pays nothing until it is exactly right.

    generated-utr   When given CDS with no UTR, is the UTR the model invents correct? A
                    correct one would open a two-pass route (CDS -> generated UTR ->
                    re-prompt) into the 18.2% condition.
                    Result: it adds UTRs to 82.7% of prompt-preserving transcripts, 3.0% of
                    them exactly right — and the invention itself collapses when the CDS is a
                    tool's rather than the reference's (44,337 UTR rows from TAIR10 CDS,
                    175 from Helixer's, 138 from ANNEVO's).

Definitions match `34_score_as_additions.py`: a structure is a tuple of sorted CDS
coordinates, an addition is an emitted structure differing from the one supplied for that
locus, and the reference is TAIR10's alternative transcripts with the primary removed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "revision" / "data"
TAIR10_GTF = DATA / "TAIR10" / "TAIR10.gtf"
PRIMARY_IDS = DATA / "TAIR10" / "primary_transcript_ids.txt"

_GM = re.compile(r"GM=([^;\s]+)")
_PARENT = re.compile(r"Parent=([^;]+)")
_ID = re.compile(r"ID=([^;]+)")
_GTF_GENE = re.compile(r'gene_id "([^"]+)"')
_GTF_TX = re.compile(r'transcript_id "([^"]+)"')


def agi(value: str) -> str:
    """`AT1G01010.TAIR10` and `AT1G01010.1` both denote gene AT1G01010."""
    return value.split(".")[0].upper()


def utr_outside(exons: list, cds: tuple) -> tuple:
    """Exonic intervals lying outside the CDS span. TAIR10's GTF has no UTR rows, so the
    UTRs the model was prompted with are derived exactly this way."""
    if not cds or not exons:
        return ()
    low, high = cds[0][0], cds[-1][1]
    segments = []
    for start, end in sorted(exons):
        if end < low or start > high:
            segments.append((start, end))
        else:
            if start < low:
                segments.append((start, low - 1))
            if end > high:
                segments.append((high + 1, end))
    return tuple(sorted(segments))


def load_reference(gtf: Path, primary_ids_path: Path) -> dict:
    """gene -> {"cds", "utr", "alternatives"} for the primary transcript."""
    primary_ids = {line.strip() for line in primary_ids_path.read_text().splitlines()
                   if line.strip()}
    cds: dict = defaultdict(lambda: defaultdict(list))
    exon: dict = defaultdict(lambda: defaultdict(list))
    with gtf.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in ("CDS", "exon"):
                continue
            gene, tx = _GTF_GENE.search(f[8]), _GTF_TX.search(f[8])
            if not (gene and tx):
                continue
            target = cds if f[2] == "CDS" else exon
            target[agi(gene.group(1))][tx.group(1)].append((int(f[3]), int(f[4])))

    out = {}
    for gene, transcripts in cds.items():
        structures = {tx: tuple(sorted(v)) for tx, v in transcripts.items()}
        primary_tx = next((tx for tx in structures if tx in primary_ids), None)
        if primary_tx is None:  # no curated primary: longest stands in, as 32/34 do
            primary_tx = max(structures, key=lambda tx: len(structures[tx]))
        primary = structures[primary_tx]
        out[gene] = {
            "cds": primary,
            "utr": utr_outside(sorted(exon[gene].get(primary_tx, [])), primary),
            "alternatives": {s for s in structures.values() if s != primary},
        }
    return out


def load_prompt(gff: Path) -> dict:
    """gene -> {"cds", "utr"} for the FIRST transcript, which is what is prompted.

    UTR rows are used when present; otherwise they are derived from exon minus CDS, which is
    what the preprocessing does for an annotation that carries exons but no UTR rows.
    """
    first_of_gene: dict = {}
    cds: dict = defaultdict(list)
    exon: dict = defaultdict(list)
    utr: dict = defaultdict(list)
    with gff.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            kind = f[2]
            if kind == "gene":
                first_of_gene.setdefault(f[8].split(";")[0].split("=")[1], None)
            elif kind in ("mRNA", "transcript"):
                tid, parent = _ID.search(f[8]), _PARENT.search(f[8])
                if tid and parent and first_of_gene.get(parent.group(1)) is None:
                    first_of_gene[parent.group(1)] = tid.group(1)
            else:
                parent = _PARENT.search(f[8])
                if not parent:
                    continue
                segment = (int(f[3]), int(f[4]))
                if kind == "CDS":
                    cds[parent.group(1)].append(segment)
                elif kind == "exon":
                    exon[parent.group(1)].append(segment)
                elif kind.endswith("UTR"):
                    utr[parent.group(1)].append(segment)

    out = {}
    for gene, tx in first_of_gene.items():
        if not tx or not cds.get(tx):
            continue
        structure = tuple(sorted(cds[tx]))
        out[gene] = {
            "cds": structure,
            "utr": (tuple(sorted(utr[tx])) if utr.get(tx)
                    else utr_outside(sorted(exon.get(tx, [])), structure)),
        }
    return out


def load_emitted(gff: Path) -> tuple[dict, dict]:
    """-> {GM=: {structures}}, {(GM=, structure): utr tuple}"""
    tx_cds: dict = defaultdict(list)
    tx_utr: dict = defaultdict(list)
    tx_gm: dict = {}
    with gff.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            gm, parent = _GM.search(f[8]), _PARENT.search(f[8])
            if not (gm and parent):
                continue
            tx = parent.group(1)
            if f[2] == "CDS":
                tx_cds[tx].append((int(f[3]), int(f[4])))
            elif f[2].endswith("UTR"):
                tx_utr[tx].append((int(f[3]), int(f[4])))
            tx_gm[tx] = gm.group(1)

    structures: dict = defaultdict(set)
    utr_of: dict = {}
    for tx, segments in tx_cds.items():
        structure = tuple(sorted(segments))
        gm = tx_gm[tx]
        structures[gm].add(structure)
        utr_of.setdefault((gm, structure), tuple(sorted(tx_utr.get(tx, []))))
    return structures, utr_of


def _pct(numerator: int, denominator: int):
    return round(100 * numerator / denominator, 1) if denominator else None


def utr_accuracy(prompt, emitted, reference) -> dict:
    """Split loci with a correct CDS by whether the supplied UTR is also correct."""
    by_cds = {}
    for gene, info in reference.items():
        by_cds.setdefault(info["cds"], gene)

    groups = {"utr_correct": [0, 0, 0], "utr_wrong": [0, 0, 0]}  # loci, additions, correct
    for gene, supplied in prompt.items():
        ref_gene = by_cds.get(supplied["cds"])
        if ref_gene is None:
            continue
        ref = reference[ref_gene]
        key = "utr_correct" if supplied["utr"] == ref["utr"] else "utr_wrong"
        additions = {s for s in emitted.get(gene, set()) if s != supplied["cds"]}
        groups[key][0] += 1
        groups[key][1] += len(additions)
        groups[key][2] += len(additions & ref["alternatives"])

    return {name: {"loci": n, "additions": a, "correct": c, "precision_pct": _pct(c, a)}
            for name, (n, a, c) in groups.items()}


def utr_closeness(prompt, emitted, reference) -> dict:
    """Bucket correct-CDS loci by how far the supplied UTR is from the reference's."""
    def covered(segments):
        out = set()
        for start, end in segments:
            out.update(range(start, end + 1))
        return out

    by_cds = {}
    for gene, info in reference.items():
        by_cds.setdefault(info["cds"], gene)

    buckets = [(0, 0), (1, 10), (11, 50), (51, 200), (201, 1000), (1001, 10 ** 9)]
    tally = {b: [0, 0, 0] for b in buckets}
    for gene, supplied in prompt.items():
        ref_gene = by_cds.get(supplied["cds"])
        if ref_gene is None:
            continue
        ref = reference[ref_gene]
        distance = len(covered(supplied["utr"]) ^ covered(ref["utr"]))
        additions = {s for s in emitted.get(gene, set()) if s != supplied["cds"]}
        for low, high in buckets:
            if low <= distance <= high:
                t = tally[(low, high)]
                t[0] += 1
                t[1] += len(additions)
                t[2] += len(additions & ref["alternatives"])
                break

    out = {}
    for (low, high), (n, a, c) in tally.items():
        label = "exact" if high == 0 else (f"{low}-{high}" if high < 10 ** 9 else f">{low - 1}")
        out[label] = {"loci": n, "additions": a, "correct": c, "precision_pct": _pct(c, a)}
    return out


def generated_utr(prompt, emitted, utr_of, reference) -> dict:
    """Of transcripts that kept the prompted CDS, how many carry a UTR, and is it right?"""
    by_cds = {}
    for gene, info in reference.items():
        by_cds.setdefault(info["cds"], gene)

    kept = with_utr = exact = 0
    for gene, supplied in prompt.items():
        if supplied["cds"] not in emitted.get(gene, set()):
            continue
        ref_gene = by_cds.get(supplied["cds"])
        if ref_gene is None:
            continue
        kept += 1
        generated = utr_of.get((gene, supplied["cds"]), ())
        if not generated:
            continue
        with_utr += 1
        if generated == reference[ref_gene]["utr"]:
            exact += 1
    return {"prompt_preserving_transcripts": kept,
            "carried_a_generated_utr": with_utr,
            "carried_pct": _pct(with_utr, kept),
            "generated_utr_exactly_correct": exact,
            "exact_pct_of_those_carrying": _pct(exact, with_utr)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True,
                    choices=["utr-accuracy", "utr-closeness", "generated-utr"])
    ap.add_argument("--input", type=Path, required=True, help="the prompt annotation")
    ap.add_argument("--output", type=Path, required=True, help="completion-mode output")
    ap.add_argument("--tair10", type=Path, default=TAIR10_GTF)
    ap.add_argument("--primary-ids", type=Path, default=PRIMARY_IDS)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    reference = load_reference(args.tair10, args.primary_ids)
    prompt = load_prompt(args.input)
    emitted, utr_of = load_emitted(args.output)
    for name, d in (("reference", reference), ("prompt", prompt), ("output", emitted)):
        if not d:
            raise ValueError(f"{name} produced no loci — refusing to report zeros that are "
                             "a parsing failure rather than a measurement")

    if args.mode == "utr-accuracy":
        result = utr_accuracy(prompt, emitted, reference)
    elif args.mode == "utr-closeness":
        result = utr_closeness(prompt, emitted, reference)
    else:
        result = generated_utr(prompt, emitted, utr_of, reference)

    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
