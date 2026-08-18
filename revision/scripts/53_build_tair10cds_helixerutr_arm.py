#!/usr/bin/env python3
"""Build the missing cell of the prompt-transfer factorial: reference CDS, predicted UTR.

THE FACTORIAL AND WHAT IS MISSING FROM IT

Completion mode adds a correct TAIR10 alternative at 18.2% of its additions when prompted
with TAIR10 itself, and at 0.0-0.2% when prompted with Helixer or ANNEVO. Three of the four
cells that decompose "CDS source x UTR source" have been run:

    reference CDS + reference UTR   `tair10selfutr`                     18.2%
    reference CDS + no UTR          `tair10self`                         4.6%
    predicted CDS + reference UTR   `helixertairutr` / `annevotairutr`   0.9-1.2%

The fourth — reference CDS carrying a PREDICTED UTR — is what this script builds. It is the
cell that separates two explanations the other three cannot tell apart:

  * UTR ACCURACY. If a wrong UTR is nearly as damaging as a wrong CDS, this cell lands near
    1% and the gap is about getting UTRs right (long reads, a dedicated UTR caller).
  * FORMAT / OOD. If the model mostly needs UTR tokens to be PRESENT and in a plausible
    place, this cell stays high (>=5%) and a large part of the gap is the prompt's shape
    rather than its correctness — which is fixable without new evidence.

The reading is calibrated against the 4.6% no-UTR arm, not against zero: a wrong UTR that
scores below 4.6% is actively worse than omitting the UTR, and one above it still helps.

CONSTRUCTION

The locus set, the CDS rows, and the gene row are taken unchanged from `tair10selfutr`
(arm E, the 18.2% cell), so this arm differs from that one in exactly one factor: which
annotation the UTR intervals came from. Concretely:

    locus set   TAIR10 genes paired to a Helixer gene whose span contains the TAIR10
                primary CDS, restricted to those whose primary transcript has exon rows —
                `pair_and_filter` from `35_build_frame_shift_prompts.py`, reused verbatim.
    CDS rows    TAIR10's primary transcript, byte-identical to arm E.
    gene/mRNA   TAIR10's own locus span, widened only where a donated UTR falls outside it
                (a feature outside the gene row would fall outside the encoder window that
                row builds). Widening is counted and reported.
    UTR rows    the paired Helixer gene's first mRNA, clipped to fall outside the TAIR10
                CDS span so no donated interval can overlap or alter the CDS.

WHY THE GENE ROW IS WIDENED RATHER THAN HELD EXACTLY AT TAIR10'S

Holding the gene row exactly at TAIR10's span — clipping any donated UTR that overruns it —
was measured and rejected. TAIR10's gene row spans every exon of every transcript at the
locus, so it already ENCODES the reference UTR extent; clipping a donated interval to it
therefore snaps that interval's outer endpoint onto the reference's. Measured over these
24,210 loci, clipping leaves the donated 5' UTR exactly equal to the reference 5' UTR at
16,349 loci and the 3' UTR exactly equal at 15,432, and strips the UTR entirely from 4,135.
A strictly-TAIR10-boundary version of this cell would thus be two thirds arm E and one sixth
arm A, and could not measure the factor it exists to measure.

That is not a defect of the construction but a property of the design: the boundary is a
FUNCTION of the UTR, so "reference boundary" and "predicted UTR" are not independently
settable. It is also why Helixer's gene boundaries agree with TAIR10's at 1 locus in 25,657
— the boundary disagrees because the UTR does. Widening is the minimal concession: the gene
row stays exactly arm E's at 2,701 loci, is never narrower than arm E's at any locus, and
moves by a median of 77 bp (mean 186) where it moves at all. For scale, the deliberate
frame perturbation in `tair10helixerframeutr` moves it by a median of 113 bp.

The residual confound is real and must be carried into any reading of this arm: a low score
here is "wrong UTR, plus the frame shift a wrong UTR implies", not "wrong UTR alone". A HIGH
score is the unconfounded result, since the frame moved and the score survived anyway.

WHY THE DONATION IS NOT COPIED FROM ARM G

`35_build_frame_shift_prompts.py:write_helixer_with_tair_utr` (arm G, the 0.9-1.2% cell) is
the mirror image of this donation and it is wrong on the minus strand. `utr_segments`
returns its two lists labelled by STRAND — on the minus strand the 5' list is the
high-coordinate side — but arm G then filters them by COORDINATE (`five` kept only where
`s < lo`, `three` only where `e > hi`). On a minus-strand locus both tests fail for every
interval, so arm G donated a UTR to 35 of 11,989 minus-strand loci against 10,001 of 11,972
plus-strand ones. Reproducing that here for the sake of symmetry would put half this arm's
loci in the no-UTR cell instead of the predicted-UTR cell, which is the one thing this
experiment must not do.

So the donation here assigns 5'/3' from side AND strand, the same way `utr_segments`
derives them for the reference arms. Helixer's own row labels are re-derived rather than
carried over, because a label is only meaningful relative to the CDS it flanks, and the CDS
these intervals now flank is TAIR10's, not Helixer's.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "revision" / "scripts"
BENCH = ROOT.parent / "polishing_benchmark"

ARM_NAME = "tair10cdshelixerutr"


@lru_cache(maxsize=1)
def _load_arm_builder():
    """Import `35_build_frame_shift_prompts.py` (a module name starting with a digit).

    Reused rather than reimplemented so this arm's locus set is the same object as arm E's
    by construction: any divergence in pairing would show up as a score difference that has
    nothing to do with the UTR source.
    """
    path = SCRIPTS / "35_build_frame_shift_prompts.py"
    spec = importlib.util.spec_from_file_location("build_frame_shift_prompts", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass/annotation resolution needs it registered
    spec.loader.exec_module(module)
    return module


def load_helixer_first_transcript_utr(gff: Path) -> dict:
    """gene -> {"seq", "strand", "utr": [(start, end)]} for Helixer's FIRST mRNA.

    First-mRNA rather than best-mRNA because that is the transcript `load_helixer_first_cds`
    treats as the prompt in every other arm; Helixer emits one per locus anyway.

    The 5'/3' distinction Helixer wrote is dropped here on purpose — see the module
    docstring. Only the intervals survive, and their side is re-derived against TAIR10's CDS.
    """
    import re

    first_tx: dict[str, str | None] = {}
    utr: dict[str, list] = defaultdict(list)
    meta: dict[str, tuple] = {}
    with gff.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            attrs = f[8]
            if f[2] == "gene":
                gene = attrs.split(";")[0].split("=")[1]
                meta[gene] = (f[0], f[6])
                first_tx.setdefault(gene, None)
            elif f[2] in ("mRNA", "transcript"):
                tid = re.search(r"ID=([^;]+)", attrs)
                par = re.search(r"Parent=([^;]+)", attrs)
                if tid and par and first_tx.get(par.group(1)) is None:
                    first_tx[par.group(1)] = tid.group(1)
            elif f[2] in ("five_prime_UTR", "three_prime_UTR"):
                par = re.search(r"Parent=([^;]+)", attrs)
                if par:
                    utr[par.group(1)].append((int(f[3]), int(f[4])))
    out = {}
    for gene, tx in first_tx.items():
        if gene not in meta:
            continue
        seq, strand = meta[gene]
        out[gene] = {"seq": seq, "strand": strand, "utr": sorted(utr.get(tx, []))}
    return out


def donate_utr(cds: list, donor: list, strand: str) -> tuple[list, list]:
    """Place `donor` intervals around `cds`, returning (five_prime, three_prime).

    Delegates to `place_donated_utr` in `35_build_frame_shift_prompts.py`, which is the same
    operation this arm's mirror image (arm G) performs in the opposite direction. Sharing one
    definition is the point: the minus-strand bug that arm G carried existed because the two
    donations were written twice, and a second copy here would be a second chance to get the
    strand handling wrong in only one of them.
    """
    return _load_arm_builder().place_donated_utr(cds, donor, strand)


def write_arm(rows: list, path: Path) -> dict:
    """Write the arm and return the counts that describe what the donation actually did."""
    stats = {
        "loci_written": 0,
        "loci_with_donated_utr": 0,
        "loci_without_donated_utr": 0,
        "loci_with_five_prime": 0,
        "loci_with_three_prime": 0,
        "loci_gene_row_widened": 0,
        "widened_bp_total": 0,
        "widened_bp_max": 0,
        "donated_utr_intervals": 0,
        "donated_utr_bp": 0,
        "loci_by_strand": {"+": 0, "-": 0},
        "loci_with_donated_utr_by_strand": {"+": 0, "-": 0},
    }
    lines = ["##gff-version 3"]
    for row in rows:
        gene, seq, strand = row["gene"], row["seq"], row["strand"]
        cds = row["cds"]
        five, three = donate_utr(cds, row["donor_utr"], strand)
        tx = f"{gene}.1"

        # The gene row is TAIR10's own locus span, which is what arm E used and what
        # training framed the encoder window on. It is widened only when a donated interval
        # would otherwise sit outside the window that row defines.
        gene_start, gene_end = row["tair_start"], row["tair_end"]
        segments = five + three
        if segments:
            widened_start = min([gene_start] + [s for s, _ in segments])
            widened_end = max([gene_end] + [e for _, e in segments])
            if widened_start != gene_start or widened_end != gene_end:
                stats["loci_gene_row_widened"] += 1
                widened = (gene_start - widened_start) + (widened_end - gene_end)
                stats["widened_bp_total"] += widened
                stats["widened_bp_max"] = max(stats["widened_bp_max"], widened)
            gene_start, gene_end = widened_start, widened_end

        lines.append(f"{seq}\tframe_test\tgene\t{gene_start}\t{gene_end}\t.\t{strand}\t.\tID={gene}")
        lines.append(f"{seq}\tframe_test\tmRNA\t{gene_start}\t{gene_end}\t.\t{strand}\t.\t"
                     f"ID={tx};Parent={gene}")
        for n, (s, e) in enumerate(cds, 1):
            lines.append(f"{seq}\tframe_test\tCDS\t{s}\t{e}\t.\t{strand}\t0\t"
                         f"ID={tx}.cds{n};Parent={tx}")
        for label, segs in (("five_prime_UTR", five), ("three_prime_UTR", three)):
            for n, (s, e) in enumerate(segs, 1):
                lines.append(f"{seq}\tframe_test\t{label}\t{s}\t{e}\t.\t{strand}\t.\t"
                             f"ID={tx}.{label}{n};Parent={tx}")

        stats["loci_written"] += 1
        stats["loci_by_strand"][strand] = stats["loci_by_strand"].get(strand, 0) + 1
        if five or three:
            stats["loci_with_donated_utr"] += 1
            stats["loci_with_donated_utr_by_strand"][strand] = (
                stats["loci_with_donated_utr_by_strand"].get(strand, 0) + 1)
        else:
            stats["loci_without_donated_utr"] += 1
        stats["loci_with_five_prime"] += 1 if five else 0
        stats["loci_with_three_prime"] += 1 if three else 0
        stats["donated_utr_intervals"] += len(five) + len(three)
        stats["donated_utr_bp"] += sum(e - s + 1 for s, e in five + three)
    path.write_text("\n".join(lines) + "\n")
    return stats


def build(tair10_gtf: Path, primary_ids_path: Path, helixer_gff: Path, out_path: Path) -> dict:
    mod = _load_arm_builder()
    primary_ids = mod.load_primary_ids(primary_ids_path)

    # `load_tair10` supplies the locus-wide gene span (what the gene row is); the CDS it
    # returns is the primary transcript's. `load_tair10_with_utr` supplies the primary's own
    # exon list, which arm E needed to derive reference UTRs and which is used here only to
    # reproduce arm E's locus restriction exactly.
    tair = mod.load_tair10(tair10_gtf, primary_ids)
    helixer_genes = mod.load_helixer_genes(helixer_gff)
    rows, pairing = mod.pair_and_filter(tair, helixer_genes)
    if not rows:
        raise SystemExit("no locus survived pairing — refusing to write an empty arm")

    with_utr = mod.load_tair10_with_utr(tair10_gtf, primary_ids)
    donor = load_helixer_first_transcript_utr(helixer_gff)

    stats = {
        "arm": ARM_NAME,
        "pairing": pairing,
        "loci_dropped_for_missing_exons": 0,
        "loci_dropped_no_helixer_partner_record": 0,
        "helixer_loci_with_any_utr": sum(1 for v in donor.values() if v["utr"]),
        "helixer_loci": len(donor),
    }

    arm_rows = []
    for row in rows:
        # Arm E's own restriction: a locus with no exon rows on the primary transcript could
        # not receive a reference UTR and was dropped there, so it is dropped here too — the
        # two arms must be scored over the same loci.
        info = with_utr.get(row["gene"])
        if not info or not info["exons"]:
            stats["loci_dropped_for_missing_exons"] += 1
            continue
        partner = donor.get(row["helixer_gene"])
        if partner is None:
            stats["loci_dropped_no_helixer_partner_record"] += 1
            continue
        arm_rows.append({**row, "cds": info["cds"], "donor_utr": partner["utr"]})

    stats.update(write_arm(arm_rows, out_path))
    stats["path"] = str(out_path)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tair10-gtf", type=Path,
                    default=ROOT / "revision" / "data" / "TAIR10" / "TAIR10.gtf")
    ap.add_argument("--primary-ids", type=Path,
                    default=ROOT / "revision" / "data" / "TAIR10" / "primary_transcript_ids.txt")
    ap.add_argument("--helixer-gff", type=Path, default=BENCH / "inputs" / "helixer_Athaliana.gff3")
    ap.add_argument("--out", type=Path,
                    default=BENCH / "inputs" / f"{ARM_NAME}_Athaliana.gff3")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.out.exists():
        raise SystemExit(
            f"{args.out} already exists — refusing to overwrite a staged benchmark input; "
            f"delete it deliberately if it is meant to be rebuilt"
        )

    stats = build(args.tair10_gtf, args.primary_ids, args.helixer_gff, args.out)
    print(json.dumps(stats, indent=2))
    if args.json:
        args.json.write_text(json.dumps(stats, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
