#!/usr/bin/env python3
"""Build the pair of prompts that isolates the coordinate frame from the gene structure.

THE QUESTION

Prompted with TAIR10 itself, completion mode adds isoforms that match a TAIR10 alternative
18.1% of the time (manuscript, Discussion). Prompted with Helixer or ANNEVO it manages
0.0-0.2%, and the 2026-08-06 benchmark ruled out every explanation that lives in the gene
structure: prompt quality (restricting to loci whose supplied CDS already equals a real
TAIR10 transcript leaves precision at 0.1-0.3%), prompt destruction (pinning the prompted
transcript during decoding leaves it at 0.2%), UTR presence, and output volume.

What remains is the coordinate frame. `preprocess.py` builds each encoder window from the
GENE row and pads it to a multiple of 6,144 nt, and GSF coordinates are written relative to
that window. Training used TAIR10's own gene boundaries. Helixer's boundaries agree with
TAIR10 on exactly 1 of 25,657 overlapping genes, because Helixer predicts UTRs and TAIR10's
gene row spans its exons. So even a locus whose CDS structure is identical to TAIR10's
arrives at the model inside a different window, with a different origin — a frame the model
never saw in training.

THE TEST

Two annotations that differ in nothing but that frame:

    arm A (`tair10self`)        TAIR10's primary CDS, inside TAIR10's own gene boundary
    arm B (`tair10helixerframe`) the SAME CDS, inside Helixer's gene boundary

Identical CDS coordinates, identical loci, identical everything the biology depends on.
If arm A reproduces something near 18.1% and arm B collapses toward 0%, the frame is the
cause and the manuscript's figure does not transfer to another tool's annotation. If both
score alike, the frame is not the cause and the explanation is still open — which is why
arm A is built at all rather than comparing against the published number.

WHAT IS HELD FIXED, AND WHY IT MATTERS

Only loci where Helixer's gene span CONTAINS TAIR10's primary CDS are emitted, so no CDS is
ever clipped to fit a frame and both arms carry byte-identical CDS rows. Both files are
written from the same locus list in the same order, so a difference in the scored result
cannot come from a difference in which loci were scored.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAIR10_GTF = ROOT / "revision" / "data" / "TAIR10" / "TAIR10.gtf"
PRIMARY_IDS = ROOT / "revision" / "data" / "TAIR10" / "primary_transcript_ids.txt"
BENCH = ROOT.parent / "polishing_benchmark"
HELIXER_GFF = BENCH / "inputs" / "helixer_Athaliana.gff3"

_GTF_TX = re.compile(r'transcript_id "([^"]+)"')
_GTF_GENE = re.compile(r'gene_id "([^"]+)"')


def load_primary_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_tair10(gtf: Path, primary_ids: set[str]) -> dict:
    """gene -> {seq, strand, cds: [(s,e)], gene_start, gene_end}.

    The CDS comes from the primary transcript only — that is the prompt. The gene span
    comes from every exon of every transcript at the locus, which is what TAIR10's own gene
    row covers and therefore what training framed the window on.
    """
    cds: dict[str, list] = defaultdict(list)
    span: dict[str, list] = {}
    meta: dict[str, tuple] = {}
    with gtf.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in ("CDS", "exon"):
                continue
            g = _GTF_GENE.search(f[8])
            t = _GTF_TX.search(f[8])
            if not (g and t):
                continue
            gene, tx = g.group(1), t.group(1)
            start, end = int(f[3]), int(f[4])
            cur = span.setdefault(gene, [start, end])
            cur[0], cur[1] = min(cur[0], start), max(cur[1], end)
            meta[gene] = (f[0], f[6])
            if f[2] == "CDS" and tx in primary_ids:
                cds[gene].append((start, end))
    out = {}
    for gene, segments in cds.items():
        seq, strand = meta[gene]
        out[gene] = {
            "seq": seq,
            "strand": strand,
            "cds": sorted(segments),
            "gene_start": span[gene][0],
            "gene_end": span[gene][1],
        }
    return out


def load_helixer_genes(gff: Path) -> dict:
    """seq -> sorted [(start, end, gene_id)] for gene rows."""
    by_seq: dict[str, list] = defaultdict(list)
    with gff.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            gene_id = f[8].split(";")[0].split("=")[1]
            by_seq[f[0]].append((int(f[3]), int(f[4]), gene_id))
    for seq in by_seq:
        by_seq[seq].sort()
    return by_seq


def pair_and_filter(tair: dict, helixer: dict) -> tuple[list, dict]:
    """Pair each TAIR10 gene to the best-overlapping Helixer gene whose span CONTAINS its
    primary CDS. Containment is the whole point: a frame that clipped the CDS would change
    the structure as well as the frame, and the comparison would answer nothing.
    """
    stats = {"tair10_genes": len(tair), "no_overlap": 0, "not_containing": 0, "paired": 0,
             "ambiguous_resolved_by_overlap": 0}
    rows = []
    for gene, info in sorted(tair.items()):
        cds_min, cds_max = info["cds"][0][0], info["cds"][-1][1]
        candidates = [
            (hs, he, hid) for hs, he, hid in helixer.get(info["seq"], [])
            if hs <= info["gene_end"] and he >= info["gene_start"]
        ]
        if not candidates:
            stats["no_overlap"] += 1
            continue
        containing = [c for c in candidates if c[0] <= cds_min and c[1] >= cds_max]
        if not containing:
            stats["not_containing"] += 1
            continue
        if len(containing) > 1:
            stats["ambiguous_resolved_by_overlap"] += 1
        best = max(containing, key=lambda c: min(c[1], info["gene_end"]) - max(c[0], info["gene_start"]))
        stats["paired"] += 1
        rows.append({
            "gene": gene, "seq": info["seq"], "strand": info["strand"], "cds": info["cds"],
            "tair_start": info["gene_start"], "tair_end": info["gene_end"],
            "helixer_start": best[0], "helixer_end": best[1], "helixer_gene": best[2],
        })
    return rows, stats


def write_arm(rows: list, path: Path, frame: str) -> int:
    """Write one arm. `frame` picks which gene boundary the window will be built from.

    CDS rows are written from the same list in both arms, so the two files differ only in
    the gene and mRNA spans.
    """
    lines = ["##gff-version 3"]
    for row in rows:
        if frame == "tair10":
            gene_start, gene_end = row["tair_start"], row["tair_end"]
        else:
            gene_start, gene_end = row["helixer_start"], row["helixer_end"]
        gene, seq, strand = row["gene"], row["seq"], row["strand"]
        tx = f"{gene}.1"
        lines.append(f"{seq}\tframe_test\tgene\t{gene_start}\t{gene_end}\t.\t{strand}\t.\tID={gene}")
        lines.append(f"{seq}\tframe_test\tmRNA\t{gene_start}\t{gene_end}\t.\t{strand}\t.\t"
                     f"ID={tx};Parent={gene}")
        for n, (s, e) in enumerate(row["cds"], 1):
            lines.append(f"{seq}\tframe_test\tCDS\t{s}\t{e}\t.\t{strand}\t0\t"
                         f"ID={tx}.cds{n};Parent={tx}")
    path.write_text("\n".join(lines) + "\n")
    return len(rows)


def utr_segments(exons: list, cds: list, strand: str) -> tuple:
    """Split the exonic sequence outside the CDS into 5' and 3' UTR intervals.

    TAIR10's GTF carries exon and CDS rows but no UTR rows, so the UTRs the manuscript's
    prompt contained were derived exactly this way by `preprocess.py`'s reader. Rebuilding
    them here is what makes an arm comparable to that run: GSF has dedicated
    five_prime_UTR/three_prime_UTR tokens, so a prompt without them is a different prompt,
    not merely a sparser one — measured at 18.6% with UTRs against 4.6% without.
    """
    if not cds:
        return [], []
    cds_min, cds_max = cds[0][0], cds[-1][1]
    upstream, downstream = [], []
    for start, end in sorted(exons):
        if end < cds_min:
            upstream.append((start, end))
        elif start > cds_max:
            downstream.append((start, end))
        else:
            if start < cds_min:
                upstream.append((start, cds_min - 1))
            if end > cds_max:
                downstream.append((cds_max + 1, end))
    if strand == "-":
        return downstream, upstream  # 5' is the high-coordinate side on the minus strand
    return upstream, downstream


def load_tair10_with_utr(gtf: Path, primary_ids: set) -> dict:
    """gene -> {seq, strand, cds, exons, gene_start, gene_end} for the PRIMARY transcript.

    Unlike `load_tair10`, the exon list is the primary's own rather than the locus-wide
    span, because UTRs must come from the transcript that is actually prompted.
    """
    cds: dict = defaultdict(list)
    exons: dict = defaultdict(list)
    span: dict = {}
    meta: dict = {}
    with gtf.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in ("CDS", "exon"):
                continue
            g, t = _GTF_GENE.search(f[8]), _GTF_TX.search(f[8])
            if not (g and t):
                continue
            gene, tx = g.group(1), t.group(1)
            start, end = int(f[3]), int(f[4])
            cur = span.setdefault(gene, [start, end])
            cur[0], cur[1] = min(cur[0], start), max(cur[1], end)
            meta[gene] = (f[0], f[6])
            if tx in primary_ids:
                (cds if f[2] == "CDS" else exons)[gene].append((start, end))
    out = {}
    for gene, segments in cds.items():
        seq, strand = meta[gene]
        out[gene] = {
            "seq": seq, "strand": strand,
            "cds": sorted(segments), "exons": sorted(exons.get(gene, [])),
            "gene_start": span[gene][0], "gene_end": span[gene][1],
        }
    return out


def write_arm_with_utr(rows: list, path: Path, frame: str) -> int:
    """Same as `write_arm`, plus the five_prime_UTR / three_prime_UTR rows."""
    lines = ["##gff-version 3"]
    for row in rows:
        gene_start, gene_end = ((row["tair_start"], row["tair_end"]) if frame == "tair10"
                                else (row["helixer_start"], row["helixer_end"]))
        gene, seq, strand = row["gene"], row["seq"], row["strand"]
        tx = f"{gene}.1"
        five, three = utr_segments(row["exons"], row["cds"], strand)
        lines.append(f"{seq}\tframe_test\tgene\t{gene_start}\t{gene_end}\t.\t{strand}\t.\tID={gene}")
        lines.append(f"{seq}\tframe_test\tmRNA\t{gene_start}\t{gene_end}\t.\t{strand}\t.\t"
                     f"ID={tx};Parent={gene}")
        for n, (s, e) in enumerate(row["cds"], 1):
            lines.append(f"{seq}\tframe_test\tCDS\t{s}\t{e}\t.\t{strand}\t0\t"
                         f"ID={tx}.cds{n};Parent={tx}")
        for label, segs in (("five_prime_UTR", five), ("three_prime_UTR", three)):
            for n, (s, e) in enumerate(segs, 1):
                lines.append(f"{seq}\tframe_test\t{label}\t{s}\t{e}\t.\t{strand}\t.\t"
                             f"ID={tx}.{label}{n};Parent={tx}")
    path.write_text("\n".join(lines) + "\n")
    return len(rows)


def _merge(intervals: list) -> list:
    """Coalesce overlapping or abutting intervals so no two rows describe the same base."""
    merged: list = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def place_donated_utr(cds: list, donor: list, strand: str) -> tuple[list, list]:
    """Place donated UTR intervals around `cds`, returning (five_prime, three_prime).

    Intervals overlapping the CDS span are clipped to the part outside it, so a donation can
    never overlap or alter the CDS it is attached to. Which side is 5' and which is 3' is
    then decided by coordinate AND strand, exactly as `utr_segments` decides it for the
    reference arms.

    That last point is the whole reason this helper exists. `write_helixer_with_tair_utr`
    used to take the strand-labelled `five`/`three` lists straight from `utr_segments` and
    filter them by raw coordinate (`five` kept only where `s < lo`, `three` only where
    `e > hi`). On the minus strand `utr_segments` puts the 5' UTR on the HIGH-coordinate
    side, so both tests rejected every interval and the locus silently received no UTR at
    all: 35 of 11,989 minus-strand loci got a donated UTR in `helixertairutr`, and 2 of
    10,412 in `annevotairutr`. Both arms therefore measured a ~50:50 mixture of the cell
    they were built to measure and the no-UTR cell, which understated them — rescoring the
    plus-strand half alone gives 2.2% and 3.5% against the 0.9% and 1.2% first reported,
    with the `tair10selfutr` control flat across the same split (18.1% / 18.4%).

    Donor labels are deliberately pooled and re-derived rather than carried over: a 5'/3'
    label is only meaningful relative to the CDS an interval flanks, and after donation that
    CDS is not the one the label was written against.
    """
    lo, hi = cds[0][0], cds[-1][1]
    low_side, high_side = [], []
    for start, end in donor:
        if end < lo:
            low_side.append((start, end))
        elif start > hi:
            high_side.append((start, end))
        else:
            if start < lo:
                low_side.append((start, lo - 1))
            if end > hi:
                high_side.append((hi + 1, end))
    low_side, high_side = _merge(low_side), _merge(high_side)
    if strand == "-":
        return high_side, low_side
    return low_side, high_side


def load_helixer_first_cds(gff: Path) -> dict:
    """gene -> {seq, strand, cds} for Helixer's FIRST mRNA — the transcript that is the
    prompt. Only CDS is kept: exon and UTR rows are dropped so this arm carries the same
    feature vocabulary as the frame-test arms and differs from them in gene structure only.
    """
    parent_of_tx: dict[str, str] = {}
    first_tx: dict[str, str] = {}
    cds: dict[str, list] = defaultdict(list)
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
                if tid and par:
                    parent_of_tx[tid.group(1)] = par.group(1)
                    if first_tx.get(par.group(1)) is None:
                        first_tx[par.group(1)] = tid.group(1)
            elif f[2] == "CDS":
                par = re.search(r"Parent=([^;]+)", attrs)
                if par:
                    cds[par.group(1)].append((int(f[3]), int(f[4])))
    out = {}
    for gene, tx in first_tx.items():
        if not tx or not cds.get(tx):
            continue
        seq, strand = meta[gene]
        out[gene] = {"seq": seq, "strand": strand, "cds": sorted(cds[tx])}
    return out


def write_reframed(genes: dict, path: Path) -> int:
    """Helixer's CDS, framed on the CDS span itself.

    The gene row is the only thing this changes, and it uses no information the tool did
    not already produce — no reference, no TAIR10 boundary. That is deliberate: if the
    coordinate frame is what separates 18.1% from 0.2%, a fix has to be reachable without
    knowing the answer, or it is not a fix but a measurement of the answer.
    """
    lines = ["##gff-version 3"]
    for gene in sorted(genes):
        info = genes[gene]
        start, end = info["cds"][0][0], info["cds"][-1][1]
        seq, strand, tx = info["seq"], info["strand"], f"{gene}.1"
        lines.append(f"{seq}\tframe_test\tgene\t{start}\t{end}\t.\t{strand}\t.\tID={gene}")
        lines.append(f"{seq}\tframe_test\tmRNA\t{start}\t{end}\t.\t{strand}\t.\t"
                     f"ID={tx};Parent={gene}")
        for n, (s, e) in enumerate(info["cds"], 1):
            lines.append(f"{seq}\tframe_test\tCDS\t{s}\t{e}\t.\t{strand}\t0\t"
                         f"ID={tx}.cds{n};Parent={tx}")
    path.write_text("\n".join(lines) + "\n")
    return len(genes)


def write_helixer_with_tair_utr(helixer: dict, tair_utr: dict, path: Path) -> dict:
    """Helixer's own CDS, given TAIR10's UTR — a diagnostic, not a deliverable.

    The 2x2 showed UTR decides whether correct alternatives are produced at all (33 hits
    without, 185-278 with) and that Helixer never gets a UTR exactly right (0 of 17,830).
    What that leaves open is whether UTR is the ONLY thing standing between Helixer and
    the manuscript's number, or whether its CDS — correct at 76.9% of loci, wrong at the
    rest — also blocks it.

    This arm answers that by handing Helixer's CDS the reference's UTR. It uses the answer
    to build the input, so nothing here could ship. It is worth running only because the
    result decides where effort goes: near 18% means UTR correction (long reads, a UTR
    caller) is the whole job, while a low number means the CDS matters too and fixing UTRs
    alone would not have paid off.

    A TAIR10 UTR interval is kept only where it falls outside the supplied CDS span, so the
    two never overlap and the CDS is never altered. Placement is delegated to
    `place_donated_utr`, whose docstring records the minus-strand bug this function carried
    until 2026-08-17 and the corrected numbers it produced.

    `loci_with_utr_by_strand` is reported for exactly that reason: the bug was invisible in
    every aggregate the original stats block emitted, because a locus that received no UTR
    was still counted as written. A donation that works on one strand and not the other is
    now visible in the stats without anyone having to think to look for it.
    """
    stats = {"helixer_loci": len(helixer), "paired_to_tair10": 0, "written": 0,
             "no_tair10_partner": 0, "partner_has_no_utr": 0,
             "loci_by_strand": defaultdict(int), "loci_with_utr_by_strand": defaultdict(int)}
    lines = ["##gff-version 3"]
    for gene in sorted(helixer):
        info = helixer[gene]
        partner = tair_utr.get(gene)
        if partner is None:
            stats["no_tair10_partner"] += 1
            continue
        stats["paired_to_tair10"] += 1
        cds = info["cds"]
        lo, hi = cds[0][0], cds[-1][1]
        five, three = place_donated_utr(
            cds, list(partner["five"]) + list(partner["three"]), info["strand"])
        stats["loci_by_strand"][info["strand"]] += 1
        if five or three:
            stats["loci_with_utr_by_strand"][info["strand"]] += 1
        else:
            stats["partner_has_no_utr"] += 1
        starts = [lo] + [s for s, _ in five + three]
        ends = [hi] + [e for _, e in five + three]
        gene_start, gene_end = min(starts), max(ends)
        seq, strand, tx = info["seq"], info["strand"], f"{gene}.1"
        lines.append(f"{seq}\tframe_test\tgene\t{gene_start}\t{gene_end}\t.\t{strand}\t.\tID={gene}")
        lines.append(f"{seq}\tframe_test\tmRNA\t{gene_start}\t{gene_end}\t.\t{strand}\t.\t"
                     f"ID={tx};Parent={gene}")
        for n, (s, e) in enumerate(cds, 1):
            lines.append(f"{seq}\tframe_test\tCDS\t{s}\t{e}\t.\t{strand}\t0\t"
                         f"ID={tx}.cds{n};Parent={tx}")
        for label, segs in (("five_prime_UTR", five), ("three_prime_UTR", three)):
            for n, (s, e) in enumerate(sorted(segs), 1):
                lines.append(f"{seq}\tframe_test\t{label}\t{s}\t{e}\t.\t{strand}\t.\t"
                             f"ID={tx}.{label}{n};Parent={tx}")
        stats["written"] += 1
    path.write_text("\n".join(lines) + "\n")
    stats["loci_by_strand"] = dict(stats["loci_by_strand"])
    stats["loci_with_utr_by_strand"] = dict(stats["loci_with_utr_by_strand"])
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tair10-gtf", type=Path, default=TAIR10_GTF)
    ap.add_argument("--primary-ids", type=Path, default=PRIMARY_IDS)
    ap.add_argument("--helixer-gff", type=Path, default=HELIXER_GFF)
    ap.add_argument("--annevo-gff", type=Path, default=BENCH / "inputs" / "annevo_Athaliana.gff3",
                    help="ANNEVO emits no UTR at all (exon == CDS everywhere) and its gene "
                         "boundaries agree with TAIR10 at 10.8% against Helixer's 0.004%, "
                         "so the UTR-donation arm is a different experiment on it.")
    ap.add_argument("--out-dir", type=Path, default=BENCH / "inputs")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite arm inputs that already exist (see the guard below)")
    args = ap.parse_args(argv)

    # Every arm this script writes has already been generated on the GPU, and each
    # prediction's provenance pins the md5 of the input it was made from. Rewriting an input
    # in place therefore invalidates a result that still looks valid — and the minus-strand
    # UTR fix means a re-run does NOT reproduce the staged files byte-for-byte. Re-runs are
    # opt-in for that reason; to regenerate the donation arms under new names instead, use
    # `56_rebuild_donation_arms_fixed.py`.
    arm_files = ["tair10self", "tair10helixerframe", "helixer_reframed",
                 "tair10selfutr", "tair10helixerframeutr", "helixertairutr", "annevotairutr"]
    existing = [p for p in (args.out_dir / f"{a}_Athaliana.gff3" for a in arm_files) if p.exists()]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite staged benchmark inputs that existing predictions were "
            "made from:\n  " + "\n  ".join(str(p) for p in existing) +
            "\npass --force if that is genuinely what you want"
        )

    primary_ids = load_primary_ids(args.primary_ids)
    tair = load_tair10(args.tair10_gtf, primary_ids)
    helixer = load_helixer_genes(args.helixer_gff)
    rows, stats = pair_and_filter(tair, helixer)
    if not rows:
        raise SystemExit("no locus survived pairing — refusing to write empty arms")

    a = args.out_dir / "tair10self_Athaliana.gff3"
    b = args.out_dir / "tair10helixerframe_Athaliana.gff3"
    stats["arm_A_loci"] = write_arm(rows, a, "tair10")
    stats["arm_B_loci"] = write_arm(rows, b, "helixer")

    identical = sum(1 for r in rows
                    if r["tair_start"] == r["helixer_start"] and r["tair_end"] == r["helixer_end"])
    stats["loci_whose_frame_is_unchanged"] = identical
    stats["arm_A_path"] = str(a)
    stats["arm_B_path"] = str(b)

    d = args.out_dir / "helixer_reframed_Athaliana.gff3"
    stats["arm_D_loci"] = write_reframed(load_helixer_first_cds(args.helixer_gff), d)
    stats["arm_D_path"] = str(d)

    # UTR-carrying counterparts of A and B. Without these the frame comparison runs at
    # 4.6% rather than the 18.6% the manuscript's own prediction scores, so a frame effect
    # measured on the CDS-only arms is measured in the wrong regime.
    with_utr = load_tair10_with_utr(args.tair10_gtf, primary_ids)
    utr_rows = []
    for row in rows:
        info = with_utr.get(row["gene"])
        if not info or not info["exons"]:
            continue
        utr_rows.append({**row, "exons": info["exons"], "cds": info["cds"]})
    e = args.out_dir / "tair10selfutr_Athaliana.gff3"
    f = args.out_dir / "tair10helixerframeutr_Athaliana.gff3"
    stats["arm_E_loci"] = write_arm_with_utr(utr_rows, e, "tair10")
    stats["arm_F_loci"] = write_arm_with_utr(utr_rows, f, "helixer")
    stats["arm_E_path"] = str(e)
    stats["arm_F_path"] = str(f)
    stats["loci_dropped_for_missing_exons"] = len(rows) - len(utr_rows)

    # arm G — Helixer's CDS with TAIR10's UTR. Diagnostic only; see the function docstring.
    tair_utr_by_helixer = {}
    for row in utr_rows:
        five, three = utr_segments(row["exons"], row["cds"], row["strand"])
        tair_utr_by_helixer[row["helixer_gene"]] = {"five": five, "three": three}
    g = args.out_dir / "helixertairutr_Athaliana.gff3"
    stats["arm_G"] = write_helixer_with_tair_utr(
        load_helixer_first_cds(args.helixer_gff), tair_utr_by_helixer, g)
    stats["arm_G_path"] = str(g)

    # arm H — the same donation applied to ANNEVO. Worth running separately because the two
    # tools fail differently: Helixer emits UTRs that are never right and over-generates on
    # a boundary that is essentially never right, while ANNEVO emits no UTR at all and its
    # boundaries agree with TAIR10 at 10.8%. If UTR is the binding constraint, ANNEVO is
    # the tool with the most to gain from having it supplied.
    annevo_cds = load_helixer_first_cds(args.annevo_gff)
    annevo_genes = load_helixer_genes(args.annevo_gff)
    annevo_rows, stats_pair = pair_and_filter(
        {k: v for k, v in with_utr.items()}, annevo_genes)
    tair_utr_by_annevo = {}
    for row in annevo_rows:
        info = with_utr.get(row["gene"])
        if not info or not info["exons"]:
            continue
        five, three = utr_segments(info["exons"], info["cds"], row["strand"])
        tair_utr_by_annevo[row["helixer_gene"]] = {"five": five, "three": three}
    h = args.out_dir / "annevotairutr_Athaliana.gff3"
    stats["arm_H"] = write_helixer_with_tair_utr(annevo_cds, tair_utr_by_annevo, h)
    stats["arm_H_pairing"] = stats_pair
    stats["arm_H_path"] = str(h)
    print(json.dumps(stats, indent=2))
    if args.json:
        args.json.write_text(json.dumps(stats, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
