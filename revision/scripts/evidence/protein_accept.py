#!/usr/bin/env python3
"""A19.2 acceptance filter over a miniprot GFF, and the per-organism support tables of issue #45.

Separated from the alignment on purpose: the GFF costs ~30 h of node time per species and the
judgement made from it does not. A39 rescored 27 audits from retained SAMs without realigning for
the same reason.

A19.2 accepts an alignment when ALL of:
    identity            >= 0.30      (miniprot's own Identity on the mRNA line)
    protein coverage    >= 0.70      (aligned query span / query length, from the ##PAF line)
    frameshifts         == 0         (fs:i: on the ##PAF line)
    every intron        canonical    (GT-AG or GC-AG, read from the cs:Z: string)
    every intron        <= 200000 nt (from the gaps between consecutive CDS blocks)

Equal-best multi-locus placements are flagged mapping_ambiguous (A19.2, family level only): a
protein whose top alignment score is shared by more than one locus. Their accepted alignments are
counted in the per-alignment statistics and listed in <species>.alignments.tsv, but they contribute
NOTHING to the locus-specific tables (<species>.tsv coverage, <species>.introns.tsv): support_version 2
(2026-09-09) filters them out before aggregation. Version 1 emitted their blocks and introns during the
streaming pass and only named the proteins afterwards, so 1.4-4.1 % of proteins per species gave
locus-specific support the protocol reserves for family level (Codex review 2026-09-09).

The independent support unit is the OrthoDB ORGANISM, not the sequence: headers are
'<organism>:<serial>', so 1000413_0:000002 contributes as 1000413_0. Two proteins from one organism
covering the same base are one organism, which is what the A18.4 weight counts.

Nothing here labels anything. A19.3 keeps protein evidence to CDS-family classes, gives intron and
splice-boundary classes weight 0 unless >= 2 organisms place the same boundary exactly, and forbids
UTRs, GSF transcripts and B1 support counts. This script reports; it does not decide.
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

SUPPORT_VERSION = 2          # 2: mapping_ambiguous proteins excluded from the locus-specific tables
CANONICAL = {("gt", "ag"), ("gc", "ag")}
MAX_INTRON = 200_000
MIN_IDENTITY = 0.30
MIN_COVERAGE = 0.70

CS_INTRON = re.compile(r"~([a-z]{2})(\d+)([a-z]{2})")
ATTR = re.compile(r"([^=;]+)=([^;]*)")


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def organism_of(protein_id):
    """OrthoDB headers are '<organism>:<serial>'. Refuse anything else rather than invent a unit."""
    org, sep, _ = protein_id.partition(":")
    if not sep or not org:
        raise ValueError(f"protein id {protein_id!r} is not '<organism>:<serial>'")
    return org


def parse_paf(line):
    f = line.rstrip("\n").split("\t")[1:]
    rec = {"qname": f[0], "qlen": int(f[1]), "qstart": int(f[2]), "qend": int(f[3]),
           "strand": f[4], "tname": f[5], "fs": None, "cs": None}
    for tag in f[12:]:
        if tag.startswith("fs:i:"):
            rec["fs"] = int(tag[5:])
        elif tag.startswith("cs:Z:"):
            rec["cs"] = tag[5:]
    return rec


def judge(paf, mrna_attrs, cds):
    """Return (accepted, reasons, introns). introns are (start, end, donor, acceptor), 1-based inclusive."""
    reasons = []
    identity = float(mrna_attrs.get("Identity", "nan"))
    if not identity >= MIN_IDENTITY:
        reasons.append("identity")
    coverage = (paf["qend"] - paf["qstart"]) / paf["qlen"] if paf["qlen"] else 0.0
    if not coverage >= MIN_COVERAGE:
        reasons.append("coverage")
    if paf["fs"] is None:
        reasons.append("frameshift_unknown")     # missing evidence is not absence of frameshifts
    elif paf["fs"] != 0:
        reasons.append("frameshift")

    blocks = sorted(cds)
    gaps = [(blocks[i][1] + 1, blocks[i + 1][0] - 1) for i in range(len(blocks) - 1)]
    motifs = CS_INTRON.findall(paf["cs"] or "")
    introns = []
    if len(motifs) != len(gaps):
        # The two readings disagree, so neither is trusted. Guessing which is right is how a
        # non-canonical intron becomes an accepted one.
        reasons.append("intron_parse_mismatch")
    else:
        for (gs, ge), (donor, _n, acceptor) in zip(gaps, motifs):
            introns.append((gs, ge, donor, acceptor))
            if (donor, acceptor) not in CANONICAL:
                reasons.append("noncanonical_intron")
            if ge - gs + 1 > MAX_INTRON:
                reasons.append("intron_too_long")
    return (not reasons), sorted(set(reasons)), introns, identity, coverage


def stream(gff, blocks_fh, introns_fh, alignments_fh):
    """One pass. Emits one row per alignment, plus its CDS blocks and introns when accepted.

    Block and intron rows carry the protein id as a trailing column: ambiguity is only known once every
    alignment of a protein has been seen, so the rows are filtered afterwards (see filter_ambiguous)."""
    paf = None
    cur = None          # (mrna_id, attrs, score, chrom, strand)
    cds = []
    per_protein = defaultdict(list)      # protein -> [(score, accepted, key)]
    stats = defaultdict(int)
    rejected = defaultdict(int)

    def flush():
        nonlocal cur, cds
        if cur is None:
            return
        mid, attrs, score, chrom, strand = cur
        target = attrs.get("Target", "").split()
        prot = target[0] if target else ""
        if paf is None or paf["qname"] != prot:
            raise SystemExit(f"REFUSED: {mid} has no matching ##PAF record (got {paf and paf['qname']!r}, want {prot!r})")
        org = organism_of(prot)
        ok, reasons, introns, identity, coverage = judge(paf, attrs, cds)
        stats["alignments"] += 1
        stats["accepted"] += int(ok)
        for r in reasons:
            rejected[r] += 1
        per_protein[prot].append((score, ok, (chrom, cds[0][0] if cds else 0, strand)))
        alignments_fh.write(f"{mid}\t{prot}\t{org}\t{chrom}\t{strand}\t{score}\t"
                            f"{identity:.4f}\t{coverage:.4f}\t{paf['fs']}\t{len(cds)}\t"
                            f"{'accepted' if ok else 'rejected'}\t{','.join(reasons) or '.'}\n")
        if ok:
            for s, e in sorted(cds):
                blocks_fh.write(f"{chrom}\t{s}\t{e}\t{org}\t{prot}\n")
            for s, e, d, a in introns:
                introns_fh.write(f"{chrom}\t{s}\t{e}\t{strand}\t{d}-{a}\t{org}\t{prot}\n")
        cur, cds = None, []

    with _open(gff) as fh:
        for line in fh:
            if line.startswith("##PAF\t"):
                flush()
                paf = parse_paf(line)
                continue
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            if f[2] == "mRNA":
                flush()
                attrs = dict(ATTR.findall(f[8]))
                cur = (attrs.get("ID", ""), attrs, int(f[5]), f[0], f[6])
            elif f[2] == "CDS" and cur is not None:
                cds.append((int(f[3]), int(f[4])))
    flush()
    return per_protein, stats, rejected


def filter_ambiguous(src_path, dst_path, ambiguous):
    """Copy support rows whose trailing protein column is not in `ambiguous`, dropping that column.

    Returns the number of rows removed. Streams line by line: the block file of one species is tens of
    millions of rows, the ambiguous set at most a few hundred thousand ids."""
    removed = 0
    with open(src_path) as src, open(dst_path, "w") as dst:
        for line in src:
            body, _, prot = line.rstrip("\n").rpartition("\t")
            if prot in ambiguous:
                removed += 1
                continue
            dst.write(body + "\n")
    return removed


def sweep_organisms(sorted_blocks, out_fh):
    """Per-base distinct-organism depth, run-length encoded. A base is covered by an organism once
    however many of its proteins align there — that is what 'independent unit = organism' means."""
    events = defaultdict(lambda: defaultdict(int))   # chrom -> pos -> delta is too big; stream instead
    del events
    cur_chrom, active, pending, last_pos, written = None, defaultdict(int), [], None, 0

    def emit(chrom, start, end, n):
        nonlocal written
        if n > 0 and end >= start:
            out_fh.write(f"{chrom}\t{start}\t{end}\t{n}\n")
            written += 1

    # sorted_blocks arrives sorted by chrom,start; use a sweep with an end-heap
    import heapq
    heap = []
    for line in sorted_blocks:
        chrom, s, e, org = line.rstrip("\n").split("\t")
        s, e = int(s), int(e)
        if chrom != cur_chrom:
            while heap:
                end, o = heapq.heappop(heap)
                if end >= last_pos:
                    emit(cur_chrom, last_pos, end, len([k for k, v in active.items() if v > 0]))
                    last_pos = end + 1
                active[o] -= 1
                if active[o] == 0:
                    del active[o]
            cur_chrom, active, heap, last_pos = chrom, defaultdict(int), [], s
        while heap and heap[0][0] < s:
            end, o = heapq.heappop(heap)
            if end >= last_pos:
                emit(chrom, last_pos, end, len(active))
                last_pos = end + 1
            active[o] -= 1
            if active[o] == 0:
                del active[o]
        if s > last_pos and active:
            emit(chrom, last_pos, s - 1, len(active))
        last_pos = max(last_pos, s)
        active[org] += 1
        heapq.heappush(heap, (e, org))
    while heap:
        end, o = heapq.heappop(heap)
        if end >= last_pos:
            emit(cur_chrom, last_pos, end, len(active))
            last_pos = end + 1
        active[o] -= 1
        if active[o] == 0:
            del active[o]
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gff", required=True, help="miniprot --gff output (.gff or .gff.gz)")
    ap.add_argument("--species", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-intermediates", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, a.species)

    blocks_p, introns_p = base + ".blocks.tmp", base + ".introns.tmp"
    with open(blocks_p, "w") as bf, open(introns_p, "w") as inf, open(base + ".alignments.tsv", "w") as af:
        af.write("alignment_id\tprotein\torganism\tchrom\tstrand\tscore\tidentity\tcoverage\t"
                 "frameshifts\tcds_blocks\tverdict\treasons\n")
        per_protein, stats, rejected = stream(a.gff, bf, inf, af)

    # A19.2: equal-best multi-locus placements are mapping_ambiguous.
    ambiguous = set()
    for prot, rows in per_protein.items():
        best = max(r[0] for r in rows)
        if sum(1 for r in rows if r[0] == best) > 1:
            ambiguous.add(prot)
    with open(base + ".mapping_ambiguous.txt", "w") as fh:
        for p in sorted(ambiguous):
            fh.write(p + "\n")
    ambiguous_accepted = sum(sum(1 for r in rows if r[1]) for prot, rows in per_protein.items() if prot in ambiguous)

    # Family level only: an ambiguous protein's accepted alignments leave the locus-specific tables here,
    # before anything is aggregated. The tables below carry no protein identity, so this is the only place
    # the exclusion can happen.
    blocks_f, introns_f = base + ".blocks.filtered.tmp", base + ".introns.filtered.tmp"
    blocks_removed = filter_ambiguous(blocks_p, blocks_f, ambiguous)
    introns_removed = filter_ambiguous(introns_p, introns_f, ambiguous)

    env = dict(os.environ, LC_ALL="C")
    srt = base + ".blocks.sorted.tmp"
    subprocess.run(["sort", "-k1,1", "-k2,2n", "-o", srt, blocks_f], check=True, env=env)
    with open(srt) as sf, open(base + ".tsv", "w") as of:
        of.write("chrom\tstart\tend\torganisms\n")
        intervals = sweep_organisms(sf, of)

    # A19.3: intron and splice-boundary classes only where >= 2 organisms place the boundary exactly.
    isrt = base + ".introns.sorted.tmp"
    subprocess.run(["sort", "-k1,1", "-k2,2n", "-k3,3n", "-o", isrt, introns_f], check=True, env=env)
    supported = 0
    with open(isrt) as sf, open(base + ".introns.tsv", "w") as of:
        of.write("chrom\tstart\tend\tstrand\tmotif\torganisms\tn_organisms\n")
        key, orgs, motif = None, set(), ""
        def flush_intron():
            nonlocal supported
            if key and len(orgs) >= 2:
                of.write(f"{key[0]}\t{key[1]}\t{key[2]}\t{key[3]}\t{motif}\t{','.join(sorted(orgs))}\t{len(orgs)}\n")
                supported += 1
        for line in sf:
            c, s, e, st, mo, org = line.rstrip("\n").split("\t")
            k = (c, s, e, st)
            if k != key:
                flush_intron()
                key, orgs, motif = k, set(), mo
            orgs.add(org)
        flush_intron()

    summary = {"species": a.species, "gff": os.path.abspath(a.gff),
               "thresholds": {"identity": MIN_IDENTITY, "coverage": MIN_COVERAGE,
                              "max_intron": MAX_INTRON, "canonical": sorted("-".join(c) for c in CANONICAL)},
               "alignments": stats["alignments"], "accepted": stats["accepted"],
               "accepted_fraction": (stats["accepted"] / stats["alignments"]) if stats["alignments"] else 0.0,
               "proteins": len(per_protein), "mapping_ambiguous": len(ambiguous),
               "ambiguous_fraction": (len(ambiguous) / len(per_protein)) if per_protein else 0.0,
               "support_version": SUPPORT_VERSION,
               "support_excludes_mapping_ambiguous": True,
               "ambiguous_alignments_excluded": ambiguous_accepted,
               "ambiguous_blocks_excluded": blocks_removed,
               "ambiguous_introns_excluded": introns_removed,
               "rejected_by_reason": dict(sorted(rejected.items())),
               "coverage_intervals": intervals,
               "introns_supported_by_2plus_organisms": supported}
    with open(base + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))

    if not a.keep_intermediates:
        for p in (blocks_p, introns_p, blocks_f, introns_f, srt, isrt):
            os.unlink(p)


if __name__ == "__main__":
    main()
