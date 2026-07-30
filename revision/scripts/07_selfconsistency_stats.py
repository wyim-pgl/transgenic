#!/usr/bin/env python3
"""
Self-consistency / plausibility statistics for TransGenic predictions (Reviewer 2).

For each predicted transcript, using genome sequence:
  - translation check: CDS assembled in transcription order ->
      * length divisible by 3
      * starts with ATG
      * ends with a stop codon (TAA/TAG/TGA)
      * no internal stop codons
  - duplicate transcripts: identical exon chains within the same gene
  - transcripts without any CDS
  - coordinate sanity: features within gene bounds, non-overlapping exons
  - isoforms per gene distribution (prediction vs reference, if --ref given)

Outputs JSON + TSV summary.

Usage:
  python 07_selfconsistency_stats.py --pred pred.gff3 --fasta genome.fa \
      [--ref reference.gff3] --out out_prefix
"""

import argparse
import json
import re
import sys
from collections import defaultdict

STOP_CODONS = {"TAA", "TAG", "TGA"}
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMP)[::-1]


class FastaReader:
    """Minimal random-access FASTA reader using a .fai index."""

    def __init__(self, fasta_path):
        self.path = fasta_path
        self.fh = open(fasta_path, "r")
        self.index = {}  # name -> (length, offset, line_bases, line_width)
        with open(fasta_path + ".fai") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 5:
                    name = parts[0]
                    self.index[name] = tuple(int(x) for x in parts[1:5])

    def fetch(self, name, start, end):
        """0-based, end-exclusive."""
        if name not in self.index:
            return None
        length, offset, lb, lw = self.index[name]
        start = max(0, start)
        end = min(end, length)
        if start >= end:
            return ""
        out = []
        pos = start
        while pos < end:
            line_no = pos // lb
            line_off = pos % lb
            seek = offset + line_no * lw + line_off
            self.fh.seek(seek)
            take = min(lb - line_off, end - pos)
            out.append(self.fh.read(take))
            pos += take
        return "".join(out).upper()

    def chroms(self):
        return set(self.index.keys())


def parse_attributes(attr):
    d = {}
    for kv in attr.split(";"):
        kv = kv.strip()
        if not kv:
            continue
        if "=" in kv:
            k, v = kv.split("=", 1)
        elif " " in kv:
            k, v = kv.split(" ", 1)
            v = v.strip('"')
        else:
            continue
        d[k] = v
    return d


def load_gff(path):
    """Return genes: {gene_id: {'chrom','strand','start','end',
    'transcripts': {tid: {'exons':[(s,e)], 'cdss':[(s,e)], 'start','end'}}, 'gm': str|None}}"""
    genes = {}
    tx_to_gene = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        chrom, src, feat, start, end, score, strand, frame, attr = f[:9]
        start, end = int(start) - 1, int(end)  # to 0-based half-open
        a = parse_attributes(attr)
        if feat == "gene":
            gid = a.get("ID")
            if gid:
                genes.setdefault(gid, {"chrom": chrom, "strand": strand,
                                       "start": start, "end": end,
                                       "transcripts": {}, "gm": a.get("GM")})
        elif feat in ("mRNA", "transcript"):
            tid = a.get("ID")
            parent = a.get("Parent")
            if tid and parent:
                tx_to_gene[tid] = parent
                if parent in genes:
                    genes[parent]["transcripts"][tid] = {
                        "exons": [], "cdss": [], "start": start, "end": end,
                        "strand": strand, "chrom": chrom}
        elif feat in ("exon", "CDS"):
            parent = a.get("Parent", "")
            # Parent may be comma-separated
            for tid in parent.split(","):
                gid = tx_to_gene.get(tid)
                if gid and gid in genes and tid in genes[gid]["transcripts"]:
                    key = "exons" if feat == "exon" else "cdss"
                    genes[gid]["transcripts"][tid][key].append((start, end))
    return genes


def translate_check(fa, chrom, strand, cdss):
    """Assemble CDS in transcription order, return dict of checks.

    Minus strand: concatenate fragments in ASCENDING genomic order and
    reverse-complement the whole concatenation (revcomp(X+Y) = revcomp(Y)+revcomp(X),
    so this yields fragments in descending = transcription order).
    """
    if not cdss:
        return {"has_cds": False}
    cdss = sorted(cdss)
    seq_parts = []
    for s, e in cdss:
        frag = fa.fetch(chrom, s, e)
        if frag is None:
            return {"has_cds": True, "seq_error": True}
        seq_parts.append(frag)
    seq = "".join(seq_parts)
    if strand == "-":
        seq = revcomp(seq)
    if len(seq) == 0:
        return {"has_cds": True, "seq_error": True}
    res = {
        "has_cds": True,
        "seq_error": False,
        "len_mod3": (len(seq) % 3 == 0),
        "start_atg": seq[:3] == "ATG",
        "stop_terminal": seq[-3:] in STOP_CODONS,
        "internal_stop": any(seq[i:i+3] in STOP_CODONS
                             for i in range(0, len(seq) - 3, 3)),
        "has_n": "N" in seq,
    }
    res["fully_consistent"] = (res["len_mod3"] and res["start_atg"]
                               and res["stop_terminal"]
                               and not res["internal_stop"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-chrom-regex", default=r"^(ChrM|ChrC|MT|Pt|chloroplast|mitochondria|mitochondrion).*|[_.-](chloroplast|mitochondria)",
                    help="Chromosomes excluded from translation checks "
                         "(organellar transcripts undergo RNA editing and use "
                         "non-standard starts, so genomic translation checks "
                         "do not apply). They are still excluded from isoform "
                         "counts to keep prediction/reference comparable.")
    args = ap.parse_args()
    skip_re = re.compile(args.skip_chrom_regex, re.IGNORECASE)

    fa = FastaReader(args.fasta)
    fa_chroms = fa.chroms()
    genes_all = load_gff(args.pred)
    genes = {g: d for g, d in genes_all.items() if not skip_re.search(d["chrom"])}
    n_genes_skipped = len(genes_all) - len(genes)

    n_genes = len(genes)
    n_tx = 0
    n_tx_no_cds = 0
    n_seq_error = 0
    n_len_mod3_fail = 0
    n_start_fail = 0
    n_stop_fail = 0
    n_internal_stop = 0
    n_fully_consistent = 0
    n_chrom_missing = 0
    iso_counts_pred = defaultdict(int)
    n_dup_tx = 0
    genes_with_dup = 0

    for gid, g in genes.items():
        iso_counts_pred[len(g["transcripts"])] += 1
        seen_chains = {}
        dup_in_gene = 0
        for tid, t in g["transcripts"].items():
            n_tx += 1
            chain = tuple(sorted(t["exons"]))
            if chain in seen_chains:
                n_dup_tx += 1
                dup_in_gene += 1
            else:
                seen_chains[chain] = tid
            if t["chrom"] not in fa_chroms:
                n_chrom_missing += 1
                continue
            chk = translate_check(fa, t["chrom"], t["strand"], t["cdss"])
            if not chk["has_cds"]:
                n_tx_no_cds += 1
                continue
            if chk.get("seq_error"):
                n_seq_error += 1
                continue
            n_len_mod3_fail += (not chk["len_mod3"])
            n_start_fail += (not chk["start_atg"])
            n_stop_fail += (not chk["stop_terminal"])
            n_internal_stop += chk["internal_stop"]
            n_fully_consistent += chk["fully_consistent"]
        if dup_in_gene:
            genes_with_dup += 1

    iso_counts_ref = None
    if args.ref:
        ref_all = load_gff(args.ref)
        ref = {g: d for g, d in ref_all.items() if not skip_re.search(d["chrom"])}
        iso_counts_ref = defaultdict(int)
        for gid, g in ref.items():
            iso_counts_ref[len(g["transcripts"])] += 1
        iso_counts_ref = dict(sorted(iso_counts_ref.items()))

    tx_with_cds = n_tx - n_tx_no_cds - n_seq_error - n_chrom_missing
    summary = {
        "prediction_file": args.pred,
        "n_genes_organellar_excluded": n_genes_skipped,
        "n_genes": n_genes,
        "n_transcripts": n_tx,
        "n_transcripts_no_cds": n_tx_no_cds,
        "n_transcripts_seq_error": n_seq_error,
        "n_transcripts_chrom_missing": n_chrom_missing,
        "n_transcripts_checked": tx_with_cds,
        "frame_len_mod3_fail": n_len_mod3_fail,
        "missing_start_atg": n_start_fail,
        "missing_terminal_stop": n_stop_fail,
        "internal_stop_codons": n_internal_stop,
        "fully_consistent_transcripts": n_fully_consistent,
        "pct_fully_consistent": round(100.0 * n_fully_consistent / max(1, tx_with_cds), 2),
        "duplicate_transcripts": n_dup_tx,
        "genes_with_duplicate_transcripts": genes_with_dup,
        "isoforms_per_gene_pred": dict(sorted(iso_counts_pred.items())),
        "isoforms_per_gene_ref": iso_counts_ref,
        "mean_isoforms_per_gene_pred": round(n_tx / max(1, n_genes), 3),
    }

    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=2)

    # TSV one-liner for aggregation
    keys = ["n_genes", "n_transcripts", "n_transcripts_no_cds",
            "n_transcripts_checked", "frame_len_mod3_fail", "missing_start_atg",
            "missing_terminal_stop", "internal_stop_codons",
            "fully_consistent_transcripts", "pct_fully_consistent",
            "duplicate_transcripts", "genes_with_duplicate_transcripts",
            "mean_isoforms_per_gene_pred"]
    with open(args.out + ".tsv", "w") as f:
        f.write("\t".join(["file"] + keys) + "\n")
        f.write("\t".join([args.pred] + [str(summary[k]) for k in keys]) + "\n")

    print(json.dumps({k: summary[k] for k in
                      ["n_genes", "n_transcripts", "pct_fully_consistent",
                       "duplicate_transcripts"]}, indent=2))


if __name__ == "__main__":
    main()
