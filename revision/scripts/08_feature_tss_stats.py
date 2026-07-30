#!/usr/bin/env python3
"""
Feature-count statistics (Reviewer 3.1), TSS/TES accuracy (Reviewer 3.3),
and training-set AS statistics (Reviewer 2, methods).

Modes:
  --mode features  : per-gene unique CDS / 5'UTR / 3'UTR / exon segment counts
                     in a reference annotation; checks against GSF vocabulary
                     limits (150 CDS, 50 5'UTR, 50 3'UTR).
  --mode tss_tes   : TSS/TES accuracy of TransGenic predictions, linking
                     predicted genes to reference genes via the GM attribute.
  --mode asstats   : alternative splicing statistics of a reference
                     (genes, transcripts, multi-transcript genes).

Usage:
  python 08_feature_tss_stats.py --mode features --gff ref.gff3 --out prefix
  python 08_feature_tss_stats.py --mode tss_tes --pred pred.gff3 --ref ref.gff3 --out prefix
  python 08_feature_tss_stats.py --mode asstats --gff ref.gff3 --out prefix
"""

import argparse
import json
import re
from collections import defaultdict

UTR5_NAMES = {"five_prime_utr", "five_prime_utrs", "5utr", "5'-utr", "utr5",
              "five_prime"}
UTR3_NAMES = {"three_prime_utr", "three_prime_utrs", "3utr", "3'-utr", "utr3",
              "three_prime"}
SKIP_CHROM = re.compile(r"^(ChrM|ChrC|MT|Pt)|chloroplast|mitochondria", re.I)


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
    """genes: {gid: {chrom,strand,start,end,gm, transcripts:{tid:{exons,cdss,utr5,utr3,start,end,strand}}}}"""
    genes = {}
    tx_to_gene = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        chrom, src, feat, start, end, score, strand, frame, attr = f[:9]
        start, end = int(start) - 1, int(end)
        a = parse_attributes(attr)
        fl = feat.lower()
        if fl == "gene":
            gid = a.get("ID")
            if gid:
                genes[gid] = {"chrom": chrom, "strand": strand, "start": start,
                              "end": end, "gm": a.get("GM"), "transcripts": {}}
        elif fl in ("mrna", "transcript"):
            tid = a.get("ID") or a.get("transcript_id")
            parent = a.get("Parent") or a.get("gene_id")
            if tid and parent and parent in genes:
                tx_to_gene[tid] = parent
                genes[parent]["transcripts"][tid] = {
                    "exons": [], "cdss": [], "utr5": [], "utr3": [],
                    "start": start, "end": end, "strand": strand,
                    "chrom": chrom}
        else:
            key = None
            if fl == "exon":
                key = "exons"
            elif fl == "cds":
                key = "cdss"
            elif fl in UTR5_NAMES:
                key = "utr5"
            elif fl in UTR3_NAMES:
                key = "utr3"
            if key:
                parent = a.get("Parent") or a.get("transcript_id") or ""
                # GTF-style files without gene/transcript feature lines:
                # create gene+transcript shells from gene_id/transcript_id
                if "transcript_id" in a and "gene_id" in a:
                    gid = a["gene_id"]
                    tid = a["transcript_id"]
                    if gid not in genes:
                        genes[gid] = {"chrom": chrom, "strand": strand,
                                      "start": start, "end": end, "gm": None,
                                      "transcripts": {}}
                    if tid not in genes[gid]["transcripts"]:
                        genes[gid]["transcripts"][tid] = {
                            "exons": [], "cdss": [], "utr5": [], "utr3": [],
                            "start": start, "end": end, "strand": strand,
                            "chrom": chrom}
                        tx_to_gene[tid] = gid
                for tid in parent.split(","):
                    gid = tx_to_gene.get(tid)
                    if gid and tid in genes[gid]["transcripts"]:
                        genes[gid]["transcripts"][tid][key].append((start, end))
    return genes


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[i]


def feature_stats(gff, out):
    genes = load_gff(gff)
    genes = {g: d for g, d in genes.items() if not SKIP_CHROM.search(d["chrom"])}
    cds_counts, u5_counts, u3_counts, exon_counts, tx_counts = [], [], [], [], []
    over_limit = {"cds": 0, "utr5": 0, "utr3": 0, "any": 0}
    has_utr_annotation = False
    for gid, g in genes.items():
        u_cds, u_5, u_3, u_ex = set(), set(), set(), set()
        for tid, t in g["transcripts"].items():
            u_cds.update(t["cdss"])
            u_5.update(t["utr5"])
            u_3.update(t["utr3"])
            u_ex.update(t["exons"])
        if u_5 or u_3:
            has_utr_annotation = True
        cds_counts.append(len(u_cds))
        u5_counts.append(len(u_5))
        u3_counts.append(len(u_3))
        exon_counts.append(len(u_ex))
        tx_counts.append(len(g["transcripts"]))
        over = False
        if len(u_cds) > 150:
            over_limit["cds"] += 1
            over = True
        if len(u_5) > 50:
            over_limit["utr5"] += 1
            over = True
        if len(u_3) > 50:
            over_limit["utr3"] += 1
            over = True
        if over:
            over_limit["any"] += 1
    n = max(1, len(genes))
    for arr in (cds_counts, u5_counts, u3_counts, exon_counts, tx_counts):
        arr.sort()

    def block(arr):
        return {"max": arr[-1] if arr else 0,
                "p50": pct(arr, 0.50), "p90": pct(arr, 0.90),
                "p99": pct(arr, 0.99), "p999": pct(arr, 0.999)}

    res = {
        "file": gff,
        "n_genes": len(genes),
        "has_utr_annotation": has_utr_annotation,
        "unique_cds_per_gene": block(cds_counts),
        "unique_utr5_per_gene": block(u5_counts),
        "unique_utr3_per_gene": block(u3_counts),
        "unique_exons_per_gene": block(exon_counts),
        "transcripts_per_gene": block(tx_counts),
        "genes_over_150_cds": over_limit["cds"],
        "genes_over_50_utr5": over_limit["utr5"],
        "genes_over_50_utr3": over_limit["utr3"],
        "genes_over_any_limit": over_limit["any"],
        "pct_genes_within_limits": round(100.0 * (n - over_limit["any"]) / n, 3),
    }
    with open(out + ".json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def resolve_gm(gm, ref_genes):
    """Resolve GM attribute to a reference gene ID (strip version suffixes)."""
    if gm in ref_genes:
        return gm
    parts = gm.split(".")
    for i in range(len(parts) - 1, 0, -1):
        cand = ".".join(parts[:i])
        if cand in ref_genes:
            return cand
    # prefix match both directions
    for rid in ref_genes:
        if rid.startswith(gm + ".") or gm.startswith(rid + "."):
            return rid
    return None


def tss_tes(pred, ref, out):
    preds = load_gff(pred)
    refs = load_gff(ref)
    deltas = []
    n_genes_linked = 0
    n_genes_unlinked = 0
    for gid, g in preds.items():
        if SKIP_CHROM.search(g["chrom"]):
            continue
        if not g["gm"]:
            n_genes_unlinked += 1
            continue
        rid = resolve_gm(g["gm"], refs)
        if not rid:
            n_genes_unlinked += 1
            continue
        n_genes_linked += 1
        rg = refs[rid]
        ref_tx = [(t["start"], t["end"], t["strand"])
                  for t in rg["transcripts"].values()]
        if not ref_tx:
            continue
        for tid, t in g["transcripts"].items():
            # TSS/TES on the transcript's own strand
            if t["strand"] == "+":
                p_tss, p_tes = t["start"], t["end"]
            else:
                p_tss, p_tes = t["end"], t["start"]
            same_strand = [(rs, re_) for rs, re_, st in ref_tx if st == t["strand"]]
            use = same_strand if same_strand else [(rs, re_) for rs, re_, _ in ref_tx]
            best = None
            for rs, re_ in use:
                if t["strand"] == "+":
                    r_tss, r_tes = rs, re_
                else:
                    r_tss, r_tes = re_, rs
                d = (abs(p_tss - r_tss), abs(p_tes - r_tes))
                if best is None or (d[0] + d[1]) < (best[0] + best[1]):
                    best = d
            if best:
                deltas.append(best)

    def agg(idx):
        arr = sorted(d[idx] for d in deltas)
        n = max(1, len(arr))
        return {
            "n": len(arr),
            "exact_pct": round(100.0 * sum(1 for d in arr if d == 0) / n, 2),
            "within_10nt_pct": round(100.0 * sum(1 for d in arr if d <= 10) / n, 2),
            "within_50nt_pct": round(100.0 * sum(1 for d in arr if d <= 50) / n, 2),
            "within_100nt_pct": round(100.0 * sum(1 for d in arr if d <= 100) / n, 2),
            "median_delta": arr[len(arr) // 2] if arr else None,
        }

    res = {
        "prediction_file": pred,
        "n_genes_linked": n_genes_linked,
        "n_genes_unlinked": n_genes_unlinked,
        "n_transcripts_compared": len(deltas),
        "TSS": agg(0),
        "TES": agg(1),
    }
    with open(out + ".json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def asstats(gff, out):
    genes = load_gff(gff)
    genes = {g: d for g, d in genes.items() if not SKIP_CHROM.search(d["chrom"])}
    n_genes = len(genes)
    tx_counts = [len(g["transcripts"]) for g in genes.values()]
    n_tx = sum(tx_counts)
    multi = sum(1 for c in tx_counts if c > 1)
    res = {
        "file": gff,
        "n_genes": n_genes,
        "n_transcripts": n_tx,
        "multi_transcript_genes": multi,
        "pct_multi_transcript": round(100.0 * multi / max(1, n_genes), 2),
        "mean_transcripts_per_gene": round(n_tx / max(1, n_genes), 3),
        "max_transcripts_per_gene": max(tx_counts) if tx_counts else 0,
    }
    with open(out + ".json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["features", "tss_tes", "asstats"])
    ap.add_argument("--gff")
    ap.add_argument("--pred")
    ap.add_argument("--ref")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "features":
        feature_stats(args.gff, args.out)
    elif args.mode == "tss_tes":
        tss_tes(args.pred, args.ref, args.out)
    else:
        asstats(args.gff, args.out)


if __name__ == "__main__":
    main()
