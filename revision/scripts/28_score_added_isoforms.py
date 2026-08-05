#!/usr/bin/env python3
"""Score only the isoforms TransGenic adds, with the supplied transcript removed.

Reviewer 2 asked what percentage of alternatively spliced isoforms are predicted correctly
at transcript level, and noted for the AUGUSTUS baseline that the given isoform should be
"removed or ignored". The alternative-transcript-only evaluation removes the primary from
the *reference* but not from the *prediction*, so every retained prompt is scored as a false
positive and the precision it reports (2.5%) describes the prompt, not the additions.

This scores the additions alone. A predicted transcript is an *addition* when its CDS
structure differs from the primary transcript supplied for that locus; the reference is the
alternative transcripts of the same annotation, primary removed. Both sides therefore
exclude the supplied model, which is the comparison the reviewer asked for.

Structures are compared on exact CDS coordinates, the same criterion used for the
composition figures in the Results, and separately on the CDS intron chain, which ignores
the first and last CDS boundaries and is the stricter-to-satisfy analogue of GFFCompare's
'=' class code.

Usage:
    python 28_score_added_isoforms.py [--species A_thaliana] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CMP = ROOT / "transgenic_comparison"
DATA = ROOT / "transgenic" / "revision" / "data"


def cds_by_transcript(path: Path, gff3: bool) -> dict:
    d: dict = defaultdict(lambda: defaultdict(list))
    lr = re.compile(r"GM=([^;\s]+)") if gff3 else re.compile(r'gene_id "([^"]+)"')
    tr = re.compile(r"Parent=([^;]+)") if gff3 else re.compile(r'transcript_id "([^"]+)"')
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            g, t = lr.search(f[8]), tr.search(f[8])
            if not (g and t):
                continue
            locus = g.group(1)
            if locus.endswith("-rc"):
                continue
            d[locus.replace(".TAIR10", "")][t.group(1)].append((int(f[3]), int(f[4])))
    return {g: {k: tuple(sorted(v)) for k, v in tx.items()} for g, tx in d.items()}


def chain(struct: tuple) -> tuple:
    return tuple((struct[i][1], struct[i + 1][0]) for i in range(len(struct) - 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="A_thaliana")
    ap.add_argument("--augustus", action="store_true",
                    help="score the AUGUSTUS posterior-sampling baseline the same way, "
                         "dropping its own first prediction per locus in place of the prompt")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.augustus:
        # AUGUSTUS carries no GM= on CDS rows and no supplied transcript; its own first
        # prediction per locus stands in for the prompt so both tools are scored on what
        # they propose beyond one primary structure.
        pred = defaultdict(dict)
        raw: dict = defaultdict(lambda: defaultdict(list))
        src = CMP / "standardized_results" / f"{args.species}_augustusSampling.gff3"
        with src.open() as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 9 or f[2] != "CDS":
                    continue
                m = re.search(r"Parent=([^;]+)", f[8])
                if not m:
                    continue
                tx = m.group(1)
                raw[tx.split(".t")[0].replace("augSmp_", "")][tx].append((int(f[3]), int(f[4])))
        for locus, txs in raw.items():
            pred[locus] = {k: tuple(sorted(v)) for k, v in txs.items()}
    else:
        pred = cds_by_transcript(
            CMP / "standardized_results" / f"{args.species}_transgenic400Mprompt_beam1.gff3", True)
    ref = cds_by_transcript(DATA / "TAIR10" / "TAIR10.gtf", False)
    art = cds_by_transcript(DATA / "AtRTD3" / "atRTD3_TS_21Feb22_transfix.gtf", False)
    primary = {}
    for line in (DATA / "TAIR10" / "primary_transcript_ids.txt").read_text().splitlines():
        if line.strip():
            primary[line.strip().split(".")[0]] = line.strip()

    added = struct_hit = chain_hit = art_hit = 0
    alt_total = alt_recovered = 0
    loci_with_addition = 0
    for locus, txs in pred.items():
        if args.augustus:
            order = sorted(txs, key=lambda k: int(re.search(r"\.t(\d+)$", k).group(1))
                           if re.search(r"\.t(\d+)$", k) else 0)
            supplied = txs[order[0]] if order else None
        else:
            supplied = ref.get(locus, {}).get(primary.get(locus))
        additions = [s for s in txs.values() if s != supplied]
        # De-duplicate: two beams or two identical emissions are one proposed structure.
        additions = list({s for s in additions})
        # The reference side always drops TAIR10's own primary, whichever tool is being
        # scored: otherwise the AUGUSTUS run would be credited for recovering the primary
        # transcript, which is not an alternative isoform.
        ref_primary = ref.get(locus, {}).get(primary.get(locus))
        alt_ref = {s for t, s in ref.get(locus, {}).items() if s != ref_primary}
        art_here = set(art.get(locus, {}).values())
        alt_total += len(alt_ref)
        if additions:
            loci_with_addition += 1
        added += len(additions)
        struct_hit += sum(1 for s in additions if s in alt_ref)
        chain_hit += sum(1 for s in additions
                         if len(chain(s)) >= 1 and chain(s) in {chain(x) for x in alt_ref})
        art_hit += sum(1 for s in additions if s in art_here)
        alt_recovered += len(alt_ref & set(additions))

    out = {
        "species": args.species,
        "tool": "augustusSampling" if args.augustus else "transgenic400Mprompt_beam1",
        "loci_scored": len(pred),
        "loci_with_at_least_one_addition": loci_with_addition,
        "added_transcripts": added,
        "reference_alternative_transcripts": alt_total,
        "added_matching_TAIR10_alternative_exact_CDS": struct_hit,
        "added_matching_TAIR10_alternative_intron_chain": chain_hit,
        "added_matching_any_AtRTD3_transcript": art_hit,
        "precision_vs_TAIR10_alternatives_pct": round(100 * struct_hit / added, 1) if added else None,
        "precision_vs_AtRTD3_pct": round(100 * art_hit / added, 1) if added else None,
        "recall_of_TAIR10_alternatives_pct": round(100 * alt_recovered / alt_total, 1) if alt_total else None,
    }
    for k, v in out.items():
        print(f"  {k:<48} {v}", file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"written: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
