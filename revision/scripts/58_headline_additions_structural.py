#!/usr/bin/env python3
"""Structural statistics for the 1,103 headline additions of the reference-prompted run.

WHY THIS EXISTS

The manuscript reports the reference-blind structural filter of
`36_filter_additions_structurally.py` (complete ORF, canonical GT..AG introns) for the
prompt-transfer re-runs only — 245 of 1,014 additions kept at 71.4% precision for the
reference condition (Table S11) — and never for the 1,103 additions that
`28_score_added_isoforms.py` scores on the full 27,413-locus prompted prediction (Table
S4d, 18.1% versus TAIR10 alternatives). This script applies the identical predicates to
those 1,103 structures and scores the kept subset with script 28's definitions, so the
headline set has the same ORF / splice / filtered-precision figures as the re-run.

CRITERIA (script 36, imported not copied)

    complete ORF        starts ATG, ends in a stop codon, length divisible by three,
                        no internal stop
    canonical introns   every intron begins GT and ends AG on the transcribed strand
    both                the filter as applied in Table S11

The addition set is exactly script 28's: distinct CDS structures per TAIR10 locus that
differ from TAIR10's curated primary transcript (`primary_transcript_ids.txt`), read from
`A_thaliana_transgenic400Mprompt_beam1.gff3` (the top-beam export, one record per
transcript).

SANITY CHECKS (the script refuses to write results if any fails)

    1. the addition set must be 1,103 structures with 200 TAIR10-alternative and 204
       AtRTD3 matches (script 28)
    2. this script's per-structure loop, run on the reference-condition re-run
       (tair10selfutr), must return the same counts as script 36's own
       `filter_predictions` and its archived `tair10selfutr_filter.json`
       (1,014 additions, 245 kept, 617 failed ORF, 152 failed splice)
    3. the manuscript's "35.7% of additions encoding a complete reading frame under
       TAIR10 coding sequence" is 4,512 / 12,628 on the Helixer-boundary arm
       (tair10helixerframeutr), not the reference condition; it is reproduced from that
       arm's archived filter JSON and by re-running script 36 on it.

Usage:
    python 58_headline_additions_structural.py [--json revision/results/baselines/headline_additions_structural.json]
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
s28 = importlib.import_module("28_score_added_isoforms")
s36 = importlib.import_module("36_filter_additions_structurally")

ROOT = HERE.parents[2]
CMP = ROOT / "transgenic_comparison"
DATA = ROOT / "transgenic" / "revision" / "data"
BENCH = ROOT / "polishing_benchmark"
ARCHIVE = ROOT / "transgenic" / "revision" / "results" / "prompt_transfer"

TRANSGENIC_PRED = CMP / "standardized_results" / "A_thaliana_transgenic400Mprompt_beam1.gff3"
TAIR10 = DATA / "TAIR10" / "TAIR10.gtf"
ATRTD3 = DATA / "AtRTD3" / "atRTD3_TS_21Feb22_transfix.gtf"
PRIMARY_IDS = DATA / "TAIR10" / "primary_transcript_ids.txt"
GENOME = DATA / "TAIR10" / "TAIR10_genome.fa"

HEADLINE = {"added_structures": 1103, "matched_TAIR10_alt_exact_CDS": 200,
            "matched_AtRTD3": 204, "reference_alternative_structures": 5580}
REFERENCE_CONDITION = {"additions": 1014, "additions_kept": 245,
                       "failed_orf": 617, "failed_splice": 152}
HELIXER_FRAME_ARM = {"additions": 12628, "failed_orf": 8116}   # 4,512 / 12,628 = 35.7%


def load_primary(path: Path) -> dict[str, str]:
    primary: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            primary[line.strip().split(".")[0]] = line.strip()
    return primary


def additions_with_flags(completed_gff: Path, supplied: dict, genome: dict,
                         locus_of_gene) -> list[dict]:
    """One record per distinct (locus, CDS structure) that differs from the supplied one.

    Same walk as script 36's `filter_predictions`, but keeping every structure with both
    predicate outcomes instead of stopping at the first failure, so ORF-only,
    splice-only and combined counts all come from one pass.
    """
    structures, gm_of, meta_of = s36.read_cds_by_transcript(completed_gff)
    seen: set = set()
    records: list[dict] = []
    for tx, struct in structures.items():
        gene = gm_of.get(tx)
        locus = locus_of_gene(gene)
        if locus is None:
            continue                       # script 28 drops "-rc" loci
        if supplied.get(locus) == struct:
            continue
        key = (locus, struct)
        if key in seen:
            continue
        seen.add(key)
        seqid, strand = meta_of[tx]
        chromosome = genome.get(seqid)
        if chromosome is None:
            raise RuntimeError(f"unknown sequence {seqid!r} in {completed_gff}")
        orf = s36.has_complete_orf(s36.spliced_cds(chromosome, struct, strand))
        splice = s36.has_canonical_introns(chromosome, struct, strand)
        records.append({"locus": locus, "struct": struct, "seqid": seqid, "strand": strand,
                        "complete_orf": orf, "canonical_introns": splice,
                        "both": orf and splice})
    return records


def attach_matches(records: list[dict], ref: dict, art: dict, primary: dict) -> None:
    for r in records:
        locus = r["locus"]
        ref_primary = ref.get(locus, {}).get(primary.get(locus))
        alt_ref = {s for t, s in ref.get(locus, {}).items() if s != ref_primary}
        alt_chains = {s28.chain(x) for x in alt_ref}
        r["hit_tair10_alt"] = r["struct"] in alt_ref
        r["hit_tair10_chain"] = len(s28.chain(r["struct"])) >= 1 and s28.chain(r["struct"]) in alt_chains
        r["hit_atrtd3"] = r["struct"] in set(art.get(locus, {}).values())


def pct(n: int, d: int) -> float | None:
    return round(100 * n / d, 1) if d else None


def summarize(records: list[dict], denominator: int, flag: str | None) -> dict:
    subset = [r for r in records if flag is None or r[flag]]
    n = len(subset)
    t10 = sum(r["hit_tair10_alt"] for r in subset)
    chain = sum(r["hit_tair10_chain"] for r in subset)
    a3 = sum(r["hit_atrtd3"] for r in subset)
    return {
        "added_structures": n,
        "loci_with_at_least_one_addition": len({r["locus"] for r in subset}),
        "matched_TAIR10_alt_exact_CDS": t10,
        "matched_TAIR10_alt_intron_chain": chain,
        "matched_AtRTD3": a3,
        "precision_vs_TAIR10_alt_pct": pct(t10, n),
        "precision_vs_AtRTD3_pct": pct(a3, n),
        "reference_alternative_structures": denominator,
        "recall_of_TAIR10_alt_pct": pct(t10, denominator),
    }


def structural_counts(records: list[dict]) -> dict:
    n = len(records)
    orf = sum(r["complete_orf"] for r in records)
    spl = sum(r["canonical_introns"] for r in records)
    both = sum(r["both"] for r in records)
    return {
        "additions": n,
        "complete_orf": orf, "complete_orf_pct": pct(orf, n),
        "canonical_introns": spl, "canonical_introns_pct": pct(spl, n),
        "both": both, "both_pct": pct(both, n),
        "neither": sum(1 for r in records if not r["complete_orf"] and not r["canonical_introns"]),
        # script 36's sequential accounting, for direct comparison with *_filter.json
        "failed_orf": n - orf,
        "failed_splice_given_orf": sum(1 for r in records if r["complete_orf"] and not r["canonical_introns"]),
        "additions_kept": both,
        "additions_discarded_pct": pct(n - both, n),
    }


def check(name: str, got: dict, want: dict) -> bool:
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    if bad:
        print(f"  SANITY FAIL {name}: {bad}", file=sys.stderr)
        return False
    print(f"  sanity ok    {name}: {', '.join(f'{k}={v}' for k, v in want.items())}",
          file=sys.stderr)
    return True


def transfer_arm(tool: str, genome: dict, ref: dict, art: dict, primary: dict) -> dict:
    """Run script 36 itself AND this script's loop on one prompt-transfer arm."""
    prompt = BENCH / "inputs" / f"{tool}_Athaliana.gff3"
    completed = BENCH / "predictions" / f"{tool}_completed.gff3"
    archived = json.loads((ARCHIVE / f"{tool}_filter.json").read_text())
    s36_stats = s36.filter_predictions(prompt, completed, GENOME, None)
    # Gene ids in the arm files carry TAIR10's ".TAIR10" suffix; strip it so the TAIR10 /
    # AtRTD3 lookups work, and keep GM-less transcripts as script 36 does (it counts them).
    def locus_of_gene(gene: str | None) -> str:
        return gene.replace(".TAIR10", "") if gene is not None else "__no_GM__"

    supplied = {locus_of_gene(g): s for g, s in s36.supplied_structures(prompt).items()}
    records = additions_with_flags(completed, supplied, genome, locus_of_gene)
    attach_matches(records, ref, art, primary)
    mine = structural_counts(records)
    return {"tool": tool, "prompt": str(prompt), "completed": str(completed),
            "archived_filter_json": archived, "script36_rerun": s36_stats,
            "this_script": mine,
            "kept_subset_scored_script28_definitions": summarize(records, 0, "both"),
            "all_additions_scored_script28_definitions": summarize(records, 0, None)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", type=Path,
                    default=ROOT / "transgenic" / "revision" / "results" / "baselines"
                    / "headline_additions_structural.json")
    ap.add_argument("--skip-transfer-check", action="store_true",
                    help="skip re-running script 36 on the prompt-transfer arms")
    args = ap.parse_args()
    args.json.parent.mkdir(parents=True, exist_ok=True)

    print("  reading genome and references ...", file=sys.stderr)
    genome = s36.load_genome(GENOME)
    ref = s28.cds_by_transcript(TAIR10, False)
    art = s28.cds_by_transcript(ATRTD3, False)
    primary = load_primary(PRIMARY_IDS)

    # --- the headline set: script 28's addition definition ------------------------------
    supplied = {locus: ref.get(locus, {}).get(primary.get(locus)) for locus in ref}

    def locus_of_gene(gene: str | None):
        if gene is None or gene.endswith("-rc"):
            return None
        return gene.replace(".TAIR10", "")

    records = additions_with_flags(TRANSGENIC_PRED, supplied, genome, locus_of_gene)
    attach_matches(records, ref, art, primary)
    pred_loci = {locus_of_gene(g) for g in s36.read_cds_by_transcript(TRANSGENIC_PRED)[1].values()}
    pred_loci.discard(None)
    denominator = 0
    for locus in pred_loci:
        ref_primary = ref.get(locus, {}).get(primary.get(locus))
        denominator += len({s for t, s in ref.get(locus, {}).items() if s != ref_primary})

    unfiltered = summarize(records, denominator, None)
    unfiltered["loci_scored"] = len(pred_loci)
    ok = check("headline additions reproduce script 28", unfiltered, HEADLINE)

    structural = structural_counts(records)
    filtered = {
        "complete_orf_only": summarize(records, denominator, "complete_orf"),
        "canonical_introns_only": summarize(records, denominator, "canonical_introns"),
        "both_complete_orf_and_canonical_introns": summarize(records, denominator, "both"),
    }
    matches_kept = {
        "TAIR10_alt_matches_kept_by_filter":
            f"{filtered['both_complete_orf_and_canonical_introns']['matched_TAIR10_alt_exact_CDS']} of "
            f"{unfiltered['matched_TAIR10_alt_exact_CDS']}",
        "AtRTD3_matches_kept_by_filter":
            f"{filtered['both_complete_orf_and_canonical_introns']['matched_AtRTD3']} of "
            f"{unfiltered['matched_AtRTD3']}",
    }
    # Structural quality of the correct versus incorrect additions.
    by_correct = {}
    for label, flag in (("matched_TAIR10_alt", True), ("unmatched_TAIR10_alt", False)):
        sub = [r for r in records if r["hit_tair10_alt"] == flag]
        by_correct[label] = structural_counts(sub)

    # --- sanity on the prompt-transfer arms --------------------------------------------
    transfer = {}
    if not args.skip_transfer_check:
        for tool, want in (("tair10selfutr", REFERENCE_CONDITION),
                           ("tair10helixerframeutr", HELIXER_FRAME_ARM)):
            print(f"  re-running script 36 on {tool} ...", file=sys.stderr)
            arm = transfer_arm(tool, genome, ref, art, primary)
            transfer[tool] = arm
            ok &= check(f"{tool}: archived filter JSON", arm["archived_filter_json"], want)
            ok &= check(f"{tool}: script 36 re-run", arm["script36_rerun"], want)
            mine = {k: arm["this_script"][k] for k in ("additions", "additions_kept", "failed_orf")}
            mine["failed_splice"] = arm["this_script"]["failed_splice_given_orf"]
            ok &= check(f"{tool}: this script's loop", mine, want)
            orf_pct = arm["this_script"]["complete_orf_pct"]
            print(f"  {tool}: complete ORF {arm['this_script']['complete_orf']} / "
                  f"{arm['this_script']['additions']} = {orf_pct}%", file=sys.stderr)
    if not ok:
        print("  refusing to write results: a sanity check failed", file=sys.stderr)
        return 1

    out = {
        "analysis": "structural filter statistics for the headline 1,103 additions (Table S4d set)",
        "date": date.today().isoformat(),
        "addition_definition": ("script 28: distinct CDS structure per TAIR10 locus differing from "
                                "TAIR10's curated primary transcript; -rc loci dropped"),
        "criteria": {"complete_orf": "ATG start, stop end, length % 3 == 0, no internal stop",
                     "canonical_introns": "every intron GT..AG on the transcribed strand "
                                          "(single-exon structures pass)",
                     "source": "36_filter_additions_structurally.py, functions imported"},
        "prediction": str(TRANSGENIC_PRED),
        "loci_scored": len(pred_loci),
        "structural": structural,
        "unfiltered": unfiltered,
        "filtered": filtered,
        "matches_kept": matches_kept,
        "structural_by_correctness": by_correct,
        "prompt_transfer_sanity": {
            "note": ("The manuscript's 35.7% complete-ORF figure is the Helixer-boundary arm "
                     "(tair10helixerframeutr: 4,512 / 12,628); the reference condition "
                     "(tair10selfutr) is 397 / 1,014 = 39.2%, with 245 / 1,014 passing both criteria."),
            "arms": transfer,
        },
        "inputs": {"genome": str(GENOME), "TAIR10": str(TAIR10), "AtRTD3": str(ATRTD3),
                   "primary_transcript_ids": str(PRIMARY_IDS)},
    }
    args.json.write_text(json.dumps(out, indent=1) + "\n")

    s, f = structural, filtered["both_complete_orf_and_canonical_introns"]
    print(f"  headline: {s['additions']} additions; complete ORF {s['complete_orf']} "
          f"({s['complete_orf_pct']}%); canonical introns {s['canonical_introns']} "
          f"({s['canonical_introns_pct']}%); both {s['both']} ({s['both_pct']}%)", file=sys.stderr)
    print(f"  filtered subset: {f['added_structures']} kept; TAIR10-alt "
          f"{f['matched_TAIR10_alt_exact_CDS']} ({f['precision_vs_TAIR10_alt_pct']}%); AtRTD3 "
          f"{f['matched_AtRTD3']} ({f['precision_vs_AtRTD3_pct']}%); recall "
          f"{f['recall_of_TAIR10_alt_pct']}%", file=sys.stderr)
    print(f"written: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
