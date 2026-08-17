#!/usr/bin/env python3
"""Score TransGenic's *added* isoforms in Z. mays: the leakage-free control.

Context. The A. thaliana additions precision of 18.1% (200/1,103) was measured on an
evaluation set of which 78.4% of loci had been seen in training, and the held-out stratum
that could have decided the question is 6.3x depleted of multi-isoform loci, so it carries
no power (5/32, Fisher p = 0.82 against train+validation).

Z. mays was excluded from the training database by the "Zm" identifier prefix filter. Only
174 legacy GRMZM gene models (176 dataset rows) lack that prefix and therefore survived into
training. Everything else in RefGen_V4 is unseen, which makes the completion-mode additions
precision on Z. mays a leakage-free control for the same quantity.

Definitions are those of revision/scripts/28_score_added_isoforms.py, unchanged:
  - an *addition* is a predicted transcript whose CDS coordinate tuple differs from the
    primary transcript supplied as the prompt for that locus;
  - identical emissions are merged (one proposed structure, not two);
  - a hit is an exact CDS coordinate match against a non-primary reference transcript;
    the intron-chain variant (GFFCompare '=' analogue) is reported alongside;
  - the primary is the FIRST transcript of the gene in reference file order, the rule
    14_make_altonly_references.py used to build TAIR10's primary_transcript_ids.txt.

Two Z. mays specifics are handled here and do NOT exist in script 28:
  - the maize export writes every locus THREE times (A. thaliana wrote it twice and was
    deduplicated to A_thaliana_transgenic400Mprompt_beam1.gff3 by script 13; no such file
    exists for maize), so the first gene record per GM is kept and the discarded copies are
    checked for structural identity rather than assumed identical;
  - the reference is a Phytozome GFF3 without GM= tags, so transcript->gene is resolved
    through mRNA Parent= instead.

There is no long-read AtRTD3 analogue for maize, so RefGen_V4 alternatives are the only
reference; the A. thaliana AtRTD3 column has no counterpart and is not reported.

Because the definitions are reimplemented rather than imported (script 28 hard-codes TAIR10
paths and the GM=/gene_id parsing of two different file formats), --verify-athaliana reruns
this file's scoring function on the A. thaliana inputs and must reproduce 200/1,103 = 18.1%
before the maize number is worth reading.

Usage:
    python 48_score_zmays_additions.py [--json out.json] [--verify-athaliana]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/gpfs/assoc/pgl/data/Transgenic")
PRED = ROOT / "transgenic_comparison" / "standardized_results" / "Z_mays_transgenic400Mprompt.gff3"
REF = ROOT / "genomes" / "Zmays_493_RefGen_V4.gene_exons.exon.gff3"
OUTDIR = ROOT / "transgenic" / "revision" / "results" / "zmays_additions"

# A. thaliana completion-mode additions, all 27,413 evaluation loci, quoted for the
# side-by-side only: revision/results/heldout_additions/split_composition_and_precision_20260811.json
AT_ALL_HITS, AT_ALL_ADDED = 200, 1103
AT_VALIDATION_HITS, AT_VALIDATION_ADDED = 13, 47
AT_ALT_STRUCTURES, AT_LOCI = 5580, 27413

GM_RE = re.compile(r"GM=([^;\s]+)")
ID_RE = re.compile(r"ID=([^;\s]+)")
PARENT_RE = re.compile(r"Parent=([^;\s]+)")

AT_PRED = (ROOT / "transgenic_comparison" / "standardized_results"
           / "A_thaliana_transgenic400Mprompt_beam1.gff3")
AT_DATA = ROOT / "transgenic" / "revision" / "data" / "TAIR10"


def wilson95(hits: int, n: int) -> list | None:
    """Wilson score interval, the estimator used for the A. thaliana strata."""
    if not n:
        return None
    z = 1.959963984540054
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0.0, centre - half), 1), round(100 * min(1.0, centre + half), 1)]


def chain(struct: tuple) -> tuple:
    """CDS intron chain: ignores the outer CDS boundaries, as script 28 does."""
    return tuple((struct[i][1], struct[i + 1][0]) for i in range(len(struct) - 1))


def read_prediction(path: Path) -> tuple[dict, dict]:
    """First gene record per GM locus -> {transcript_id: CDS tuple}.

    The maize export repeats each locus; later records are collected separately so their
    structures can be compared against the kept record instead of being trusted blindly.
    """
    gene_order: dict = defaultdict(list)        # locus -> [gene_id, ...] in file order
    gene_locus: dict = {}                       # gene_id -> locus
    tx_gene: dict = {}                          # transcript_id -> gene_id
    cds: dict = defaultdict(list)               # transcript_id -> [(start, end), ...]
    no_gm_genes = 0

    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            kind, attrs = f[2], f[8]
            if kind == "gene":
                gm = GM_RE.search(attrs)
                gid = ID_RE.search(attrs)
                if not gid:
                    continue
                if not gm:
                    no_gm_genes += 1
                    continue
                locus = gm.group(1)
                if locus.endswith("-rc"):
                    continue
                gene_order[locus].append(gid.group(1))
                gene_locus[gid.group(1)] = locus
            elif kind in ("mRNA", "transcript"):
                tid, par = ID_RE.search(attrs), PARENT_RE.search(attrs)
                if tid and par and par.group(1) in gene_locus:
                    tx_gene[tid.group(1)] = par.group(1)
            elif kind == "CDS":
                par = PARENT_RE.search(attrs)
                if par and par.group(1) in tx_gene:
                    cds[par.group(1)].append((int(f[3]), int(f[4])))

    struct_of = {t: tuple(sorted(v)) for t, v in cds.items()}
    by_gene: dict = defaultdict(dict)
    for tid, gid in tx_gene.items():
        if tid in struct_of:
            by_gene[gid][tid] = struct_of[tid]

    kept, duplicate_records, duplicates_discordant = {}, 0, 0
    for locus, gids in gene_order.items():
        kept[locus] = by_gene.get(gids[0], {})
        first_set = set(kept[locus].values())
        for gid in gids[1:]:
            duplicate_records += 1
            if set(by_gene.get(gid, {}).values()) != first_set:
                duplicates_discordant += 1

    meta = {
        "gene_records_total": sum(len(v) for v in gene_order.values()),
        "distinct_GM_loci": len(gene_order),
        "gene_records_without_GM_tag": no_gm_genes,
        "records_per_locus_histogram": dict(sorted(
            {k: sum(1 for v in gene_order.values() if len(v) == k)
             for k in {len(v) for v in gene_order.values()}}.items())),
        "duplicate_records_discarded": duplicate_records,
        "duplicate_records_disagreeing_with_kept_record": duplicates_discordant,
    }
    return kept, meta


def read_reference(path: Path) -> tuple[dict, dict, dict]:
    """RefGen_V4 -> ({gene: {tx: CDS tuple}}, {gene: primary tx}, counts).

    Phytozome GFF3 carries no GM=; transcripts are attached through mRNA Parent=, and the
    primary is the first mRNA of the gene in file order (NOT the longest=1 flag, which for
    e.g. Zm00001d027231 names T007 while the prompt supplied is T001).
    """
    tx_gene: dict = {}
    primary: dict = {}
    cds: dict = defaultdict(list)
    n_genes = 0

    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            kind, attrs = f[2], f[8]
            if kind == "gene":
                n_genes += 1
            elif kind in ("mRNA", "transcript"):
                tid, par = ID_RE.search(attrs), PARENT_RE.search(attrs)
                if not (tid and par):
                    continue
                tx_gene[tid.group(1)] = par.group(1)
                primary.setdefault(par.group(1), tid.group(1))
            elif kind == "CDS":
                par = PARENT_RE.search(attrs)
                if par and par.group(1) in tx_gene:
                    cds[par.group(1)].append((int(f[3]), int(f[4])))

    ref: dict = defaultdict(dict)
    for tid, struct in cds.items():
        ref[tx_gene[tid]][tid] = tuple(sorted(struct))

    counts = {
        "reference_genes": n_genes,
        "reference_transcripts": len(tx_gene),
        "reference_transcripts_with_CDS": len(cds),
        "reference_genes_with_CDS": len(ref),
    }
    return dict(ref), primary, counts


def score(pred: dict, ref: dict, primary: dict, loci: list, label: str,
          refname: str = "RefGenV4") -> dict:
    added = struct_hit = chain_hit = 0
    alt_structures = alt_transcripts = alt_recovered = 0
    loci_with_addition = missing_primary = multi_iso_loci = 0
    # An addition proposed where the reference annotates no alternative at all cannot be
    # scored as a hit, so the two situations are counted apart: it separates "the model
    # invents structures" from "the reference has nothing to invent against".
    added_at_multi_iso = hit_at_multi_iso = 0

    for locus in loci:
        txs = pred[locus]
        ref_here = ref.get(locus, {})
        ref_primary = ref_here.get(primary.get(locus))
        if ref_primary is None:
            missing_primary += 1

        additions = list({s for s in txs.values() if s != ref_primary})
        alt_ref = {s for s in ref_here.values() if s != ref_primary}
        alt_transcripts += sum(1 for t, s in ref_here.items()
                               if t != primary.get(locus) and s != ref_primary)
        alt_structures += len(alt_ref)
        if alt_ref:
            multi_iso_loci += 1
            added_at_multi_iso += len(additions)
        if additions:
            loci_with_addition += 1
        added += len(additions)
        hits_here = sum(1 for s in additions if s in alt_ref)
        struct_hit += hits_here
        hit_at_multi_iso += hits_here
        chain_ref = {chain(x) for x in alt_ref}
        chain_hit += sum(1 for s in additions if len(chain(s)) >= 1 and chain(s) in chain_ref)
        alt_recovered += len(alt_ref & set(additions))

    return {
        "stratum": label,
        "loci_scored": len(loci),
        "loci_with_a_reference_alternative": multi_iso_loci,
        "loci_whose_supplied_primary_has_no_reference_CDS": missing_primary,
        "loci_with_at_least_one_addition": loci_with_addition,
        "added_transcripts": added,
        "reference_alternative_transcripts": alt_transcripts,
        f"reference_distinct_alternative_CDS_structures": alt_structures,
        f"added_matching_{refname}_alternative_exact_CDS": struct_hit,
        f"added_matching_{refname}_alternative_intron_chain": chain_hit,
        "precision_exact_CDS_pct": round(100 * struct_hit / added, 1) if added else None,
        "wilson95_ci_pct": wilson95(struct_hit, added),
        "precision_intron_chain_pct": round(100 * chain_hit / added, 1) if added else None,
        "recall_of_reference_alternatives_pct": (
            round(100 * alt_recovered / alt_structures, 2) if alt_structures else None),
        "alternative_structures_per_locus": round(alt_structures / len(loci), 3) if loci else None,
        "additions_at_loci_that_have_a_reference_alternative": added_at_multi_iso,
        "precision_restricted_to_those_loci_pct": (
            round(100 * hit_at_multi_iso / added_at_multi_iso, 1) if added_at_multi_iso else None),
    }


def prompt_identity_check(pred: dict, ref_order: dict, struct: dict, longest: dict) -> dict:
    """Confirm the prompt really was each gene's FIRST reference transcript.

    If the prompt were some other transcript, the true primary would be counted as an
    addition and would match the reference by construction, inflating precision. The test
    is whether the prediction retains the first transcript's exact CDS structure.
    """
    n = first_kept = longest_kept = neither = 0
    for locus, txs in pred.items():
        order = ref_order.get(locus, [])
        if not order:
            continue
        n += 1
        emitted = set(txs.values())
        if struct.get(order[0]) in emitted:
            first_kept += 1
        if struct.get(longest.get(locus)) in emitted:
            longest_kept += 1
        if not any(struct.get(t) in emitted for t in order):
            neither += 1
    return {
        "loci_tested": n,
        "retains_reference_FIRST_transcript_structure": first_kept,
        "retains_reference_FIRST_transcript_pct": round(100 * first_kept / n, 1) if n else None,
        "retains_reference_longest_flag_structure": longest_kept,
        "retains_reference_longest_flag_pct": round(100 * longest_kept / n, 1) if n else None,
        "retains_no_reference_structure": neither,
        "reading": ("the prompt is the first transcript in reference file order, not the "
                    "Phytozome longest=1 transcript; this matches the rule "
                    "14_make_altonly_references.py used for TAIR10"),
    }


def read_reference_ordered(path: Path) -> tuple[dict, dict, dict]:
    """Transcript order per gene, CDS structures, and the longest=1 transcript per gene."""
    tx_gene: dict = {}
    order: dict = defaultdict(list)
    longest: dict = {}
    cds: dict = defaultdict(list)
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            if f[2] in ("mRNA", "transcript"):
                t, p = ID_RE.search(f[8]), PARENT_RE.search(f[8])
                if not (t and p):
                    continue
                tx_gene[t.group(1)] = p.group(1)
                order[p.group(1)].append(t.group(1))
                if "longest=1" in f[8]:
                    longest[p.group(1)] = t.group(1)
            elif f[2] == "CDS":
                p = PARENT_RE.search(f[8])
                if p and p.group(1) in tx_gene:
                    cds[p.group(1)].append((int(f[3]), int(f[4])))
    return dict(order), {t: tuple(sorted(v)) for t, v in cds.items()}, longest


def verify_athaliana() -> dict:
    """Rerun this file's score() on the A. thaliana inputs; it must return 200/1,103."""
    pred: dict = defaultdict(dict)
    with AT_PRED.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            g, t = GM_RE.search(f[8]), PARENT_RE.search(f[8])
            if not (g and t) or g.group(1).endswith("-rc"):
                continue
            pred[g.group(1).replace(".TAIR10", "")].setdefault(t.group(1), []).append(
                (int(f[3]), int(f[4])))
    pred = {g: {t: tuple(sorted(v)) for t, v in tx.items()} for g, tx in pred.items()}

    ref: dict = defaultdict(dict)
    raw: dict = defaultdict(lambda: defaultdict(list))
    gid_re, tid_re = re.compile(r'gene_id "([^"]+)"'), re.compile(r'transcript_id "([^"]+)"')
    with (AT_DATA / "TAIR10.gtf").open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            g, t = gid_re.search(f[8]), tid_re.search(f[8])
            if not (g and t):
                continue
            raw[g.group(1).replace(".TAIR10", "")][t.group(1)].append((int(f[3]), int(f[4])))
    ref = {g: {t: tuple(sorted(v)) for t, v in tx.items()} for g, tx in raw.items()}

    primary = {}
    for line in (AT_DATA / "primary_transcript_ids.txt").read_text().splitlines():
        if line.strip():
            primary[line.strip().split(".")[0]] = line.strip()

    s = score(pred, ref, primary, sorted(pred), "A. thaliana, all evaluation loci", "TAIR10")
    s["reproduces_script_28"] = (s["added_transcripts"] == AT_ALL_ADDED
                                 and s["added_matching_TAIR10_alternative_exact_CDS"] == AT_ALL_HITS)
    s["expected_from_script_28"] = {"added_transcripts": AT_ALL_ADDED,
                                    "exact_CDS_hits": AT_ALL_HITS, "precision_pct": 18.1}
    return s


def two_proportion_z(h1: int, n1: int, h2: int, n2: int) -> dict:
    """Pooled two-proportion z test; the strata here are far too large for Fisher to matter."""
    if not (n1 and n2):
        return {}
    p1, p2 = h1 / n1, h2 / n2
    p = (h1 + h2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return {}
    z = (p1 - p2) / se
    pval = math.erfc(abs(z) / math.sqrt(2))
    return {"pct": [round(100 * p1, 1), round(100 * p2, 1)],
            "counts": [[h1, n1], [h2, n2]],
            "z": round(z, 3), "p_value": float(f"{pval:.3g}")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=OUTDIR / "zmays_additions_precision.json")
    ap.add_argument("--verify-athaliana", action="store_true",
                    help="rerun score() on the A. thaliana inputs and check it returns 200/1103")
    args = ap.parse_args()

    print(f"reading prediction: {PRED}", file=sys.stderr)
    pred, pred_meta = read_prediction(PRED)
    print(f"reading reference:  {REF}", file=sys.stderr)
    ref, primary, ref_counts = read_reference(REF)
    ref_order, ref_struct, ref_longest = read_reference_ordered(REF)
    prompt_check = prompt_identity_check(pred, ref_order, ref_struct, ref_longest)

    at_check = None
    if args.verify_athaliana:
        print("verifying against A. thaliana (script 28 reproduction)...", file=sys.stderr)
        at_check = verify_athaliana()
        print(f"  reproduces script 28: {at_check['reproduces_script_28']} "
              f"({at_check['added_matching_TAIR10_alternative_exact_CDS']}/"
              f"{at_check['added_transcripts']})", file=sys.stderr)

    loci_all = sorted(pred)
    grmzm = sorted(l for l in loci_all if l.startswith("GRMZM"))
    loci_ex = sorted(l for l in loci_all if not l.startswith("GRMZM"))
    unmatched = [l for l in loci_all if l not in ref]

    strata = {
        "ALL": score(pred, ref, primary, loci_all, "all predicted loci"),
        "EXCLUDING_GRMZM": score(pred, ref, primary, loci_ex,
                                 "legacy GRMZM loci removed (leakage-free)"),
        "GRMZM_ONLY": score(pred, ref, primary, grmzm,
                            "legacy GRMZM loci only (these DID enter training)"),
    }

    ex = strata["EXCLUDING_GRMZM"]
    out = {
        "generated": "2026-08-17",
        "question": ("Does TransGenic's completion-mode additions precision hold up on a species "
                     "that was excluded from training? Z. mays is that control: the 'Zm' prefix "
                     "filter kept all but 174 legacy GRMZM gene models out of the database."),
        "inputs": {
            "prediction": str(PRED),
            "reference": str(REF),
            "prediction_note": ("no Z_mays_transgenic400Mprompt_beam1.gff3 exists; unlike "
                                "A. thaliana (2 records/locus, deduplicated by script 13), the "
                                "maize export writes 3 records per locus, so this script keeps "
                                "the first gene record per GM itself"),
            "reference_note": ("RefGen_V4 is the TAIR10 analogue; maize has no long-read AtRTD3 "
                               "counterpart, so no second reference is scored"),
            "definitions": "revision/scripts/28_score_added_isoforms.py, unchanged",
        },
        "prediction_file_structure": pred_meta,
        "reference_file_structure": ref_counts,
        "prompt_identity_check": prompt_check,
        "athaliana_reproduction_check": at_check,
        "coverage": {
            "predicted_loci": len(loci_all),
            "predicted_loci_absent_from_RefGenV4": len(unmatched),
            "predicted_loci_legacy_GRMZM": len(grmzm),
            "legacy_GRMZM_gene_models_in_training": 174,
            "note": ("174 legacy GRMZM gene models (176 dataset rows) survived the Zm-prefix "
                     "filter; %d of them carry a prediction here and are excluded in the "
                     "sensitivity stratum" % len(grmzm)),
        },
        "strata": strata,
        "comparison_with_A_thaliana": {
            "source": ("revision/results/heldout_additions/"
                       "split_composition_and_precision_20260811.json"),
            "A_thaliana_all_27413_loci": {
                "added_transcripts": AT_ALL_ADDED,
                "exact_CDS_hits": AT_ALL_HITS,
                "precision_pct": round(100 * AT_ALL_HITS / AT_ALL_ADDED, 1),
                "wilson95_ci_pct": wilson95(AT_ALL_HITS, AT_ALL_ADDED),
                "leakage": "78.4% of loci seen in training",
            },
            "A_thaliana_validation_arm": {
                "added_transcripts": AT_VALIDATION_ADDED,
                "exact_CDS_hits": AT_VALIDATION_HITS,
                "precision_pct": round(100 * AT_VALIDATION_HITS / AT_VALIDATION_ADDED, 1),
                "wilson95_ci_pct": wilson95(AT_VALIDATION_HITS, AT_VALIDATION_ADDED),
                "note": "quoted, not recomputed",
            },
            "Z_mays_excluding_GRMZM": {
                "added_transcripts": ex["added_transcripts"],
                "exact_CDS_hits": ex["added_matching_RefGenV4_alternative_exact_CDS"],
                "precision_pct": ex["precision_exact_CDS_pct"],
                "wilson95_ci_pct": ex["wilson95_ci_pct"],
                "leakage": "none (species excluded from the training database)",
            },
            "test_Zmays_exGRMZM_vs_A_thaliana_all": two_proportion_z(
                ex["added_matching_RefGenV4_alternative_exact_CDS"], ex["added_transcripts"],
                AT_ALL_HITS, AT_ALL_ADDED),
            "test_Zmays_exGRMZM_vs_A_thaliana_validation": two_proportion_z(
                ex["added_matching_RefGenV4_alternative_exact_CDS"], ex["added_transcripts"],
                AT_VALIDATION_HITS, AT_VALIDATION_ADDED),
            "alternative_structure_density": {
                "A_thaliana_TAIR10_alt_structures_per_locus": round(AT_ALT_STRUCTURES / AT_LOCI, 3),
                "Z_mays_RefGenV4_alt_structures_per_locus": ex["alternative_structures_per_locus"],
                "why_it_matters": ("recall denominators are not comparable across the two "
                                   "species; precision numerators are"),
            },
        },
    }

    for name, s in strata.items():
        print(f"\n[{name}]", file=sys.stderr)
        for k, v in s.items():
            print(f"  {k:<56} {v}", file=sys.stderr)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1))
    print(f"\nwritten: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
