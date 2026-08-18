#!/usr/bin/env python3
"""Score the isoforms TransGenic adds in SELF-PROMPTED mode, prompt transcript removed.

Reviewer 2's question — "what percentage of alternatively spliced isoforms are predicted
correctly, with the given isoform removed or ignored" — has been answered twice:

    TransGenic, reference-prompted     200 / 1,103   = 18.1%   (28_score_added_isoforms.py)
    AUGUSTUS posterior sampling        574 / 43,433  =  1.3%   (28 --augustus)

Those two are not the same experiment. TransGenic was handed a curated TAIR10 transcript as
its prompt; AUGUSTUS was handed nothing and its own first prediction stood in for the prompt.
Every reader who compares 18.1% against 1.3% is comparing a reference-prompted method against
a fully ab initio one, and the reference-prompted side gets the locus boundaries, the reading
frame and one true isoform for free.

Self-prompted mode closes that gap. It is the same model with no external annotation: the
model predicts de novo, its own first transcript becomes the prompt, and it completes from
there (Methods L122). Scored the way AUGUSTUS is scored — own first transcript removed — it
is the symmetric ab initio comparator that the 18.1% vs 1.3% contrast is missing.

WHAT IS SCORED

Identical definitions to `28_score_added_isoforms.py`, reproduced rather than reimplemented:

    addition        a distinct CDS structure at a locus that differs from the structure
                    supplied for that locus, identical emissions collapsed to one
    exact CDS       equality of the full sorted CDS coordinate tuple
    intron chain    equality of the CDS intron chain, which ignores the first and last CDS
                    boundaries; guarded on a non-empty chain so single-exon structures
                    cannot match for free
    the reference   TAIR10's alternative transcripts (the curated primary removed) and,
                    scored separately with nothing removed, any AtRTD3 transcript

TWO THINGS THE SELF-PROMPTED OUTPUT FORCES, each exposed as a flag rather than hard-coded,
so the SAME code path can re-derive the published reference-prompted and AUGUSTUS numbers:

1. `--prompt-source`. `28` reads the supplied structure for the reference-prompted run out
   of TAIR10 (`tair10-primary`), because that prediction really was prompted with TAIR10's
   primary transcript. Self-prompted mode has no external prompt: the model prompts itself
   with its own de novo first transcript, so the supplied structure is the locus's FIRST
   transcript in file order (`first-transcript`) — exactly the substitution `28 --augustus`
   already makes for AUGUSTUS, and the reason the two are comparable.

2. `--locus-key`. `28` groups the reference-prompted output by its `GM=` tag and looks the
   reference up by that same string, because there `GM=AT1G01010.TAIR10` IS a TAIR10 gene id
   (`gm`). Self-prompted `GM=` values are `A_thaliana_g000001` — the model's own de novo
   locus ids, which carry no reference identity at all. Those loci are therefore paired to
   TAIR10 by reciprocal (Jaccard) overlap, resolved one-to-one (`overlap`), reusing
   `32_score_polishing.py`'s matcher rather than a second implementation of it.

   `overlap` is the weaker instrument of the two, and it is weaker in a direction that
   matters: a locus that matches no TAIR10 gene stays in the denominator and can never
   contribute a hit. Running the reference-prompted file through BOTH keys sizes that cost
   directly instead of leaving it as a caveat — see `--locus-key overlap` on
   `A_thaliana_transgenic400Mprompt_beam1.gff3`.

DUPLICATE EXPORT

`A_thaliana_transgenic400M_prompt_denovo.gff3` writes every locus THREE times: 82,194 gene
rows over 27,398 `GM=` values, 93,690 mRNAs. The three records per locus are verified
structurally identical here (`duplicate_records_all_identical`), and only the first is kept.
This is checked rather than assumed, and the check is not cosmetic — if the copies ever
differed, keeping the first would silently discard emissions. It is also, on this file,
numerically inert: additions are counted as a SET of distinct structures per locus, so a
tripled record contributes no new structure either way. `--no-dedup` scores without it so
that inertness is demonstrated rather than claimed.

Usage:
    # the self-prompted measurement
    python 51_score_selfprompted_additions.py \\
        --pred .../A_thaliana_transgenic400M_prompt_denovo.gff3 \\
        --locus-key overlap --prompt-source first-transcript --tool transgenic400M_selfprompted

    # the reproduction check: must print 1103 / 200 / 18.1
    python 51_score_selfprompted_additions.py \\
        --pred .../A_thaliana_transgenic400Mprompt_beam1.gff3 \\
        --locus-key gm --prompt-source tair10-primary
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
CMP = ROOT / "transgenic_comparison"
DATA = ROOT / "transgenic" / "revision" / "data"
TAIR10_GTF = DATA / "TAIR10" / "TAIR10.gtf"
ATRTD3_GTF = DATA / "AtRTD3" / "atRTD3_TS_21Feb22_transfix.gtf"
PRIMARY_IDS = DATA / "TAIR10" / "primary_transcript_ids.txt"

TAIR10_GENE_SUFFIX = ".TAIR10"


def _load_sibling(filename: str, name: str) -> ModuleType:
    """Import a numbered sibling script by path — `import 32_...` is not valid syntax."""
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_POLISH = _load_sibling("32_score_polishing.py", "_polishing_scorer_51")
match_by_overlap = _POLISH.match_by_overlap
_resolve_one_to_one = _POLISH._resolve_one_to_one


def chain(struct: tuple) -> tuple:
    """The CDS intron chain: the gaps between consecutive CDS segments.

    Character-for-character `28_score_added_isoforms.py.chain`. A single-segment structure
    has an empty chain, which is why every caller guards on `len(chain(s)) >= 1`.
    """
    return tuple((struct[i][1], struct[i + 1][0]) for i in range(len(struct) - 1))


def _agi(gene_id: str) -> str:
    """`AT1G01010.TAIR10` -> `AT1G01010`; anything else unchanged."""
    if gene_id.endswith(TAIR10_GENE_SUFFIX):
        return gene_id[: -len(TAIR10_GENE_SUFFIX)]
    return gene_id


def gtf_cds_by_gene(path: Path) -> tuple[dict, dict, dict]:
    """({gene: {tx: struct}}, {gene: seqid}, {gene: strand}) over a GTF's CDS rows.

    The first return value is `28_score_added_isoforms.py.cds_by_transcript(path, gff3=False)`
    exactly — same regexes, same `.TAIR10` stripping, same `-rc` skip, same sorted tuple —
    so the reference side of this module and of `28` cannot drift. Transcript insertion
    order follows file order. Seqid and strand ride along for the positional matcher and the
    antisense diagnostic; `28` needs neither because it never matches by position.
    """
    segs: dict = defaultdict(lambda: defaultdict(list))
    seqid: dict = {}
    strands: dict = defaultdict(set)
    gene_re = re.compile(r'gene_id "([^"]+)"')
    tx_re = re.compile(r'transcript_id "([^"]+)"')
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            g, t = gene_re.search(f[8]), tx_re.search(f[8])
            if not (g and t):
                continue
            locus = g.group(1)
            if locus.endswith("-rc"):
                continue
            locus = locus.replace(TAIR10_GENE_SUFFIX, "")
            segs[locus][t.group(1)].append((int(f[3]), int(f[4])))
            seqid[locus] = f[0]
            strands[locus].add(f[6])
    structs = {g: {t: tuple(sorted(v)) for t, v in tx.items()} for g, tx in segs.items()}
    strand = {g: (next(iter(s)) if len(s) == 1 else "?") for g, s in strands.items()}
    return structs, seqid, strand


def gff3_prediction(path: Path) -> tuple[dict, dict, dict, dict]:
    """Parse a prediction GFF3 into per-locus, per-gene-record transcript structures.

    Returns ({locus: {record: {tx: struct}}}, {locus: seqid}, {locus: strand},
             {locus: [record, …] in file order}), where `locus` is the `GM=` value with any
    `.TAIR10` suffix stripped and `record` is the gene row's `ID=`.

    Keeping the gene RECORD as a level of its own is what makes the duplicate export
    measurable: `28` reads `GM=` off CDS rows and flattens straight to transcripts, which is
    correct for its inputs but cannot see that the self-prompted file writes each locus three
    times. `GM=` is read off the mRNA row rather than the CDS row so that AUGUSTUS — which
    tags gene and mRNA rows but not CDS rows — parses through the same code path.
    """
    tx_segs: dict = defaultdict(list)
    tx_record: dict = {}
    tx_locus: dict = {}
    tx_seq: dict = {}
    tx_strand: dict = {}
    order: list = []
    record_order: dict = defaultdict(list)
    record_locus: dict = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            if f[2] == "gene":
                gid = _attr(f[8], "ID")
                gm = _attr(f[8], "GM")
                if gid is not None:
                    record_locus[gid] = _agi(gm) if gm else gid
            elif f[2] == "mRNA":
                tid = _attr(f[8], "ID")
                parent = _attr(f[8], "Parent")
                gm = _attr(f[8], "GM")
                if tid is None:
                    continue
                record = parent if parent is not None else tid
                locus = _agi(gm) if gm else record_locus.get(record, record)
                tx_record[tid] = record
                tx_locus[tid] = locus
                order.append(tid)
                if record not in record_order[locus]:
                    record_order[locus].append(record)
            elif f[2] == "CDS":
                parent = _attr(f[8], "Parent")
                if parent is None:
                    continue
                tx_segs[parent].append((int(f[3]), int(f[4])))
                tx_seq[parent] = f[0]
                tx_strand[parent] = f[6]

    loci: dict = defaultdict(lambda: defaultdict(dict))
    seqid: dict = {}
    strands: dict = defaultdict(set)
    for tid in order:
        if tid not in tx_segs:
            continue  # an mRNA with no CDS rows is not a coding structure
        locus = tx_locus[tid]
        loci[locus][tx_record[tid]][tid] = tuple(sorted(tx_segs[tid]))
        seqid[locus] = tx_seq[tid]
        strands[locus].add(tx_strand[tid])
    strand = {g: (next(iter(s)) if len(s) == 1 else "?") for g, s in strands.items()}
    return (loci, seqid, strand,
            {g: [r for r in record_order[g] if r in loci[g]] for g in loci})


def _attr(attributes: str, key: str) -> str | None:
    for field in attributes.rstrip(";").split(";"):
        k, _, v = field.strip().partition("=")
        if k == key:
            return v
    return None


def _augustus_order(txs: dict) -> dict:
    """Transcripts re-ordered by AUGUSTUS's `.tN` suffix, `28 --augustus`'s convention.

    AUGUSTUS's posterior samples are named `augSmp_AT1G01010.t1 … .tN` and `28` sorts on N
    to pick the stand-in prompt rather than trusting file order. Applied unconditionally
    here: on every other input the `.tN` suffix already agrees with file order, so this is a
    no-op there and the AUGUSTUS reproduction needs no separate branch.
    """
    def key(t: str) -> int:
        m = re.search(r"\.t(\d+)$", t)
        return int(m.group(1)) if m else 0
    return {t: txs[t] for t in sorted(txs, key=key)}


def _dedup_records(loci: dict, records: dict) -> tuple[dict, dict]:
    """Collapse a duplicated export to its first gene record per locus.

    Returns ({locus: {tx: struct}}, report). The report states the record-count
    distribution and whether every record of a locus was structurally identical — the
    condition under which dropping all but the first is lossless. When it does NOT hold the
    records are kept and the fact is reported, because silently keeping the first would
    discard real emissions.
    """
    multiplicity = Counter(len(records[g]) for g in loci)
    identical = True
    for locus, recs in records.items():
        if len(recs) < 2:
            continue
        blocks = {tuple(loci[locus][r].values()) for r in recs}
        if len(blocks) != 1:
            identical = False
            break
    flat: dict = {}
    for locus, recs in records.items():
        if identical and recs:
            flat[locus] = dict(loci[locus][recs[0]])
        else:
            merged: dict = {}
            for r in recs:
                merged.update(loci[locus][r])
            flat[locus] = merged
    return flat, {
        "gene_records_per_locus": {str(k): v for k, v in sorted(multiplicity.items())},
        "duplicate_records_all_identical": identical,
        "kept": "first gene record per locus" if identical else "all records merged",
    }


def _flatten(loci: dict, records: dict) -> dict:
    """Every transcript of every record, no deduplication — the `--no-dedup` control."""
    flat: dict = {}
    for locus, recs in records.items():
        merged: dict = {}
        for r in recs:
            merged.update(loci[locus][r])
        flat[locus] = merged
    return flat


def _span(txs: dict, seq: str) -> tuple:
    starts = [s for segs in txs.values() for s, _e in segs]
    ends = [e for segs in txs.values() for _s, e in segs]
    return (seq, min(starts), max(ends))


def map_loci_to_reference(pred: dict, pred_seq: dict, ref: dict, ref_seq: dict,
                          locus_key: str) -> tuple[dict, dict]:
    """{prediction locus: reference AGI} plus a diagnostics dict.

    `gm` is identity — the prediction's `GM=` already IS the reference gene id, which is
    `28`'s assumption and is true only of the reference-prompted and AUGUSTUS files. A locus
    whose id is absent from the reference maps to None and is scored against an empty
    reference, exactly as `28`'s `ref.get(locus, {})` does.

    `overlap` pairs by reciprocal overlap and resolves one-to-one, so a de novo locus split
    across two TAIR10 genes contributes to one of them, not both.
    """
    if locus_key == "gm":
        mapped = {locus: (locus if locus in ref else None) for locus in pred}
        return mapped, {
            "locus_key": "gm",
            "loci_matched_to_reference": sum(1 for v in mapped.values() if v is not None),
            "loci_without_reference_match": sum(1 for v in mapped.values() if v is None),
            "split_predictions": None,
        }

    pred_spans = {(*_span(txs, pred_seq[locus]), locus): txs
                  for locus, txs in pred.items() if txs}
    ref_spans = {(*_span(txs, ref_seq[gene]), gene): txs for gene, txs in ref.items() if txs}
    raw = match_by_overlap(pred_spans, ref_spans)
    if not raw:
        raise RuntimeError(
            "no predicted locus overlaps any reference locus on the same sequence — this is "
            "the sequence-name-mismatch failure mode, check seqids match exactly "
            f"(prediction seqs={sorted({s[0] for s in pred_spans})}, "
            f"reference seqs={sorted({s[0] for s in ref_spans})})")
    resolved, split = _resolve_one_to_one(raw)
    mapped = {locus: None for locus in pred}
    for p_span, r_span in resolved.items():
        mapped[p_span[3]] = r_span[3]
    return mapped, {
        "locus_key": "overlap",
        "loci_matched_to_reference": sum(1 for v in mapped.values() if v is not None),
        "loci_without_reference_match": sum(1 for v in mapped.values() if v is None),
        "split_predictions": split,
        "loci_overlapping_before_one_to_one_resolution": len(raw),
    }


def score(pred_gff: Path, locus_key: str, prompt_source: str, tair10_gtf: Path = TAIR10_GTF,
          atrtd3_gtf: Path = ATRTD3_GTF, primary_ids_path: Path = PRIMARY_IDS,
          tool: str | None = None, dedup: bool = True, cli: dict | None = None) -> dict:
    loci, pred_seq, pred_strand, records = gff3_prediction(pred_gff)
    if not loci:
        raise ValueError(f"no CDS-bearing transcripts found in {pred_gff} — refusing to "
                         "score, this would silently report all zeros")
    pred, dedup_report = (_dedup_records(loci, records) if dedup
                          else (_flatten(loci, records),
                                {"gene_records_per_locus": {
                                     str(k): v for k, v in
                                     sorted(Counter(len(records[g]) for g in loci).items())},
                                 "duplicate_records_all_identical": None,
                                 "kept": "all records merged (--no-dedup)"}))
    pred = {locus: _augustus_order(txs) for locus, txs in pred.items()}

    ref, ref_seq, ref_strand = gtf_cds_by_gene(tair10_gtf)
    art, _art_seq, _art_strand = gtf_cds_by_gene(atrtd3_gtf)
    for name, d, p in (("TAIR10 reference", ref, tair10_gtf),
                       ("AtRTD3 reference", art, atrtd3_gtf)):
        if not d:
            raise ValueError(f"no CDS-bearing transcripts found in {name} file {p}")

    primary: dict = {}
    for line in primary_ids_path.read_text().splitlines():
        if line.strip():
            primary[line.strip().split(".")[0]] = line.strip()

    mapped, mapping_diag = map_loci_to_reference(pred, pred_seq, ref, ref_seq, locus_key)

    t: Counter = Counter()
    for locus, txs in pred.items():
        gene = mapped.get(locus)
        ref_txs = ref.get(gene, {}) if gene is not None else {}
        ref_primary = ref_txs.get(primary.get(gene)) if gene is not None else None

        if prompt_source == "tair10-primary":
            supplied = ref_primary
        else:
            supplied = next(iter(txs.values())) if txs else None

        # A set: two beams, a tripled export or two identical emissions are one structure.
        additions = {s for s in txs.values() if s != supplied}
        # The reference side always drops TAIR10's own primary, whichever prompt source is
        # in play: recovering the primary is not recovering an alternative isoform.
        alt_ref = {s for s in ref_txs.values() if s != ref_primary}
        art_here = set(art.get(gene, {}).values()) if gene is not None else set()
        alt_chains = {chain(x) for x in alt_ref}

        t["loci_scored"] += 1
        t["reference_alternative_transcripts"] += len(alt_ref)
        t["added_transcripts"] += len(additions)
        if additions:
            t["loci_with_at_least_one_addition"] += 1
        if supplied is not None and supplied in txs.values():
            t["loci_where_prompt_survived"] += 1
        if gene is None:
            t["additions_at_loci_without_reference_match"] += len(additions)
        if ref_primary is not None and ref_primary in additions:
            t["added_matching_TAIR10_primary"] += 1

        exact = additions & alt_ref
        t["added_matching_TAIR10_alternative_exact_CDS"] += len(exact)
        t["added_matching_any_AtRTD3_transcript"] += len(additions & art_here)
        t["alt_recovered"] += len(alt_ref & additions)
        if exact and gene is not None and pred_strand.get(locus, "?") != ref_strand.get(gene, "?"):
            t["added_matching_TAIR10_alternative_exact_CDS_opposite_strand"] += len(exact)

        prompt_chain = chain(supplied) if supplied is not None else None
        for structure in additions:
            c = chain(structure)
            if len(c) >= 1 and c in alt_chains:
                t["added_matching_TAIR10_alternative_intron_chain"] += 1
                if c == prompt_chain:
                    t["intron_chain_reusing_prompt_chain"] += 1
                else:
                    t["intron_chain_distinct_from_prompt"] += 1

    added = t["added_transcripts"]
    alt_total = t["reference_alternative_transcripts"]

    def pct(n: int, d: int) -> float | None:
        """A percentage, or None when the denominator is 0 — never 0.0, which would read as
        a measured result rather than an absent one."""
        return round(100 * n / d, 1) if d else None

    return {
        "provenance": {
            "prediction": str(pred_gff),
            "reference_TAIR10": str(tair10_gtf),
            "reference_AtRTD3": str(atrtd3_gtf),
            "primary_ids": str(primary_ids_path),
            "tool": tool,
            "locus_key": locus_key,
            "prompt_source": prompt_source,
            "argv": list(cli["argv"]) if cli else None,
        },
        "definitions": {
            "addition": ("a distinct CDS structure at a locus that differs from the "
                         "structure supplied for that locus, identical emissions collapsed"),
            "supplied_structure": ("TAIR10's curated primary transcript at the matched locus"
                                   if prompt_source == "tair10-primary" else
                                   "the locus's own first transcript, the self-prompt"),
            "TAIR10_alternative": ("any TAIR10 transcript at the matched locus other than "
                                   "the curated primary from primary_transcript_ids.txt"),
            "AtRTD3_match": "equality with ANY AtRTD3 transcript at the matched locus",
        },
        "loci_scored": t["loci_scored"],
        "loci_with_at_least_one_addition": t["loci_with_at_least_one_addition"],
        "added_transcripts": added,
        "reference_alternative_transcripts": alt_total,
        "added_matching_TAIR10_alternative_exact_CDS":
            t["added_matching_TAIR10_alternative_exact_CDS"],
        "added_matching_TAIR10_alternative_intron_chain":
            t["added_matching_TAIR10_alternative_intron_chain"],
        "added_matching_any_AtRTD3_transcript": t["added_matching_any_AtRTD3_transcript"],
        "precision_vs_TAIR10_alternatives_pct":
            pct(t["added_matching_TAIR10_alternative_exact_CDS"], added),
        "precision_vs_AtRTD3_pct": pct(t["added_matching_any_AtRTD3_transcript"], added),
        "recall_of_TAIR10_alternatives_pct": pct(t["alt_recovered"], alt_total),
        "decomposition": {
            "precision_vs_TAIR10_alternatives_intron_chain_pct":
                pct(t["added_matching_TAIR10_alternative_intron_chain"], added),
            "added_matching_TAIR10_alternative_intron_chain_reusing_prompt_chain":
                t["intron_chain_reusing_prompt_chain"],
            "added_matching_TAIR10_alternative_intron_chain_distinct_from_prompt":
                t["intron_chain_distinct_from_prompt"],
            # Reported beside the headline, never inside it. An addition equal to TAIR10's
            # primary is not an alternative isoform; it is the model proposing the
            # reference's main transcript where its own prompt was something else.
            "added_matching_TAIR10_primary": t["added_matching_TAIR10_primary"],
            "loci_where_prompt_survived": t["loci_where_prompt_survived"],
            "additions_at_loci_without_reference_match":
                t["additions_at_loci_without_reference_match"],
            "precision_vs_TAIR10_alternatives_at_reference_matched_loci_pct":
                pct(t["added_matching_TAIR10_alternative_exact_CDS"],
                    added - t["additions_at_loci_without_reference_match"]),
        },
        "diagnostics": {
            "prediction_loci": len(loci),
            "prediction_gene_records": sum(len(r) for r in records.values()),
            "prediction_transcripts_before_dedup": sum(
                len(txs) for recs in loci.values() for txs in recs.values()),
            "prediction_transcripts_after_dedup": sum(len(txs) for txs in pred.values()),
            "duplicate_export": dedup_report,
            "mapping": mapping_diag,
            "reference_loci": len(ref),
            "added_matching_TAIR10_alternative_exact_CDS_opposite_strand":
                t["added_matching_TAIR10_alternative_exact_CDS_opposite_strand"],
        },
    }


def _print_result(res: dict, indent: str = "  ") -> None:
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{indent}{k}:", file=sys.stderr)
            _print_result(v, indent + "  ")
        else:
            print(f"{indent}{k:<62} {v}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score the isoforms a prediction adds beyond the structure supplied for "
                    "each locus, with self-prompted mode as the intended subject.")
    ap.add_argument("--pred", type=Path, required=True, help="the prediction GFF3")
    ap.add_argument("--locus-key", choices=("gm", "overlap"), default="overlap",
                    help="'gm' when GM= is already a reference gene id (reference-prompted, "
                         "AUGUSTUS); 'overlap' for self-prompted, whose GM= is its own")
    ap.add_argument("--prompt-source", choices=("first-transcript", "tair10-primary"),
                    default="first-transcript",
                    help="what is removed as the prompt at each locus")
    ap.add_argument("--tair10", type=Path, default=TAIR10_GTF)
    ap.add_argument("--atrtd3", type=Path, default=ATRTD3_GTF)
    ap.add_argument("--primary-ids", type=Path, default=PRIMARY_IDS)
    ap.add_argument("--tool", default=None, help="tool label recorded in the provenance")
    ap.add_argument("--no-dedup", action="store_true",
                    help="score without collapsing a duplicated export, as a control")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    for label, path in (("--pred", args.pred), ("--tair10", args.tair10),
                        ("--atrtd3", args.atrtd3), ("--primary-ids", args.primary_ids)):
        if not path.exists():
            print(f"MISSING {label}: {path}", file=sys.stderr)
            return 1

    res = score(args.pred, args.locus_key, args.prompt_source, args.tair10, args.atrtd3,
                args.primary_ids, args.tool, not args.no_dedup,
                {"argv": list(argv) if argv is not None else sys.argv[1:]})
    _print_result(res)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=1))
        print(f"written: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
