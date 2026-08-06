#!/usr/bin/env python3
"""Score the alternatively spliced isoforms TransGenic adds to an EXTERNAL annotation.

The Discussion claims completion mode is designed to add isoforms to annotations "such as
those produced by Helixer or ANNEVO", and the response to reviewers flagged that this
combination had never been benchmarked. This is the benchmark. It answers one question:
prompted with a single-transcript annotation from another tool, how many of the isoforms
TransGenic adds are correct?

This is `28_score_added_isoforms.py` generalised from a TAIR10-prompted prediction to an
arbitrary external annotation, and its definitions are reproduced exactly so the numbers
slot into the same table:

    added_structures      distinct CDS structures in the output that differ from the
                          structure supplied for that locus, identical emissions collapsed
    exact CDS             equality of the full sorted CDS coordinate tuple
    intron chain          equality of the CDS intron chain, which ignores the first and
                          last CDS boundaries — the stricter-to-satisfy analogue of
                          GFFCompare's '=' class code
    the reference         TAIR10's alternative transcripts (primary removed) and, scored
                          separately with no primary removed, any AtRTD3 transcript

Three things differ from `28_score_added_isoforms.py`, each forced by the input:

1. `28` reads the supplied structure out of TAIR10 itself, because the prediction it scores
   was prompted with TAIR10 and its `GM=` values ARE TAIR10 gene ids. Here the prompt comes
   from the tool, so the supplied structure is read from `--input` — the FIRST mRNA of that
   gene in file order, which is what `preprocess.py` hands the decoder (see
   `examples/prompt_mode.py`'s `get_single_decoder_input_ids`, "elements of the first
   transcript"). This is the same convention Task 1 of the polishing benchmark ruled on and
   the same one `28 --augustus` already used.

2. Loci are paired across files by provenance and position, never by gene id. Every staged
   input and every completion-mode output has been through
   `transgenic_comparison/standardize_gff.py`, which renames genes by sort position, so
   `A_thaliana_g000001` in a Helixer input and `A_thaliana_g000001` in its own completion
   are NOT guaranteed to be the same locus — any locus dropped upstream shifts the
   numbering. Input to output pairs on the model's `GM=` tag, input to reference pairs on
   reciprocal (Jaccard) overlap resolved one-to-one, both via `32_score_polishing.py`'s
   matcher.

3. `GM=` is matched against the input gene row's FIRST attribute value, whatever its key,
   not against `ID=`. That is how the pinned `preprocess.py:314` derives it
   (`attributes.split(';')[0].split('=')[1]`), replicated in
   `33_run_polishing_benchmark.py._first_attribute_value` and reused here. Matching it
   against `ID=` is not a hypothetical error: it is why `32_score_polishing.py` reports
   `gm_paired: 0` for GeMoMa, whose gene rows lead with `Name=`.

WHAT THE HEADLINE NUMBER IS NOT, and why the decomposition below it exists

`added_structures` is a count of structures the model emitted beyond the one it was given.
It is not, by itself, a count of isoforms it added, and the polishing run showed exactly how
that goes wrong: 84-89% of it sat at loci where the prompted structure had been destroyed,
so the same event was booked as damage there and as a gain here. Every headline count is
therefore also reported split by whether the prompt survived at that locus
(`prompt_survived_loci` / `prompt_destroyed_loci`), and the precision restricted to the
surviving loci is the one that answers the question in the title.

Four more quantities exist for the same reason — each is a way this number could be wrong
in the direction that flatters it, and each is reported rather than assumed away:

    added_structures_vs_all_input_transcripts
        `32_score_polishing.py`'s definition: differs from EVERY transcript the input
        already held, not just the prompt. The two coincide exactly when the input is
        single-transcript per locus (Helixer and ANNEVO are), and diverge otherwise.

        This module's headline differs from `32`'s `added_structures` on TWO independent
        axes, so the field that reconciles them is
        `added_structures_vs_all_input_transcripts_at_reference_matched_loci`: `32` counts
        only at loci matched one-to-one to the reference, and only structures absent from
        the whole input locus. Measured on the 2026-08-06 polishing predictions, that field
        reproduces `32` exactly — GeMoMa 21,739, BRAKER3 128, EGAPx 22,711 — while the
        headline here is larger because it also counts additions at loci with no TAIR10
        gene and re-emissions of the input's non-prompted isoforms. Neither number is
        "the" answer; quoting either without the other is how the two analyses would
        appear to disagree when they do not.

    intron chain, reusing the prompt's own chain vs distinct from it
        Intron-chain precision ran 48x higher than exact-CDS precision on the polishing run,
        and almost all of that gap was terminus variants of the supplied transcript — not
        new splice structures. A chain hit whose chain equals the prompt's is counted, and
        counted separately.

    added_structures_at_reference_matched_loci
        Additions at input loci with no overlapping TAIR10 gene cannot match anything, and
        stay in the headline denominator (28's convention). The restricted denominator is
        reported so the resulting precision floor is not mistaken for a measurement.

    exact-CDS matches on the opposite strand
        Structures are compared as coordinate tuples, so an antisense overlapping gene can
        produce a coordinate-identical match that means nothing. Counted, not silently
        credited.

A quantity that cannot be computed is reported as null with the count that made it
uncomputable beside it (`reference_loci_without_AtRTD3_counterpart`,
`reference_alternative_transcripts_at_loci_without_output`), never as 0.

Usage:
    python 34_score_as_additions.py --input <tool.gff3> --output <completed.gff3> \\
        --tool <name> [--tair10 ...] [--atrtd3 ...] [--primary-ids ...] [--json out.json]
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
DATA = ROOT / "transgenic" / "revision" / "data"
TAIR10_GTF = DATA / "TAIR10" / "TAIR10.gtf"
ATRTD3_GTF = DATA / "AtRTD3" / "atRTD3_TS_21Feb22_transfix.gtf"
PRIMARY_IDS = DATA / "TAIR10" / "primary_transcript_ids.txt"

# TAIR10.gtf writes gene_id "AT1G01010.TAIR10" and transcript_id "AT1G01010.1.TAIR10";
# AtRTD3 writes the bare AGI locus id. Joining the two needs the suffix off.
TAIR10_GENE_SUFFIX = ".TAIR10"

_GTF_GENE = re.compile(r'gene_id "([^"]+)"')
_GTF_TX = re.compile(r'transcript_id "([^"]+)"')


def _load_sibling(filename: str, name: str) -> ModuleType:
    """Import a numbered sibling script by path — `import 32_...` is not valid syntax."""
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_POLISH = _load_sibling("32_score_polishing.py", "_polishing_scorer")
_RUNNER = _load_sibling("33_run_polishing_benchmark.py", "_polishing_runner")

cds_structures = _POLISH.cds_structures
match_by_overlap = _POLISH.match_by_overlap
_resolve_one_to_one = _POLISH._resolve_one_to_one
_representative = _POLISH._representative
_primary_representative = _POLISH._primary_representative
_load_primary_ids = _POLISH._load_primary_ids
_group_by_gene = _POLISH._group_by_gene
_parse_hierarchy = _POLISH._parse_hierarchy
_tx_and_gene = _POLISH._tx_and_gene
_attr = _POLISH._attr
_first_attribute_value = _RUNNER._first_attribute_value


def chain(struct: tuple) -> tuple:
    """The CDS intron chain: the gaps between consecutive CDS segments.

    Identical to `28_score_added_isoforms.py.chain`. A single-segment structure has an
    empty chain, which is why every caller guards on `len(chain(s)) >= 1` before treating
    a chain hit as a match — otherwise every single-exon structure would match every
    single-exon reference transcript for free.
    """
    return tuple((struct[i][1], struct[i + 1][0]) for i in range(len(struct) - 1))


def _agi(gene_id: str) -> str:
    """`AT1G01010.TAIR10` -> `AT1G01010`; anything else unchanged."""
    if gene_id.endswith(TAIR10_GENE_SUFFIX):
        return gene_id[: -len(TAIR10_GENE_SUFFIX)]
    return gene_id


def gtf_cds_structures(path: Path) -> tuple[dict, dict]:
    """({(seq, start, end, gene): {tx: struct}}, {gene: strand}) over a GTF's CDS rows.

    Same shape as `32_score_polishing.py.cds_structures` so the same matcher works on
    both; TAIR10.gtf and AtRTD3 are GTF, the tool inputs and model outputs are GFF3.
    Transcript insertion order follows file order, which is what
    `_primary_representative`'s fallback relies on.
    """
    tx_segs: dict[str, list] = defaultdict(list)
    tx_gene: dict[str, str] = {}
    tx_seq: dict[str, str] = {}
    tx_strand: dict[str, str] = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            gene, tx = _GTF_GENE.search(f[8]), _GTF_TX.search(f[8])
            if not (gene and tx):
                continue
            tx_segs[tx.group(1)].append((int(f[3]), int(f[4])))
            tx_gene[tx.group(1)] = gene.group(1)
            tx_seq[tx.group(1)] = f[0]
            tx_strand[tx.group(1)] = f[6]
    structs = {tx: tuple(sorted(segs)) for tx, segs in tx_segs.items()}
    strands: dict[str, set] = defaultdict(set)
    for tx, gene in tx_gene.items():
        strands[gene].add(tx_strand[tx])
    return (_group_by_gene(structs, tx_gene, tx_seq),
            {g: (next(iter(s)) if len(s) == 1 else "?") for g, s in strands.items()})


def gff3_gene_strands(path: Path) -> dict[str, str]:
    """{gene_id: strand} from a GFF3's CDS rows; "?" when a gene's rows disagree.

    Resolved through the same (transcript, gene) hierarchy `cds_structures` walks, so a
    gene here is the same gene there. Reads from `_parse_hierarchy`'s cached lines rather
    than re-reading a 100+ MB file.
    """
    id_type, id_parent, id_gbkey, _gene_gm, lines = _parse_hierarchy(path)
    strands: dict[str, set] = defaultdict(set)
    for line in lines:
        f = line.split("\t")
        if len(f) < 9 or f[2] != "CDS":
            continue
        status, _tx, gene, _fallback = _tx_and_gene(
            _attr(f[8], "Parent"), _attr(f[8], "ID"), id_type, id_parent, id_gbkey)
        if status != "ok":
            continue
        strands[gene].add(f[6])
    return {g: (next(iter(s)) if len(s) == 1 else "?") for g, s in strands.items()}


def input_gm_keys(path: Path) -> tuple[dict[str, str], int]:
    """({gene_id: the value the model will stamp as GM=}, ambiguous gene count).

    The model derives `GM=` as the first attribute value on the input gene row whatever its
    key. Two gene rows sharing that value make the tag ambiguous; picking either would be a
    silent mis-attribution, so those genes are dropped from the map (sending them to the
    positional fallback) and counted instead.
    """
    _id_type, _id_parent, _id_gbkey, _gene_gm, lines = _parse_hierarchy(path)
    first: dict[str, str] = {}
    for line in lines:
        f = line.split("\t")
        if len(f) < 9 or f[2] != "gene":
            continue
        fid = _attr(f[8], "ID")
        value = _first_attribute_value(f[8])
        if fid and value:
            first[fid] = value
    counts = Counter(first.values())
    ambiguous = {g for g, v in first.items() if counts[v] > 1}
    return {g: v for g, v in first.items() if g not in ambiguous}, len(ambiguous)


def pair_input_to_output(inp: dict, out: dict, input_gff: Path,
                         output_gff: Path) -> tuple[dict, dict]:
    """Pair each input locus to at most one output locus, preferring `GM=` over position.

    Structurally this is `32_score_polishing.py.pair_input_to_output` — same GM-first rule,
    same "claimed output loci are removed from the fallback pool", same collision handling,
    and the positional fallback IS that module's `match_by_overlap`. The one difference is
    the key: the GM value is looked up through `input_gm_keys` (the gene row's first
    attribute value) rather than assumed to be the gene's `ID=`. That module was left
    unchanged because changing it would move published polishing numbers.
    """
    gm_key, ambiguous = input_gm_keys(input_gff)
    out_gm = _POLISH._gene_gm_values(output_gff)
    gm_to_out: dict[str, list] = defaultdict(list)
    for o in out:
        gm = out_gm.get(o[3])
        if gm is not None:
            gm_to_out[gm].append(o)

    stats = dict(gm_paired=0, overlap_fallback=0, output_split=0, output_merged=0,
                 input_without_gm_key=0, ambiguous_input_gm_keys=ambiguous)
    raw_gm_pairs: dict = {}
    needs_overlap: dict = {}
    for i in inp:
        key = gm_key.get(i[3])
        if key is None:
            stats["input_without_gm_key"] += 1
            needs_overlap[i] = True
            continue
        candidates = gm_to_out.get(key, [])
        if len(candidates) == 1:
            raw_gm_pairs[i] = candidates[0]
        elif len(candidates) > 1:
            stats["output_split"] += 1
        else:
            needs_overlap[i] = True

    out_usage = Counter(raw_gm_pairs.values())
    pairs: dict = {}
    for i, o in raw_gm_pairs.items():
        if out_usage[o] > 1:
            stats["output_merged"] += 1
            continue
        pairs[i] = o
        stats["gm_paired"] += 1

    if needs_overlap:
        claimed = set(pairs.values())
        available = {o: txs for o, txs in out.items() if o not in claimed}
        fallback = match_by_overlap({i: inp[i] for i in needs_overlap}, available)
        fallback_usage = Counter(fallback.values())
        for i, o in fallback.items():
            if fallback_usage[o] > 1:
                stats["output_merged"] += 1
                continue
            pairs[i] = o
            stats["overlap_fallback"] += 1

    return pairs, stats


def _reference_context(ref: dict, ref_strand: dict, art_by_gene: dict,
                       primary_ids: set[str] | None) -> tuple[dict, int, int]:
    """Per reference locus: its alternative structures, their chains, its strand and its
    AtRTD3 counterpart. Returns (context, primary_fallback_count, alternatives_genome_wide).

    "Alternative" means every transcript at the locus except the curated primary, resolved
    through `primary_transcript_ids.txt` and NOT through file order — the same split
    `32_score_polishing.py` makes. The AtRTD3 side removes nothing, matching
    `28_score_added_isoforms.py`, which asks only whether the structure is annotated
    anywhere in AtRTD3.
    """
    context: dict = {}
    fallback = 0
    genome_wide = 0
    for span, txs in ref.items():
        primary_struct, fell_back = _primary_representative(txs, primary_ids)
        if fell_back:
            fallback += 1
        alternatives = {s for s in txs.values() if s != primary_struct}
        genome_wide += len(alternatives)
        art = art_by_gene.get(_agi(span[3]))
        context[span] = {
            "alt": alternatives,
            "alt_chains": {chain(s) for s in alternatives},
            "strand": ref_strand.get(span[3], "?"),
            "art": set(art.values()) if art is not None else None,
        }
    return context, fallback, genome_wide


def _tally(inp: dict, out: dict, io_pairs: dict, in_to_ref: dict, context: dict,
           out_strand: dict) -> Counter:
    """Walk every input locus once and count. All headline and decomposition counters come
    out of this single pass, so they cannot drift apart.
    """
    t: Counter = Counter()
    for i_span, in_txs in inp.items():
        o_span = io_pairs.get(i_span)
        r_span = in_to_ref.get(i_span)
        if o_span is None:
            t["loci_without_output"] += 1
            if r_span is not None:
                t["reference_alternative_transcripts_at_loci_without_output"] += len(
                    context[r_span]["alt"])
            continue

        t["loci_scored"] += 1
        if len(in_txs) > 1:
            t["input_loci_with_multiple_transcripts"] += 1
        supplied = _representative(in_txs)
        input_all = set(in_txs.values())
        output_all = set(out[o_span].values())
        # A set, so two beams or two identical emissions are one proposed structure.
        additions = {s for s in output_all if s != supplied}
        survived = supplied in output_all

        t["added_structures"] += len(additions)
        t["added_structures_vs_all_input_transcripts"] += len(output_all - input_all)
        if additions:
            t["loci_with_at_least_one_addition"] += 1
        if survived:
            t["prompt_survived_loci"] += 1
            t["added_at_loci_where_prompt_survived"] += len(additions)
        else:
            t["prompt_destroyed_loci"] += 1
            t["added_at_loci_where_prompt_destroyed"] += len(additions)

        if r_span is None:
            t["input_loci_without_reference_match"] += 1
            continue

        ctx = context[r_span]
        t["added_structures_at_reference_matched_loci"] += len(additions)
        t["added_structures_vs_all_input_transcripts_at_reference_matched_loci"] += len(
            output_all - input_all)
        t["reference_alternative_transcripts"] += len(ctx["alt"])

        exact = additions & ctx["alt"]
        t["added_matching_TAIR10_alternative_exact_CDS"] += len(exact)
        t["alt_recovered"] += len(ctx["alt"] & additions)
        if survived:
            t["added_matching_TAIR10_alternative_exact_CDS_at_prompt_survived_loci"] += len(
                exact)
        if exact and out_strand.get(o_span[3], "?") != ctx["strand"]:
            t["added_matching_TAIR10_alternative_exact_CDS_opposite_strand"] += len(exact)

        prompt_chain = chain(supplied)
        for structure in additions:
            c = chain(structure)
            if len(c) >= 1 and c in ctx["alt_chains"]:
                t["added_matching_TAIR10_alternative_intron_chain"] += 1
                if c == prompt_chain:
                    t["intron_chain_reusing_prompt_chain"] += 1
                else:
                    t["intron_chain_distinct_from_prompt"] += 1

        if ctx["art"] is None:
            t["reference_loci_without_AtRTD3_counterpart"] += 1
        else:
            t["added_structures_at_AtRTD3_covered_loci"] += len(additions)
            t["added_matching_any_AtRTD3_transcript"] += len(additions & ctx["art"])
    return t


def _pct(numerator: int, denominator: int) -> float | None:
    """A percentage, or None when the denominator is 0 — never 0.0, which would read as a
    measured result rather than an absent one."""
    return round(100 * numerator / denominator, 1) if denominator else None


def score(input_gff: Path, output_gff: Path, tair10_gtf: Path = TAIR10_GTF,
          atrtd3_gtf: Path = ATRTD3_GTF, primary_ids_path: Path | None = PRIMARY_IDS,
          tool: str | None = None, cli: dict | None = None) -> dict:
    inp = cds_structures(input_gff)
    out = cds_structures(output_gff)
    ref, ref_strand = gtf_cds_structures(tair10_gtf)
    art_loci, _art_strand = gtf_cds_structures(atrtd3_gtf)
    for name, d, p in (("input", inp, input_gff), ("output", out, output_gff),
                       ("TAIR10 reference", ref, tair10_gtf),
                       ("AtRTD3 reference", art_loci, atrtd3_gtf)):
        if not d:
            raise ValueError(f"no CDS-bearing transcripts found in {name} file {p} — "
                             "refusing to score, this would silently report all zeros")
    art_by_gene = {span[3]: txs for span, txs in art_loci.items()}

    raw_pairs = match_by_overlap(inp, ref)
    if not raw_pairs:
        raise RuntimeError(
            "no input locus overlaps any reference locus on the same sequence — this is "
            "the sequence-name-mismatch failure mode, check seqids match exactly "
            f"(input seqs={sorted({s[0] for s in inp})}, "
            f"reference seqs={sorted({s[0] for s in ref})})")
    in_to_ref, split_predictions = _resolve_one_to_one(raw_pairs)

    io_pairs, pairing = pair_input_to_output(inp, out, input_gff, output_gff)
    if not io_pairs:
        raise RuntimeError(
            "no input locus could be paired to any output locus (no GM= match and no "
            "positional overlap) — check the output corresponds to this input "
            f"(input genes={sorted({s[3] for s in inp})[:5]}…, "
            f"output GM values={sorted(set(_POLISH._gene_gm_values(output_gff).values()))[:5]}…)")

    primary_ids = _load_primary_ids(primary_ids_path)
    context, primary_fallback, alternatives_genome_wide = _reference_context(
        ref, ref_strand, art_by_gene, primary_ids)
    out_strand = gff3_gene_strands(output_gff)

    t = _tally(inp, out, io_pairs, in_to_ref, context, out_strand)
    # The one-to-one resolution sets aside every prediction but the best when a tool splits
    # one reference gene across several loci. That is right for the denominator, but it
    # also removes those loci's additions from the numerator, which is a one-directional
    # loss. Scoring the raw many-to-one pairing bounds how much it costs.
    t_raw = _tally(inp, out, io_pairs, raw_pairs, context, out_strand)

    added = t["added_structures"]
    added_matched = t["added_structures_at_reference_matched_loci"]
    added_survived = t["added_at_loci_where_prompt_survived"]
    added_art = t["added_structures_at_AtRTD3_covered_loci"]
    alt_total = t["reference_alternative_transcripts"]

    gene_rows = _POLISH._gene_row_count(input_gff)
    genes_without_cds = gene_rows - len(inp)

    return {
        "provenance": {
            "input": str(input_gff),
            "output": str(output_gff),
            "reference_TAIR10": str(tair10_gtf),
            "reference_AtRTD3": str(atrtd3_gtf),
            "primary_ids": str(primary_ids_path) if primary_ids_path else None,
            "tool": tool,
            "argv": list(cli["argv"]) if cli else None,
            "cli_args": cli["args"] if cli else None,
        },
        "definitions": {
            "added_structure": ("a distinct CDS structure in the output that differs from "
                                "the structure supplied for that locus (the input's first "
                                "mRNA in file order), identical emissions collapsed to one"),
            "TAIR10_alternative": ("any TAIR10 transcript at the matched locus other than "
                                   "the curated primary from primary_transcript_ids.txt"),
            "AtRTD3_match": "equality with ANY AtRTD3 transcript at the matched locus",
        },
        "loci_scored": t["loci_scored"],
        "loci_with_at_least_one_addition": t["loci_with_at_least_one_addition"],
        "added_structures": added,
        "reference_alternative_transcripts": alt_total,
        "added_matching_TAIR10_alternative_exact_CDS":
            t["added_matching_TAIR10_alternative_exact_CDS"],
        "added_matching_TAIR10_alternative_intron_chain":
            t["added_matching_TAIR10_alternative_intron_chain"],
        "added_matching_any_AtRTD3_transcript": t["added_matching_any_AtRTD3_transcript"],
        "precision_vs_TAIR10_alternatives_pct":
            _pct(t["added_matching_TAIR10_alternative_exact_CDS"], added),
        "precision_vs_TAIR10_alternatives_intron_chain_pct":
            _pct(t["added_matching_TAIR10_alternative_intron_chain"], added),
        "precision_vs_AtRTD3_pct": _pct(t["added_matching_any_AtRTD3_transcript"], added),
        "recall_of_TAIR10_alternatives_pct": _pct(t["alt_recovered"], alt_total),
        "decomposition": {
            "added_structures_vs_all_input_transcripts":
                t["added_structures_vs_all_input_transcripts"],
            "input_loci_with_multiple_transcripts": t["input_loci_with_multiple_transcripts"],
            "prompt_survived_loci": t["prompt_survived_loci"],
            "prompt_destroyed_loci": t["prompt_destroyed_loci"],
            "added_at_loci_where_prompt_survived": added_survived,
            "added_at_loci_where_prompt_destroyed": t["added_at_loci_where_prompt_destroyed"],
            "added_matching_TAIR10_alternative_exact_CDS_at_prompt_survived_loci":
                t["added_matching_TAIR10_alternative_exact_CDS_at_prompt_survived_loci"],
            "precision_vs_TAIR10_alternatives_at_prompt_survived_loci_pct":
                _pct(t["added_matching_TAIR10_alternative_exact_CDS_at_prompt_survived_loci"],
                     added_survived),
            "added_matching_TAIR10_alternative_intron_chain_reusing_prompt_chain":
                t["intron_chain_reusing_prompt_chain"],
            "added_matching_TAIR10_alternative_intron_chain_distinct_from_prompt":
                t["intron_chain_distinct_from_prompt"],
            "added_structures_at_reference_matched_loci": added_matched,
            "added_structures_vs_all_input_transcripts_at_reference_matched_loci":
                t["added_structures_vs_all_input_transcripts_at_reference_matched_loci"],
            "precision_vs_TAIR10_alternatives_at_reference_matched_loci_pct":
                _pct(t["added_matching_TAIR10_alternative_exact_CDS"], added_matched),
            "added_structures_at_AtRTD3_covered_loci": added_art,
            "precision_vs_AtRTD3_at_covered_loci_pct":
                _pct(t["added_matching_any_AtRTD3_transcript"], added_art),
        },
        "diagnostics": {
            "input_loci": len(inp),
            "input_loci_matched_to_reference": len(in_to_ref),
            "input_loci_without_reference_match": t["input_loci_without_reference_match"],
            "split_predictions": split_predictions,
            "loci_without_output": t["loci_without_output"],
            "reference_alternative_transcripts_at_loci_without_output":
                t["reference_alternative_transcripts_at_loci_without_output"],
            "output_loci": len(out),
            "output_loci_without_input_partner": len(out) - len(set(io_pairs.values())),
            "reference_loci": len(ref),
            "reference_alternative_transcripts_genome_wide": alternatives_genome_wide,
            "reference_primary_fallback_to_file_order": primary_fallback,
            "reference_loci_without_AtRTD3_counterpart":
                t["reference_loci_without_AtRTD3_counterpart"],
            "added_matching_TAIR10_alternative_exact_CDS_opposite_strand":
                t["added_matching_TAIR10_alternative_exact_CDS_opposite_strand"],
            "input_gene_rows": gene_rows,
            "input_coding_loci": len(inp),
            "input_genes_without_cds": genes_without_cds if genes_without_cds >= 0 else None,
            "input_noncoding_features_excluded":
                _POLISH._noncoding_feature_counts(input_gff),
            "ambiguous_input_gm_keys": pairing.pop("ambiguous_input_gm_keys"),
            "pairing": pairing,
            "without_one_to_one_resolution": {
                "note": ("the same tally with split predictions left in — bounds how much "
                         "the one-to-one reference resolution removes from the numerator"),
                "added_structures_at_reference_matched_loci":
                    t_raw["added_structures_at_reference_matched_loci"],
                "added_matching_TAIR10_alternative_exact_CDS":
                    t_raw["added_matching_TAIR10_alternative_exact_CDS"],
                "added_matching_TAIR10_alternative_intron_chain":
                    t_raw["added_matching_TAIR10_alternative_intron_chain"],
                "added_matching_any_AtRTD3_transcript":
                    t_raw["added_matching_any_AtRTD3_transcript"],
                "reference_alternative_transcripts":
                    t_raw["reference_alternative_transcripts"],
            },
        },
    }


def _print_result(res: dict, indent: str = "  ") -> None:
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{indent}{k}:", file=sys.stderr)
            _print_result(v, indent + "  ")
        else:
            print(f"{indent}{k:<64} {v}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score the alternatively spliced isoforms TransGenic adds to an "
                    "external annotation used as the completion-mode prompt.")
    ap.add_argument("--input", type=Path, required=True,
                    help="the external annotation used as the prompt")
    ap.add_argument("--output", type=Path, required=True,
                    help="TransGenic's completion of that annotation")
    ap.add_argument("--tair10", type=Path, default=TAIR10_GTF)
    ap.add_argument("--atrtd3", type=Path, default=ATRTD3_GTF)
    ap.add_argument("--primary-ids", type=Path, default=PRIMARY_IDS)
    ap.add_argument("--tool", default=None, help="tool label recorded in the provenance")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    for label, path in (("--input", args.input), ("--output", args.output),
                        ("--tair10", args.tair10), ("--atrtd3", args.atrtd3)):
        if not path.exists():
            print(f"MISSING {label}: {path}", file=sys.stderr)
            return 1

    cli = {
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    res = score(args.input, args.output, args.tair10, args.atrtd3, args.primary_ids,
                args.tool, cli)
    _print_result(res)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=1))
        print(f"written: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
