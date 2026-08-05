#!/usr/bin/env python3
"""Score whether completion mode moved an external annotation toward TAIR10.

Every completion-mode result in the manuscript prompts the model with TAIR10's own
primary transcript, which is returned unchanged at 99.4% of loci — there is nothing to
repair. External annotations (GeMoMa, BRAKER3, EGAPx) disagree with TAIR10, so they are
the first input on which polishing is measurable.

For each locus present in both the input annotation and the reference, the input model
and the output model are each classified as matching TAIR10 or not, giving four cells:

    preserved_correct   input right, output right
    repaired            input wrong, output right     <- polishing
    damaged             input right, output wrong     <- the cost of polishing
    still_wrong         input wrong, output wrong

Two definitions of "matching" are reported side by side, because they disagree by
design, not by accident:

    cds_level           the representative equals ANY reference transcript at the locus
    cds_level_primary   the representative equals the reference's PRIMARY transcript only,
                         resolved via revision/data/TAIR10/primary_transcript_ids.txt

TAIR10's GFF3 is coordinate-sorted, so "first transcript in file order" is the
leftmost-starting isoform at a locus, not the biological primary — of TAIR10's 5,804
multi-isoform loci, 2,496 resolve to something other than ".1". `cds_level` sidesteps
that entirely (any transcript matching is "correct"); `cds_level_primary` answers the
stricter question by reading the curated primary-transcript list instead of guessing
from file order. The equivalent split (`utr_level`/`utr_level_primary`) is computed the
same way for CDS+UTR (exon-chain) structures, when the input supports it (see below).

"The input/output model" for a locus is still singular either way: GeMoMa, BRAKER3 and
EGAPx already carry alternative isoforms of their own (mRNA count exceeds gene count in
every staged input — GeMoMa alone has 7,326 extra transcripts), so the representative
is the FIRST mRNA for that gene in file order — the convention
`28_score_added_isoforms.py --augustus` uses for the AUGUSTUS baseline. This "first in
file order" convention is about the input/prediction side (which transcript stands in
for what the tool proposed); it is never used to pick the reference's primary — that is
what `cds_level_primary` is for. Structures the output adds beyond everything the input
already had (not just its representative) are counted separately and scored against the
reference's other transcripts at that locus, so a tool's own pre-existing extra isoforms
are never credited to the output as something it "added".

Two of the three staged inputs cannot support a UTR-level comparison: GeMoMa emits no
exon rows at all, and BRAKER3's exon coordinates equal its CDS coordinates everywhere (no
UTR was predicted). This is detected from both the input AND the output file (whichever
side lacks real UTR signal makes the comparison unmeasurable), not hardcoded by tool
name, so the UTR-level table reports "N/A" with a reason instead of a misleading zero.

Input and output loci are paired on the model's own `GM=` provenance tag when present —
completion mode stamps every gene/mRNA/CDS/UTR row of its output with the exact gene id
string of the locus it was prompted from (see `src/transgenic/utils/gsf.py`'s
`extra_attributes` and `examples/prompt_mode.py`'s `gm_id`) — falling back to positional
overlap only where `GM=` is absent. Position is what drifts under damage, which is
exactly the case `damaged` exists to count, so anchoring the pairing on identity instead
of position is what keeps a badly-damaged locus from silently vanishing from the
denominator instead of being counted as damage. Input loci are also matched to the
reference one-to-one: when GeMoMa/BRAKER3/EGAPx split one TAIR10 gene into several of
their own predictions, only the best-overlapping one is scored against it, and the rest
are counted in `split_predictions` instead of each contributing (and each getting scored)
as if it were a distinct comparable locus.

EGAPx also carries non-coding loci (lnc_RNA, pseudogene, misc_RNA-flavoured "transcript"
rows) that have no CDS at all. These are excluded from the locus counts automatically,
because loci are only ever discovered by walking up from CDS rows — but that exclusion is
silent unless reported, so `input_gene_rows`, `input_coding_loci`,
`input_genes_without_cds` and `input_noncoding_features_excluded` are all written into the
output JSON so the three tools' denominators reconcile instead of just differing
unexplained.

Usage:
    python 32_score_polishing.py --input <staged.gff3> --output <completed.gff3> \\
        [--reference <tair10.gff3>] [--tool <name>] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = (ROOT / "transgenic_comparison" / "reference_annotations"
             / "Athaliana_167_TAIR10.gene.clean.gff3")
PRIMARY_IDS = ROOT / "transgenic" / "revision" / "data" / "TAIR10" / "primary_transcript_ids.txt"

# A CDS (or exon) row's Parent is normally an mRNA (or, in some GFF3 dialects, a bare
# "transcript") id. This module's own unit tests use a flattened 2-level fixture where
# CDS rows are parented directly to the gene, with no mRNA row in between at all — see
# `_tx_and_gene` for how both shapes resolve to the same (transcript, gene) pair.
TRANSCRIPT_TYPES = {"mRNA", "transcript"}

# Feature types that are never protein-coding, by type alone, regardless of dialect.
# "transcript" is deliberately NOT here: EGAPx uses that exact feature type for both its
# non-coding misc_RNA rows and — in other GFF3 dialects — a coding transcript can also be
# typed "transcript". The two are told apart by `gbkey=`, not by the type string; see
# `_is_noncoding`.
NONCODING_TYPES = {"lnc_RNA", "pseudogene"}
NONCODING_GBKEYS = {"misc_RNA", "ncRNA"}

_ATTR_RE_CACHE: dict[str, re.Pattern] = {}


def _attr(attributes: str, key: str) -> str | None:
    """Read `key=value` out of a GFF3 attributes column, anchored on `key` itself.

    Unanchored `key=` would match inside a *different* attribute whose name happens to
    end in `key` (e.g. looking for `ID=` matching inside `GeneID=`). GeMoMa writes
    `Name=` before `ID=` on the same line, so attribute ordering is no safeguard either.
    """
    pattern = _ATTR_RE_CACHE.get(key)
    if pattern is None:
        pattern = _ATTR_RE_CACHE[key] = re.compile(rf"(?:^|;){key}=([^;\n]+)")
    m = pattern.search(attributes)
    return m.group(1) if m else None


@lru_cache(maxsize=8)
def _parse_hierarchy(path: Path) -> tuple[dict, dict, dict, dict, tuple]:
    """One pass over `path`: ID -> feature type, ID -> Parent, ID -> gbkey,
    gene ID -> GM= (only ever present on completion-mode output), and the raw lines.

    Cached per path so a single `score()` call — which reads CDS rows and, when the
    input carries UTR information, exon rows too — does not re-read a 100+ MB GFF3
    from disk more than once.
    """
    id_type: dict[str, str] = {}
    id_parent: dict[str, str] = {}
    id_gbkey: dict[str, str] = {}
    gene_gm: dict[str, str] = {}
    lines = path.read_text().splitlines()
    for line in lines:
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9:
            continue
        fid = _attr(f[8], "ID")
        if fid:
            id_type[fid] = f[2]
            fparent = _attr(f[8], "Parent")
            if fparent:
                id_parent[fid] = fparent
            gbkey = _attr(f[8], "gbkey")
            if gbkey:
                id_gbkey[fid] = gbkey
            if f[2] == "gene":
                gm = _attr(f[8], "GM")
                if gm:
                    gene_gm[fid] = gm
    return id_type, id_parent, id_gbkey, gene_gm, tuple(lines)


def _is_noncoding(ptype: str | None, gbkey: str | None) -> bool:
    return gbkey in NONCODING_GBKEYS or ptype in NONCODING_TYPES


def _tx_and_gene(fparent: str | None, fid: str | None,
                  id_type: dict, id_parent: dict, id_gbkey: dict
                  ) -> tuple[str, str, str, bool]:
    """Resolve a CDS/exon row's own (Parent, ID) to (status, transcript_id, gene_id,
    was_fallback).

    Real annotations: Parent is an mRNA/transcript id -> transcript is that id, gene is
    its own Parent. This module's fixtures skip the mRNA level entirely (CDS parented
    straight to the gene, with the transcript identity only distinguishable by the CDS
    row's own ID) — that shape is detected because the Parent value was never declared
    as a transcript anywhere in the file, and handled directly. On a real, well-formed
    3-level file this fallback should never fire; `was_fallback=True` lets the caller
    count how often it did.

    status is "excluded" when the row is parented to a declared non-coding feature.
    """
    if fparent is None:
        return "excluded", None, None, False
    ptype = id_type.get(fparent)
    if _is_noncoding(ptype, id_gbkey.get(fparent)):
        return "excluded", None, None, False
    if ptype in TRANSCRIPT_TYPES:
        return "ok", fparent, id_parent.get(fparent, fparent), False
    return "ok", (fid or fparent), fparent, True


def _transcript_structures(path: Path, feature: str) -> tuple[dict, dict, dict, int]:
    """{tx_id: sorted segment tuple}, {tx_id: gene_id}, {tx_id: seq}, fallback_count
    over `feature` rows.

    Dict insertion order follows file order, which is what lets callers pick "the
    first transcript for a gene" without a separate ordering pass.

    Raises if a CDS row is parented to a declared non-coding feature: on real data this
    never happens (verified against the staged GeMoMa/BRAKER3/EGAPx files), so if it ever
    does, that is a real surprise worth a human looking at, not a locus quietly dropped.
    The same situation for exon rows is routine (EGAPx's lnc_RNA/pseudogene/misc_RNA loci
    legitimately have exon structure — 7,274/1,221/480 such rows in the staged file) and
    is only counted, never raised.
    """
    id_type, id_parent, id_gbkey, _gene_gm, lines = _parse_hierarchy(path)
    tx_segs: dict[str, list] = defaultdict(list)
    tx_gene: dict[str, str] = {}
    tx_seq: dict[str, str] = {}
    fallback_count = 0
    excluded_count = 0
    for line in lines:
        f = line.split("\t")
        if len(f) < 9 or f[2] != feature:
            continue
        fid = _attr(f[8], "ID")
        fparent = _attr(f[8], "Parent")
        status, tx, gene, fallback = _tx_and_gene(fparent, fid, id_type, id_parent, id_gbkey)
        if status == "excluded":
            excluded_count += 1
            continue
        if fallback:
            fallback_count += 1
        tx_segs[tx].append((int(f[3]), int(f[4])))
        tx_gene[tx] = gene
        tx_seq[tx] = f[0]
    if feature == "CDS" and excluded_count:
        raise RuntimeError(
            f"{excluded_count} CDS row(s) in {path} are parented to a declared "
            "non-coding feature (lnc_RNA/pseudogene/misc_RNA/ncRNA) — refusing to fold "
            "them into a locus; this needs a human look, not a silent drop"
        )
    structs = {tx: tuple(sorted(segs)) for tx, segs in tx_segs.items()}
    return structs, tx_gene, tx_seq, fallback_count


def _group_by_gene(structs: dict, tx_gene: dict, tx_seq: dict) -> dict:
    """{(seq, start, end, gene_id): {tx_id: struct}} grouping transcripts under genes.

    The span is the union of all of that gene's transcripts' segments, so a gene with
    several isoforms of different extents is still one locus, not one per isoform. The
    gene id rides along as a 4th tuple element purely to keep the key unique — BRAKER3
    alone has 8 genes whose aggregate CDS span exactly coincides with another gene's
    (adjacent or nested single-exon predictions), and a bare (seq, start, end) key would
    silently drop one of every such pair. `_overlaps`/`_jaccard` only ever read indices
    0-2, so the extra element does not change any matching behaviour.
    """
    gene_txs: dict[str, dict] = defaultdict(dict)
    for tx, segs in structs.items():  # insertion order == file discovery order
        gene_txs[tx_gene[tx]][tx] = segs
    out: dict = {}
    for gene, txs in gene_txs.items():
        seq = tx_seq[next(iter(txs))]
        starts = [s for segs in txs.values() for s, _e in segs]
        ends = [e for segs in txs.values() for _s, e in segs]
        out[(seq, min(starts), max(ends), gene)] = txs
    return out


def cds_structures(path: Path) -> dict:
    """{locus: {transcript: ((start, end), …)}} over CDS rows, grouped by gene."""
    structs, tx_gene, tx_seq, _fallback = _transcript_structures(path, "CDS")
    return _group_by_gene(structs, tx_gene, tx_seq)


def _representative(txs: dict) -> tuple:
    """The first transcript for this locus in file order — "the input/output model"."""
    return next(iter(txs.values()))


def _load_primary_ids(path: Path | None) -> set[str] | None:
    if path is None or not path.exists():
        return None
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _primary_representative(txs: dict, primary_ids: set[str] | None) -> tuple[tuple, bool]:
    """(structure, fell_back_to_file_order) for a reference locus's true primary
    transcript, resolved via `primary_ids` (TAIR10's curated primary_transcript_ids.txt)
    when available. Falls back to file order — the same convention `_representative`
    uses — when the reference isn't TAIR10 or this gene isn't covered by the list.
    """
    if primary_ids:
        for tx, struct in txs.items():
            if tx in primary_ids:
                return struct, False
    return _representative(txs), True


def _gene_gm_values(path: Path) -> dict[str, str]:
    """{gene_id: GM value}, only ever populated on completion-mode output."""
    _id_type, _id_parent, _id_gbkey, gene_gm, _lines = _parse_hierarchy(path)
    return gene_gm


def _overlaps(a: tuple, b: tuple) -> bool:
    return a[0] == b[0] and a[1] <= b[2] and b[1] <= a[2]


def _jaccard(a: tuple, b: tuple) -> float:
    """Reciprocal overlap of two 1-based inclusive spans (intersection / union)."""
    ov = min(a[2], b[2]) - max(a[1], b[1]) + 1
    if ov <= 0:
        return 0.0
    union = (a[2] - a[1] + 1) + (b[2] - b[1] + 1) - ov
    return ov / union if union > 0 else 0.0


def match_by_overlap(pred: dict, ref: dict, stats: dict | None = None) -> dict:
    """Pair predicted loci to reference loci by reciprocal overlap on the same sequence.

    Scored by Jaccard overlap (intersection / union of the two spans), not raw overlap
    length: TAIR10 has nested and overlapping gene models, and a gene that is fully
    engulfed by a much larger neighbour ties on raw overlap length with its true,
    equal-span match. Reciprocal overlap breaks that tie correctly — confirmed by the
    scorer's own self-test (score the reference against itself; every locus must land
    in preserved_correct, none in repaired) surfacing exactly this failure mode.

    A second, rarer tie remains even under Jaccard: a handful of TAIR10 loci share the
    exact same aggregate span as a neighbour — overlapping/antisense gene models with
    coincidentally identical extents (e.g. AT1G16860 and AT1G16858). Position alone
    cannot break that tie, so the representative structure does: whichever candidate's
    *first-in-file-order transcript* — the same one the transition table classifies on
    — equals `pred[p]`'s wins a tied score. This is not gene-id based (gene ids are not
    comparable across annotation tools), so it applies equally to GeMoMa/BRAKER3/EGAPx
    vs TAIR10, not just the same-tool self-test that exposed it. This tie-break is
    scored on the same equality the transition table then reports, so whichever tied
    sibling a locus is matched to, it can never land in `damaged`/`still_wrong` purely
    because of the tie — bounded to the loci that actually tie (≤85 of 27,416 TAIR10
    loci at the exon level, 0 at the CDS level), and reported via `stats` so a reader
    can size it rather than discover it by accident.

    If `stats` is given, `stats["ties_broken_by_structure"]` is incremented whenever two
    candidates tie on Jaccard score but differ on the structural bonus.
    """
    by_seq: dict = defaultdict(list)
    for span in ref:
        by_seq[span[0]].append(span)
    for k in by_seq:
        by_seq[k].sort(key=lambda s: s[1])
    pairs = {}
    for p in pred:
        p_repr = _representative(pred[p])
        best, best_key = None, (0.0, 0)
        overridden_by_bonus = False
        for r in by_seq.get(p[0], []):
            if r[1] > p[2]:
                break
            if _overlaps(p, r):
                score = _jaccard(p, r)
                bonus = 1 if p_repr == _representative(ref[r]) else 0
                key = (score, bonus)
                if key > best_key:
                    # A strictly-better key that ties on raw score with the previous
                    # best means the bonus, not position, decided the winner.
                    if best is not None and score == best_key[0]:
                        overridden_by_bonus = True
                    best, best_key = r, key
        if best:
            pairs[p] = best
            if overridden_by_bonus and stats is not None:
                stats["ties_broken_by_structure"] = stats.get("ties_broken_by_structure", 0) + 1
    return pairs


def _resolve_one_to_one(raw_pairs: dict, pred: dict, ref: dict) -> tuple[dict, int]:
    """Keep the single best-scoring pred locus per reference locus.

    GeMoMa/BRAKER3/EGAPx can split one TAIR10 gene into several of their own predicted
    loci (619/366/86 reference genes receive >=2 input loci in the real staged data).
    `match_by_overlap` is pred-centric and happily assigns every one of them to the same
    best-overlapping reference locus; left as-is, that reference gene is scored — and its
    repairs/damage counted — once per splitting prediction. Here only the best-Jaccard
    prediction per reference locus is kept; the rest are reported via the returned count
    instead of silently inflating `loci_compared`.
    """
    by_ref: dict = defaultdict(list)
    for p, r in raw_pairs.items():
        by_ref[r].append(p)
    resolved: dict = {}
    split_predictions = 0
    for r, preds in by_ref.items():
        if len(preds) == 1:
            resolved[preds[0]] = r
            continue
        best = max(preds, key=lambda p: _jaccard(p, r))
        resolved[best] = r
        split_predictions += len(preds) - 1
    return resolved, split_predictions


def pair_input_to_output(inp: dict, out: dict, output_gff: Path) -> tuple[dict, dict]:
    """Pair each input locus to at most one output locus.

    Prefers the model's own `GM=` provenance tag over position: completion-mode output
    stamps every row with the exact gene id string of the input locus it was prompted
    from, so this pairing survives however far the output's *position* drifts, which
    matters specifically for `damaged` loci — the more a locus is damaged, the more
    likely it drifts positionally toward a neighbour, and a position-only pairing would
    silently lose exactly the loci this metric exists to catch. Falls back to positional
    overlap only for input loci with no usable GM= match.
    """
    out_gm = _gene_gm_values(output_gff)
    gm_to_out: dict[str, list] = defaultdict(list)
    for o in out:
        gm = out_gm.get(o[3])
        if gm is not None:
            gm_to_out[gm].append(o)

    raw_gm_pairs: dict = {}
    stats = dict(gm_paired=0, overlap_fallback=0, output_split=0, output_merged=0)
    needs_overlap: dict = {}
    for i in inp:
        candidates = gm_to_out.get(i[3], [])
        if len(candidates) == 1:
            raw_gm_pairs[i] = candidates[0]
        elif len(candidates) > 1:
            stats["output_split"] += 1  # this input's model was split into >1 outputs
        else:
            needs_overlap[i] = True

    out_usage = Counter(raw_gm_pairs.values())
    pairs: dict = {}
    for i, o in raw_gm_pairs.items():
        if out_usage[o] > 1:
            stats["output_merged"] += 1  # >1 input loci claim the same output via GM
            continue
        pairs[i] = o
        stats["gm_paired"] += 1

    if needs_overlap:
        fallback = match_by_overlap({i: inp[i] for i in needs_overlap}, out)
        for i, o in fallback.items():
            pairs[i] = o
        stats["overlap_fallback"] = len(fallback)

    return pairs, stats


def _bump(table: dict, in_ok: bool, out_ok: bool) -> None:
    table["loci_compared"] += 1
    if in_ok and out_ok:
        table["preserved_correct"] += 1
    elif not in_ok and out_ok:
        table["repaired"] += 1
    elif in_ok and not out_ok:
        table["damaged"] += 1
    else:
        table["still_wrong"] += 1


def _new_table() -> dict:
    return dict(loci_compared=0, preserved_correct=0, repaired=0, damaged=0, still_wrong=0,
                added_structures=0, added_matching_reference=0, loci_without_output=0)


def _finalize_table(table: dict) -> None:
    n = table["loci_compared"]
    table["repaired_pct"] = round(100 * table["repaired"] / n, 2) if n else None
    table["damaged_pct"] = round(100 * table["damaged"] / n, 2) if n else None
    table["added_precision_pct"] = (
        round(100 * table["added_matching_reference"] / table["added_structures"], 1)
        if table["added_structures"] else None)


def _score_one_level(inp: dict, out: dict, ref: dict, input_gff: Path, output_gff: Path,
                      primary_ids: set[str] | None) -> dict:
    """Score one feature level (CDS or exon), returning both the `headline` (any
    reference transcript matches) and `primary` (primary-transcript-only) tables, plus
    the pairing diagnostics (`split_predictions`, `ties_broken_by_structure`,
    `pairing`).
    """
    match_stats: dict = {}
    in_to_ref_raw = match_by_overlap(inp, ref, match_stats)
    if inp and ref and not in_to_ref_raw:
        raise RuntimeError(
            "no input locus overlaps any reference locus on the same sequence — this "
            "is the sequence-name-mismatch failure mode, check seqids match exactly "
            f"(input seqs={sorted({s[0] for s in inp})}, "
            f"reference seqs={sorted({s[0] for s in ref})})"
        )
    in_to_ref, split_predictions = _resolve_one_to_one(in_to_ref_raw, inp, ref)

    io_pairs, io_stats = pair_input_to_output(inp, out, output_gff)
    if inp and out and not io_pairs:
        raise RuntimeError(
            "no input locus could be paired to any output locus (no GM= match and no "
            "positional overlap) — this is the sequence-name/identity-mismatch failure "
            "mode, check the output actually corresponds to this input"
            f" (input genes={sorted({s[3] for s in inp})[:5]}…, "
            f"output genes={sorted({s[3] for s in out})[:5]}…)"
        )

    headline = _new_table()
    primary = _new_table()
    primary_fallback = 0

    for i_span, r_span in in_to_ref.items():
        ref_txs = ref[r_span]
        o_span = io_pairs.get(i_span)
        if o_span is None:
            headline["loci_without_output"] += 1
            primary["loci_without_output"] += 1
            continue
        out_txs = out[o_span]

        in_repr = _representative(inp[i_span])
        out_repr = _representative(out_txs)
        ref_all = set(ref_txs.values())
        _bump(headline, in_repr in ref_all, out_repr in ref_all)

        ref_primary_struct, fell_back = _primary_representative(ref_txs, primary_ids)
        if fell_back:
            primary_fallback += 1
        _bump(primary, in_repr == ref_primary_struct, out_repr == ref_primary_struct)

        in_all = set(inp[i_span].values())
        out_all = set(out_txs.values())
        added = out_all - in_all
        for table in (headline, primary):
            table["added_structures"] += len(added)
            table["added_matching_reference"] += len(added & ref_all)

    _finalize_table(headline)
    _finalize_table(primary)
    headline["definition"] = ("correct = representative transcript equals ANY reference "
                               "transcript at this locus")
    primary["definition"] = ("correct = representative transcript equals the reference's "
                              "PRIMARY transcript only")
    primary["primary_source"] = str(PRIMARY_IDS) if primary_ids else None
    primary["primary_fallback_to_file_order"] = primary_fallback

    return dict(headline=headline, primary=primary, split_predictions=split_predictions,
                ties_broken_by_structure=match_stats.get("ties_broken_by_structure", 0),
                pairing=io_stats)


def _utr_signal_one(path: Path, label: str) -> tuple[bool, str]:
    cds, _, _, _ = _transcript_structures(path, "CDS")
    exon, _, _, _ = _transcript_structures(path, "exon")
    if not exon:
        return False, f"{label} carries no exon rows: CDS+UTR agreement is not measurable"
    extended = sum(
        1 for tx, cseg in cds.items()
        if (eseg := exon.get(tx)) and
        (eseg[0][0] < cseg[0][0] or eseg[-1][1] > cseg[-1][1] or len(eseg) > len(cseg))
    )
    if extended == 0:
        return False, (f"{label} exon coordinates equal CDS coordinates for every "
                        "transcript: no UTR was predicted")
    return True, ""


def _utr_signal(input_gff: Path, output_gff: Path) -> tuple[bool, str]:
    """Whether BOTH `input_gff` and `output_gff` carry real CDS+UTR (exon) information.

    GeMoMa emits no exon rows at all; BRAKER3 emits exon rows identical to its CDS rows
    (no UTR was predicted). Both are detected here rather than assumed by tool name, so
    a future input is handled the same way its data says it should be. Gated on the
    output too: an input with real UTR signal paired with an output that carries no
    exon rows would otherwise report a full UTR table of zeros instead of N/A.
    """
    has_in, reason = _utr_signal_one(input_gff, "input")
    if not has_in:
        return False, reason
    has_out, reason = _utr_signal_one(output_gff, "output")
    if not has_out:
        return False, reason
    return True, ""


def _noncoding_feature_counts(path: Path) -> dict:
    """Count of each excluded non-coding feature type (`NONCODING_TYPES` plus the
    misc_RNA-flavoured "transcript" rows `NONCODING_GBKEYS` catches) in `path`."""
    id_gbkey: dict[str, str] = {}
    counts: Counter = Counter()
    lines = path.read_text().splitlines()
    for line in lines:
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9:
            continue
        fid = _attr(f[8], "ID")
        if fid:
            gbkey = _attr(f[8], "gbkey")
            if gbkey:
                id_gbkey[fid] = gbkey
    for line in lines:
        f = line.split("\t")
        if len(f) < 3:
            continue
        ftype = f[2]
        if ftype in NONCODING_TYPES:
            counts[ftype] += 1
        elif ftype == "transcript":
            fid = _attr(f[8], "ID") if len(f) >= 9 else None
            if id_gbkey.get(fid) in NONCODING_GBKEYS:
                counts[ftype] += 1
    return dict(counts)


def _gene_row_count(path: Path) -> int:
    """Count of `gene`-type feature rows in `path` (ruling 3's cross-tool denominator)."""
    n = 0
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) >= 3 and f[2] == "gene":
            n += 1
    return n


def score(input_gff: Path, output_gff: Path, ref_gff: Path = REFERENCE,
          primary_ids_path: Path | None = PRIMARY_IDS, tool: str | None = None) -> dict:
    inp = cds_structures(input_gff)
    out = cds_structures(output_gff)
    ref = cds_structures(ref_gff)
    for name, d, p in (("input", inp, input_gff), ("output", out, output_gff),
                        ("reference", ref, ref_gff)):
        if not d:
            raise ValueError(f"no CDS-bearing transcripts found in {name} file {p} — "
                              "refusing to score, this would silently report all zeros")

    primary_ids = _load_primary_ids(primary_ids_path)
    cds_result = _score_one_level(inp, out, ref, input_gff, output_gff, primary_ids)

    has_utr, reason = _utr_signal(input_gff, output_gff)
    if has_utr:
        inp_u = _group_by_gene(*_transcript_structures(input_gff, "exon")[:3])
        out_u = _group_by_gene(*_transcript_structures(output_gff, "exon")[:3])
        ref_u = _group_by_gene(*_transcript_structures(ref_gff, "exon")[:3])
        utr_result = _score_one_level(inp_u, out_u, ref_u, input_gff, output_gff, primary_ids)
        utr_level = utr_result["headline"]
        utr_level_primary = utr_result["primary"]
    else:
        utr_level = {"status": "N/A", "reason": reason}
        utr_level_primary = {"status": "N/A", "reason": reason}

    cds_level = cds_result["headline"]
    cds_level["split_predictions"] = cds_result["split_predictions"]
    cds_level["ties_broken_by_structure"] = cds_result["ties_broken_by_structure"]
    cds_level_primary = cds_result["primary"]

    _, _, _, input_fallback = _transcript_structures(input_gff, "CDS")
    _, _, _, output_fallback = _transcript_structures(output_gff, "CDS")
    _, _, _, reference_fallback = _transcript_structures(ref_gff, "CDS")

    gene_rows = _gene_row_count(input_gff)
    return {
        "provenance": {"input": str(input_gff), "output": str(output_gff),
                       "reference": str(ref_gff), "tool": tool},
        "pairing": cds_result["pairing"],
        "cds_level": cds_level,
        "cds_level_primary": cds_level_primary,
        "utr_level": utr_level,
        "utr_level_primary": utr_level_primary,
        "input_gene_rows": gene_rows,
        "input_coding_loci": len(inp),
        "input_genes_without_cds": gene_rows - len(inp),
        "input_noncoding_features_excluded": _noncoding_feature_counts(input_gff),
        "undeclared_parent_fallback": {
            "input": input_fallback, "output": output_fallback, "reference": reference_fallback,
        },
    }


def _print_result(res: dict, indent: str = "  ") -> None:
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{indent}{k}:", file=sys.stderr)
            _print_result(v, indent + "  ")
        else:
            print(f"{indent}{k:<28} {v}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--primary-ids", type=Path, default=PRIMARY_IDS)
    ap.add_argument("--tool", default=None)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    for label, p in (("--input", args.input), ("--output", args.output),
                      ("--reference", args.reference)):
        if not p.exists():
            print(f"MISSING {label}: {p}", file=sys.stderr)
            return 1

    res = score(args.input, args.output, args.reference, args.primary_ids, args.tool)
    _print_result(res)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=1))
        print(f"written: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
