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

"Matching" means a predicted transcript whose CDS coordinates equal a reference
transcript's exactly. Structures the output adds beyond the input are counted separately
and scored against the reference's other transcripts at that locus.

A locus is a gene, not a transcript. GeMoMa, BRAKER3 and EGAPx already carry alternative
isoforms of their own (mRNA count exceeds gene count in every staged input — GeMoMa alone
has 7,326 extra transcripts), so "the input model" for a locus is deliberately singular:
the FIRST mRNA for that gene in file order, the same convention
`28_score_added_isoforms.py --augustus` uses for the AUGUSTUS baseline. The tool's own
pre-existing extra isoforms are tracked too (to compute what the output *adds*) but never
stand in for "the input model", and never count as something the output added — a locus's
already-present alternative transcripts are not something the model gets credit for
inventing.

Two of the three staged inputs cannot support a UTR-level comparison: GeMoMa emits no
exon rows at all, and BRAKER3's exon coordinates equal its CDS coordinates everywhere (no
UTR was predicted). Both are detected from the file itself, not hardcoded by tool name, so
the UTR-level table reports "N/A" with a reason instead of a misleading zero. EGAPx (and
the reference) carry real UTR extension, so the analogous CDS+UTR transition table is
computed from exon-chain equality wherever the input supports it.

EGAPx also carries non-coding loci (lnc_RNA, pseudogene, plain "transcript" features) that
have no CDS at all. These are excluded from the locus counts automatically, because loci
are only ever discovered by walking up from CDS rows — but that exclusion is silent unless
reported, so the count of excluded non-coding rows is written into the output JSON.

Usage:
    python 32_score_polishing.py --input <staged.gff3> --output <completed.gff3> \\
        [--reference <tair10.gff3>] [--json out.json]
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

# A CDS (or exon) row's Parent is normally an mRNA id (real 3-level gene -> mRNA -> CDS
# annotations). This module's own unit tests use a flattened 2-level fixture where CDS
# rows are parented directly to the gene, with no mRNA row in between at all — see
# `_tx_and_gene` for how both shapes resolve to the same (transcript, gene) pair.
TRANSCRIPT_TYPES = {"mRNA"}

# Feature types that are never protein-coding and must not be pulled into a locus:
# EGAPx stages 2,048 lnc_RNA, 378 pseudogene and 77 bare "transcript" rows alongside its
# 25,131 coding genes. None of them carry CDS children in practice (verified against the
# staged file), so this exclusion is normally a no-op — it exists so a future input that
# *does* attach CDS to one of these fails loudly instead of quietly inflating a locus.
NONCODING_TYPES = {"lnc_RNA", "pseudogene", "transcript"}

_ATTR_RE_CACHE: dict[str, re.Pattern] = {}


def _attr(attributes: str, key: str) -> str | None:
    pattern = _ATTR_RE_CACHE.get(key)
    if pattern is None:
        pattern = _ATTR_RE_CACHE[key] = re.compile(rf"{key}=([^;\n]+)")
    m = pattern.search(attributes)
    return m.group(1) if m else None


@lru_cache(maxsize=8)
def _parse_hierarchy(path: Path) -> tuple[dict, dict, tuple]:
    """One pass over `path`: ID -> feature type, ID -> Parent, and the raw lines.

    Cached per path so a single `score()` call — which reads CDS rows and, when the
    input carries UTR information, exon rows too — does not re-read a 100+ MB GFF3
    from disk more than once.
    """
    id_type: dict[str, str] = {}
    id_parent: dict[str, str] = {}
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
    return id_type, id_parent, tuple(lines)


def _tx_and_gene(fparent: str | None, fid: str | None,
                  id_type: dict, id_parent: dict) -> tuple[str, str] | None:
    """Resolve a CDS/exon row's own (Parent, ID) to (transcript_id, gene_id).

    Real annotations: Parent is an mRNA id -> transcript is that mRNA, gene is the
    mRNA's own Parent. This module's fixtures skip the mRNA level entirely (CDS
    parented straight to the gene, with the transcript identity only distinguishable
    by the CDS row's own ID) — that shape is detected because the Parent value was
    never declared as an mRNA anywhere in the file, and handled directly.

    Returns None when the row is parented to a non-coding feature (`NONCODING_TYPES`)
    and must be excluded rather than silently folded into a locus.
    """
    if fparent is None:
        return None
    ptype = id_type.get(fparent)
    if ptype in NONCODING_TYPES:
        return None
    if ptype in TRANSCRIPT_TYPES:
        return fparent, id_parent.get(fparent, fparent)
    return (fid or fparent), fparent


def _transcript_structures(path: Path, feature: str) -> tuple[dict, dict, dict]:
    """{tx_id: sorted segment tuple}, {tx_id: gene_id}, {tx_id: seq} over `feature` rows.

    Dict insertion order follows file order, which is what lets callers pick "the
    first transcript for a gene" without a separate ordering pass.
    """
    id_type, id_parent, lines = _parse_hierarchy(path)
    tx_segs: dict[str, list] = defaultdict(list)
    tx_gene: dict[str, str] = {}
    tx_seq: dict[str, str] = {}
    for line in lines:
        f = line.split("\t")
        if len(f) < 9 or f[2] != feature:
            continue
        fid = _attr(f[8], "ID")
        fparent = _attr(f[8], "Parent")
        resolved = _tx_and_gene(fparent, fid, id_type, id_parent)
        if resolved is None:
            continue
        tx, gene = resolved
        tx_segs[tx].append((int(f[3]), int(f[4])))
        tx_gene[tx] = gene
        tx_seq[tx] = f[0]
    structs = {tx: tuple(sorted(segs)) for tx, segs in tx_segs.items()}
    return structs, tx_gene, tx_seq


def _group_by_gene(structs: dict, tx_gene: dict, tx_seq: dict) -> dict:
    """{(seq, start, end, gene_id): {tx_id: struct}} grouping transcripts under genes.

    The span is the union of all of that gene's transcripts' segments, so a gene with
    several isoforms of different extents is still one locus, not one per isoform. The
    gene id rides along as a 4th tuple element purely to keep the key unique — BRAKER3
    alone has 8 genes whose aggregate CDS span exactly coincides with another gene's
    (adjacent or nested single-exon predictions), and a bare (seq, start, end) key would
    silently drop one of every such pair. `_overlaps`/`match_by_overlap` only ever read
    indices 0-2, so the extra element does not change any matching behaviour.
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
    return _group_by_gene(*_transcript_structures(path, "CDS"))


def _representative(txs: dict) -> tuple:
    """The first transcript for this locus in file order — "the input/output model"."""
    return next(iter(txs.values()))


def _overlaps(a: tuple, b: tuple) -> bool:
    return a[0] == b[0] and a[1] <= b[2] and b[1] <= a[2]


def match_by_overlap(pred: dict, ref: dict) -> dict:
    """Pair predicted loci to reference loci by reciprocal overlap on the same sequence.

    Scored by Jaccard overlap (intersection / union of the two spans), not raw overlap
    length: TAIR10 has nested and overlapping gene models, and a gene that is fully
    engulfed by a much larger neighbour ties on raw overlap length with its true,
    equal-span match. Reciprocal overlap breaks that tie correctly — confirmed by the
    scorer's own self-test (score the reference against itself; every locus must land
    in preserved_correct, none in repaired) surfacing exactly this failure mode before
    the fix.

    A second, rarer tie remains even under Jaccard: a handful of TAIR10 loci share the
    exact same aggregate span as a neighbour — overlapping/antisense gene models with
    coincidentally identical extents (e.g. AT1G16860 and AT1G16858). Position alone
    cannot break that tie, so the representative structure does: whichever candidate's
    *first-in-file-order transcript* — the same one the transition table classifies on
    — equals `pred[p]`'s wins a tied score. This is not gene-id based (gene ids are not
    comparable across annotation tools), so it applies equally to GeMoMa/BRAKER3/EGAPx
    vs TAIR10, not just the same-tool self-test that exposed it. When the tied siblings
    truly have identical structure (not just identical span), the tie-break is moot —
    either one classifies the locus the same way — but when a sibling's span coincides
    while its structure differs by even one boundary, this is what keeps the locus from
    being scored against the wrong gene's model.
    """
    by_seq: dict = defaultdict(list)
    for span in ref:
        by_seq[span[0]].append(span)
    for k in by_seq:
        by_seq[k].sort(key=lambda s: s[1])
    pairs = {}
    for p in pred:
        p_len = p[2] - p[1]
        p_repr = _representative(pred[p])
        best, best_key = None, (0.0, 0)
        for r in by_seq.get(p[0], []):
            if r[1] > p[2]:
                break
            if _overlaps(p, r):
                ov = min(p[2], r[2]) - max(p[1], r[1])
                union = p_len + (r[2] - r[1]) - ov
                score = ov / union if union > 0 else 0.0
                bonus = 1 if p_repr == _representative(ref[r]) else 0
                key = (score, bonus)
                if key > best_key:
                    best, best_key = r, key
        if best:
            pairs[p] = best
    return pairs


def _transition_table(inp: dict, out: dict, ref: dict) -> dict:
    """The repaired/damaged/preserved/still-wrong table for one feature level."""
    in_pairs = match_by_overlap(inp, ref)
    if inp and ref and not in_pairs:
        raise RuntimeError(
            "no input locus overlaps any reference locus on the same sequence — this "
            "is the sequence-name-mismatch failure mode, check seqids match exactly "
            f"(input seqs={sorted({s[0] for s in inp})}, "
            f"reference seqs={sorted({s[0] for s in ref})})"
        )
    out_pairs = match_by_overlap(out, ref)
    out_by_ref: dict = defaultdict(dict)
    for o, r in out_pairs.items():
        out_by_ref[r].update(out[o])

    res = dict(loci_compared=0, preserved_correct=0, repaired=0, damaged=0, still_wrong=0,
               added_structures=0, added_matching_reference=0, loci_without_output=0)
    for i_span, r_span in in_pairs.items():
        ref_txs = ref[r_span]
        out_txs = out_by_ref.get(r_span)
        if not out_txs:
            res["loci_without_output"] += 1
            continue
        res["loci_compared"] += 1

        in_repr = _representative(inp[i_span])
        ref_repr = _representative(ref_txs)
        out_repr = _representative(out_txs)
        in_ok = in_repr == ref_repr
        out_ok = out_repr == ref_repr
        if in_ok and out_ok:
            res["preserved_correct"] += 1
        elif not in_ok and out_ok:
            res["repaired"] += 1
        elif in_ok and not out_ok:
            res["damaged"] += 1
        else:
            res["still_wrong"] += 1

        # Additions: what the output has beyond what the input already had — at this
        # locus, in full, not just its single representative — so a tool's own
        # pre-existing extra isoforms are never counted as something it "added".
        in_all = set(inp[i_span].values())
        out_all = set(out_txs.values())
        ref_all = set(ref_txs.values())
        added = out_all - in_all
        res["added_structures"] += len(added)
        res["added_matching_reference"] += len(added & ref_all)

    n = res["loci_compared"] or 1
    res["repaired_pct"] = round(100 * res["repaired"] / n, 2)
    res["damaged_pct"] = round(100 * res["damaged"] / n, 2)
    res["added_precision_pct"] = (
        round(100 * res["added_matching_reference"] / res["added_structures"], 1)
        if res["added_structures"] else None)
    return res


def _utr_signal(input_gff: Path) -> tuple[bool, str]:
    """Whether `input_gff` carries real CDS+UTR (exon) information.

    GeMoMa emits no exon rows at all; BRAKER3 emits exon rows identical to its CDS
    rows (no UTR was predicted). Both are detected here rather than assumed by tool
    name, so a future input is handled the same way its data says it should be.
    """
    cds, _, _ = _transcript_structures(input_gff, "CDS")
    exon, _, _ = _transcript_structures(input_gff, "exon")
    if not exon:
        return False, "input carries no exon rows: CDS+UTR agreement is not measurable"
    extended = sum(
        1 for tx, cseg in cds.items()
        if (eseg := exon.get(tx)) and
        (eseg[0][0] < cseg[0][0] or eseg[-1][1] > cseg[-1][1] or len(eseg) > len(cseg))
    )
    if extended == 0:
        return False, ("input exon coordinates equal CDS coordinates for every "
                        "transcript: no UTR was predicted")
    return True, ""


def _noncoding_feature_counts(path: Path) -> dict:
    """Count of each excluded non-coding feature type (`NONCODING_TYPES`) in `path`."""
    counts: Counter = Counter()
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) >= 3 and f[2] in NONCODING_TYPES:
            counts[f[2]] += 1
    return dict(counts)


def score(input_gff: Path, output_gff: Path, ref_gff: Path = REFERENCE) -> dict:
    inp, out, ref = (cds_structures(p) for p in (input_gff, output_gff, ref_gff))
    for name, d, p in (("input", inp, input_gff), ("output", out, output_gff),
                        ("reference", ref, ref_gff)):
        if not d:
            raise ValueError(f"no CDS-bearing transcripts found in {name} file {p} — "
                              "refusing to score, this would silently report all zeros")

    res = _transition_table(inp, out, ref)

    has_utr, reason = _utr_signal(input_gff)
    if has_utr:
        inp_u = _group_by_gene(*_transcript_structures(input_gff, "exon"))
        out_u = _group_by_gene(*_transcript_structures(output_gff, "exon"))
        ref_u = _group_by_gene(*_transcript_structures(ref_gff, "exon"))
        res["utr_level"] = _transition_table(inp_u, out_u, ref_u)
    else:
        res["utr_level"] = {"status": "N/A", "reason": reason}

    res["input_coding_loci"] = len(inp)
    res["input_noncoding_features_excluded"] = _noncoding_feature_counts(input_gff)
    return res


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
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    for label, p in (("--input", args.input), ("--output", args.output),
                      ("--reference", args.reference)):
        if not p.exists():
            print(f"MISSING {label}: {p}", file=sys.stderr)
            return 1

    res = score(args.input, args.output, args.reference)
    _print_result(res)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=1))
        print(f"written: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
