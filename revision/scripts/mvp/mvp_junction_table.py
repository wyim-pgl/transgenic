#!/usr/bin/env python3
"""Read-supported junction table for the B6 MVP scoring readout (exploratory).

Wraps the frozen B1 evidence scorer `revision/scripts/59_evidence_support.py` —
its SAM parsing, whole-read-group acceptance (`alignment_rejection`: primary
only, MAPQ >= 20, supplementary/chimeric and cross-contig/strand discarded) and
`transcript_strand` (A39 FLAG x ts:A composition) are REUSED, not reimplemented.
Support thresholds are the scorer's frozen `THRESHOLDS`: ONT 3 / PacBio 2
(A-HiFi counts under the PacBio threshold). Counts are READS (UMI status
unknown; never molecules).

Restriction scope (Track B must-fix 2, scope 1): a junction is tabulated only
when its intron span lies INSIDE a panel evaluation WINDOW (30,720 nt centred
on the gene, shared derivation `mvp_windows.evaluation_window`). The panel-GENE
restriction applies to predicted junctions and the recall denominator in
mvp_score.py, not here.

Output TSV columns:
    chrom  intron_start0  intron_end0  tx_strand  n_ONT  n_HiFi  supported
Intron endpoints are 0-based inclusive; `supported` is true iff
n_HiFi >= 2 or n_ONT >= 3. The complete-chain columns
(chain_support_single_read / chain_callable) are added when
59_evidence_support.py itself runs on the same BAMs (spec §2), not fabricated
here. A QC JSON sidecar records read accept/reject counts per input.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from transgenic.evidence import mvp_windows  # noqa: E402


def load_evidence_support():
    path = _SCRIPTS / "59_evidence_support.py"
    spec = importlib.util.spec_from_file_location("evidence_support_59", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ES = load_evidence_support()

# Frozen thresholds; a drifted scorer must stop this wrapper.
assert ES.THRESHOLDS["ONT"] == 3 and ES.THRESHOLDS["PacBio"] == 2, \
    "frozen 59_evidence_support THRESHOLDS drifted (expected ONT 3 / PacBio 2)"
SOURCE_THRESHOLDS = {"ONT": ES.THRESHOLDS["ONT"], "HiFi": ES.THRESHOLDS["PacBio"]}


class JunctionTableError(ValueError):
    pass


def parse_alignment_spec(text):
    parts = text.split(":", 2)
    if len(parts) != 3 or parts[0] not in SOURCE_THRESHOLDS or \
            parts[1] not in ("uf=true", "uf=false"):
        raise JunctionTableError(
            f"--alignment must be SOURCE:uf=BOOL:PATH with SOURCE in "
            f"{sorted(SOURCE_THRESHOLDS)}; got {text!r}")
    source, uf_text, path = parts
    uf = uf_text == "uf=true"
    if source == "HiFi" and uf:
        raise JunctionTableError("HiFi (splice:hq) has no -uf mode; uf must be false")
    return source, uf, Path(path)


def collect_junctions(alignment_path, uf, windows, samtools="samtools"):
    """Yield (junction_key, read_name) for accepted reads; return QC counters.

    junction_key = (chrom, intron_start0, intron_end0, tx_strand) with 0-based
    INCLUSIVE endpoints (parse_sam introns are 1-based inclusive; converted
    here once). Junctions with unknown strand or outside every panel window are
    counted in QC, never tabulated.
    """
    groups = defaultdict(list)
    empty_exon = set()
    for line in ES.sam_lines(str(alignment_path), samtools):
        try:
            aln = ES.parse_sam(line)
        except ES.EmptyExonError:
            # Author decision 2026-09-22: a record with a zero-reference-base
            # exon (e.g. CIGAR `N I N`) rejects its READ, not the run. Every
            # record of that read is dropped and the read is counted in QC.
            empty_exon.add(line.split("\t", 1)[0])
            continue
        if aln is not None:
            groups[aln.read].append(aln)
    qc = Counter()
    pairs = []
    for read in empty_exon:
        groups.pop(read, None)
    qc["rejected:empty_exon"] = len(empty_exon)
    for read in sorted(groups):
        records = groups[read]
        rejection = ES.alignment_rejection(records)
        if rejection:
            qc["rejected:" + rejection] += 1
            continue
        aln = next(r for r in records if not r.flag & (4 | 0x900))
        strand = ES.transcript_strand(aln, uf=uf)
        qc["accepted_reads"] += 1
        for start1, end1 in aln.introns:
            start0, end0 = start1 - 1, end1 - 1
            if strand is None:
                qc["junctions_strand_unknown"] += 1
                continue
            if not mvp_windows.junction_within_windows(aln.contig, start0, end0,
                                                       windows):
                qc["junctions_outside_panel_windows"] += 1
                continue
            qc["junctions_in_panel_windows"] += 1
            pairs.append(((aln.contig, start0, end0, strand), read))
    qc["raw_reads"] = len(groups) + len(empty_exon)
    return pairs, qc


def build_table(alignments, windows, samtools="samtools"):
    reads_by_source = {source: defaultdict(set) for source in SOURCE_THRESHOLDS}
    qc = {}
    seen = set()
    for source, uf, path in alignments:
        identity = Path(path).resolve()
        if identity in seen:
            raise JunctionTableError("duplicate alignment input")
        seen.add(identity)
        pairs, counters = collect_junctions(path, uf, windows, samtools)
        for key, read in pairs:
            reads_by_source[source][key].add((str(path), read))
        qc[f"{source}:{path}"] = dict(sorted(counters.items()))
    keys = sorted(set(reads_by_source["ONT"]) | set(reads_by_source["HiFi"]))
    rows = []
    for key in keys:
        n_ont = len(reads_by_source["ONT"].get(key, ()))
        n_hifi = len(reads_by_source["HiFi"].get(key, ()))
        supported = (n_hifi >= SOURCE_THRESHOLDS["HiFi"]
                     or n_ont >= SOURCE_THRESHOLDS["ONT"])
        chrom, start0, end0, strand = key
        rows.append({"chrom": chrom, "intron_start0": start0, "intron_end0": end0,
                     "tx_strand": strand, "n_ONT": n_ont, "n_HiFi": n_hifi,
                     "supported": supported})
    return rows, qc


HEADER = ("chrom", "intron_start0", "intron_end0", "tx_strand",
          "n_ONT", "n_HiFi", "supported")


def write_table(rows, out_path):
    with open(out_path, "w") as f:
        f.write("\t".join(HEADER) + "\n")
        for row in rows:
            f.write("\t".join(str(row[c]).lower() if c == "supported"
                              else str(row[c]) for c in HEADER) + "\n")


def read_table(path):
    """Load a table written by write_table, fail-closed on schema drift."""
    rows = []
    with open(path) as f:
        header = tuple(f.readline().rstrip("\n").split("\t"))
        if header != HEADER:
            raise JunctionTableError(f"junction table header mismatch: {header!r}")
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(HEADER):
                raise JunctionTableError("junction table row width mismatch")
            if fields[6] not in ("true", "false"):
                raise JunctionTableError("supported must be true/false")
            rows.append({"chrom": fields[0], "intron_start0": int(fields[1]),
                         "intron_end0": int(fields[2]), "tx_strand": fields[3],
                         "n_ONT": int(fields[4]), "n_HiFi": int(fields[5]),
                         "supported": fields[6] == "true"})
    keys = set()
    for row in rows:
        key = tuple(row[c] for c in HEADER[:4])
        if (key in keys or row['tx_strand'] not in ('+', '-') or
                not 0 <= row['intron_start0'] <= row['intron_end0'] or
                min(row['n_ONT'], row['n_HiFi']) < 0 or
                row['supported'] != (row['n_ONT'] >= 3 or row['n_HiFi'] >= 2)):
            raise JunctionTableError('invalid, duplicate, or inconsistent junction row')
        keys.add(key)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", action="append", required=True,
                        metavar="SOURCE:uf=BOOL:PATH",
                        help="repeatable; SOURCE is ONT or HiFi; uf must state "
                             "the audited -uf decision explicitly")
    parser.add_argument("--panels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--qc-out", required=True, type=Path)
    parser.add_argument("--samtools", default="samtools")
    args = parser.parse_args(argv)

    expected = mvp_windows.PANEL_SHA256
    loci = mvp_windows.load_panel(args.panels, expected_sha256=expected)
    windows = mvp_windows.windows_by_chromosome(loci)
    alignments = [parse_alignment_spec(text) for text in args.alignment]
    rows, qc = build_table(alignments, windows, args.samtools)
    write_table(rows, args.out)
    qc_report = {
        "panel_sha256": expected,
        "panel_sha256_is_frozen": expected == mvp_windows.PANEL_SHA256,
        "thresholds": SOURCE_THRESHOLDS,
        "counts_are": "reads (UMI status unknown; never molecules)",
        "scorer": "59_evidence_support.py (sha256 "
                  + hashlib.sha256((_SCRIPTS / "59_evidence_support.py")
                                   .read_bytes()).hexdigest() + ")",
        "restriction": "junction contained in a 30,720-nt panel evaluation window",
        "per_input": qc,
        "junctions": len(rows),
        "supported_junctions": sum(r["supported"] for r in rows),
    }
    args.qc_out.write_text(json.dumps(qc_report, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
